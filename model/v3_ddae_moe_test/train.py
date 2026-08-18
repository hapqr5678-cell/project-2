"""Train the warm-started residual decoder MoE experiment.

The initial model exactly reproduces the v2_ddae_fsce checkpoint.  Only the
router and zero-initialized residual experts train for the first 50 epochs;
the complete model is then fine-tuned at a lower learning rate.  The best
validation-deviance checkpoint, including the epoch-0 baseline, is retained.
"""

from __future__ import annotations

import argparse
import copy
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.abspath(f"{os.path.dirname(__file__)}/../.."))
from ae import (  # noqa: E402
    MOE_SCALE,
    N_EXPERTS,
    ROUTER_TEMPERATURE,
    Patches,
    ResidualDecoderMoEAE,
    build_fsce_graph,
    corrupt,
    fsce_loss,
    load_baseline_checkpoint,
    moe_balance_loss,
    poisson_deviance,
    poisson_nll,
)
from config.dataset import PATCHES, ensure_patches, result  # noqa: E402
from config.train_log import open_log  # noqa: E402


VERSION = "v3_ddae_moe_test"
BASELINE_VERSION = "v2_ddae_fsce"
BASELINE_CKPT = result(BASELINE_VERSION, "ae.pt")
OUT = result(VERSION, "latents.npz")
CKPT = result(VERSION, "ae.pt")

LATENT_DIM = 2
EPOCHS = 1000
BATCH = 256
LR_MOE = 3e-4
LR_FINETUNE = 1e-4
MIN_LR = 1e-5
WEIGHT_DECAY = 1e-6
FREEZE_BASE_EPOCHS = 50
VAL_FRAC = 0.1
SEED = 0
GRAD_CLIP = 5.0

N_NEIGHBORS = 15
GRAPH_METRIC = "euclidean"
EDGE_BATCH = 256
LAMBDA_FSCE = 0.005
WARMUP_EPOCHS = 200

NOISE_P = 0.3
NOISE_MODE = "thinning"
LAMBDA_BALANCE = 0.0
REPORT_EVERY = 5

device = (
    "mps"
    if torch.backends.mps.is_available()
    else "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument(
        "--baseline-checkpoint", default=BASELINE_CKPT
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Run the experiment without writing checkpoint or latents.",
    )
    return parser.parse_args()


def evaluate(model, data, indices, log_lam_null=None):
    model.eval()
    nll_values, deviance_values, null_values = [], [], []
    with torch.no_grad():
        for start in range(0, len(indices), BATCH):
            batch = indices[start : start + BATCH]
            x = data.agg(batch).to(device)
            _, log_lam = model(x)
            nll_values.append(poisson_nll(log_lam, x).cpu())
            deviance_values.append(poisson_deviance(log_lam, x).cpu())
            if log_lam_null is not None:
                null_values.append(poisson_deviance(
                    log_lam_null.expand(len(batch), -1), x
                ).cpu())
    nll = torch.cat(nll_values).mean().item()
    deviance = torch.cat(deviance_values).mean().item()
    null_deviance = (
        torch.cat(null_values).mean().item() if null_values else None
    )
    return nll, deviance, null_deviance


def infer_all(model, data):
    model.eval()
    z_values, error_values, gate_values = [], [], []
    with torch.no_grad():
        for start in range(0, data.n, BATCH):
            indices = torch.arange(start, min(start + BATCH, data.n))
            x = data.agg(indices).to(device)
            z, log_lam, gates = model.forward_with_gates(x)
            z_values.append(z.cpu())
            error_values.append(poisson_deviance(log_lam, x).cpu())
            gate_values.append(gates.cpu())
    return (
        torch.cat(z_values).numpy(),
        torch.cat(error_values).numpy(),
        torch.cat(gate_values).numpy(),
    )


def run(
    data,
    train_idx,
    val_idx,
    edge_i,
    edge_j,
    edge_weight,
    fsce_a,
    fsce_b,
    epochs,
    baseline_checkpoint,
    log,
):
    torch.manual_seed(SEED)
    model = ResidualDecoderMoEAE(
        latent_dim=LATENT_DIM,
        n_experts=N_EXPERTS,
        router_temperature=ROUTER_TEMPERATURE,
        moe_scale=MOE_SCALE,
    ).to(device)
    missing = load_baseline_checkpoint(
        model, baseline_checkpoint, map_location=device
    )
    log(f"loaded baseline {baseline_checkpoint}; new MoE tensors={len(missing)}")

    log_lam_null = (
        data.agg(train_idx)
        .mean(dim=0, keepdim=True)
        .clamp_min(1e-8)
        .log()
        .to(device)
    )

    # The zero-initialized experts make this the exact baseline validation score.
    initial_nll, initial_dev, initial_null = evaluate(
        model, data, val_idx, log_lam_null
    )
    initial_explained = 1.0 - initial_dev / initial_null
    log(
        f"epoch    0 | baseline_val_nll {initial_nll:.5f} | "
        f"val_dev {initial_dev:.5f} | expl_dev {initial_explained:.5f}"
    )
    best_dev = initial_dev
    best_epoch = 0
    best_state = copy.deepcopy(model.state_dict())

    model.set_base_trainable(False)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=LR_MOE, weight_decay=WEIGHT_DECAY
    )
    scheduler = None
    n_edges = len(edge_i)

    for epoch in range(epochs):
        try:
            if epoch == FREEZE_BASE_EPOCHS:
                model.set_base_trainable(True)
                for group in optimizer.param_groups:
                    group["lr"] = LR_FINETUNE
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer,
                    T_max=max(epochs - FREEZE_BASE_EPOCHS, 1),
                    eta_min=MIN_LR,
                )
                log(
                    f"unfroze baseline at epoch {epoch + 1}; "
                    f"lr={LR_FINETUNE}"
                )

            model.train()
            fsce_weight_t = LAMBDA_FSCE * min(
                1.0, (epoch + 1) / max(WARMUP_EPOCHS, 1)
            )
            generator = torch.Generator().manual_seed(SEED + epoch)
            permutation = train_idx[
                torch.randperm(len(train_idx), generator=generator)
            ]
            total_nll = 0.0
            total_fsce = 0.0
            total_balance = 0.0
            gate_sum = torch.zeros(N_EXPERTS, device=device)
            gate_count = 0

            for start in range(0, len(permutation), BATCH):
                batch = permutation[start : start + BATCH]
                x_clean = data.agg(batch)
                x_noisy = corrupt(
                    x_clean, NOISE_P, NOISE_MODE, generator=generator
                )
                x_clean = x_clean.to(device)
                x_noisy = x_noisy.to(device)
                _, log_lam, gates = model.forward_with_gates(x_noisy)
                reconstruction = poisson_nll(log_lam, x_clean).mean()

                sampled = torch.randint(
                    0, n_edges, (EDGE_BATCH,), generator=generator
                )
                positive_i = edge_i[sampled]
                positive_j = edge_j[sampled]
                positive_weight = edge_weight[sampled]
                negative_i = torch.randint(
                    0, len(train_idx), (EDGE_BATCH,), generator=generator
                )
                negative_j = torch.randint(
                    0, len(train_idx), (EDGE_BATCH,), generator=generator
                )
                # FSCE graph indices are local to train_idx; map back to patches.
                left_patch = train_idx[torch.cat([positive_i, negative_i])]
                right_patch = train_idx[torch.cat([positive_j, negative_j])]
                left_count = corrupt(
                    data.agg(left_patch),
                    NOISE_P,
                    NOISE_MODE,
                    generator=generator,
                ).to(device)
                right_count = corrupt(
                    data.agg(right_patch),
                    NOISE_P,
                    NOISE_MODE,
                    generator=generator,
                ).to(device)
                pair_weight = torch.cat([
                    positive_weight, torch.zeros(EDGE_BATCH)
                ]).to(device)
                z_left = model.encode(left_count)
                z_right = model.encode(right_count)
                fuzzy = fsce_loss(
                    z_left, z_right, pair_weight, fsce_a, fsce_b
                ).mean()
                balance = moe_balance_loss(gates)

                loss = (
                    reconstruction
                    + fsce_weight_t * fuzzy
                    + LAMBDA_BALANCE * balance
                )
                if not torch.isfinite(loss):
                    raise FloatingPointError(
                        f"non-finite loss at epoch {epoch + 1}"
                    )
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                optimizer.step()

                batch_size = len(batch)
                total_nll += reconstruction.item() * batch_size
                total_fsce += fuzzy.item() * batch_size
                total_balance += balance.item() * batch_size
                gate_sum += gates.detach().sum(dim=0)
                gate_count += batch_size

            if scheduler is not None:
                scheduler.step()

            should_report = (
                (epoch + 1) % REPORT_EVERY == 0
                or epoch == 0
                or epoch + 1 == epochs
            )
            if should_report:
                val_nll, val_dev, val_null = evaluate(
                    model, data, val_idx, log_lam_null
                )
                explained = 1.0 - val_dev / val_null
                improved = val_dev < best_dev
                if improved:
                    best_dev = val_dev
                    best_epoch = epoch + 1
                    best_state = copy.deepcopy(model.state_dict())
                usage = ",".join(
                    f"{value:.3f}"
                    for value in (gate_sum / gate_count).tolist()
                )
                current_lr = optimizer.param_groups[0]["lr"]
                marker = " | BEST" if improved else ""
                log(
                    f"epoch {epoch + 1:4d} | "
                    f"train_nll {total_nll / len(train_idx):.5f} | "
                    f"train_fsce {total_fsce / len(train_idx):.5f} | "
                    f"moe_balance {total_balance / len(train_idx):.6f} | "
                    f"gate_usage [{usage}] | lr {current_lr:.2e} | "
                    f"val_nll {val_nll:.5f} | val_dev {val_dev:.5f} | "
                    f"expl_dev {explained:.5f}{marker}"
                )
        except KeyboardInterrupt:
            log(
                f"[interrupted] epoch {epoch + 1}; restoring best checkpoint"
            )
            break

    model.load_state_dict(best_state)
    log(f"restored best epoch {best_epoch}, val_dev={best_dev:.5f}")
    return model, best_epoch, best_dev


def main():
    args = parse_args()
    if args.epochs <= 0:
        raise ValueError("--epochs must be positive")
    if not os.path.exists(args.baseline_checkpoint):
        raise FileNotFoundError(
            f"baseline checkpoint not found: {args.baseline_checkpoint}"
        )

    log = open_log(VERSION, {
        "BASELINE_VERSION": BASELINE_VERSION,
        "LATENT_DIM": LATENT_DIM,
        "EPOCHS": args.epochs,
        "BATCH": BATCH,
        "LR_MOE": LR_MOE,
        "LR_FINETUNE": LR_FINETUNE,
        "MIN_LR": MIN_LR,
        "WEIGHT_DECAY": WEIGHT_DECAY,
        "FREEZE_BASE_EPOCHS": FREEZE_BASE_EPOCHS,
        "VAL_FRAC": VAL_FRAC,
        "SEED": SEED,
        "N_NEIGHBORS": N_NEIGHBORS,
        "GRAPH_METRIC": GRAPH_METRIC,
        "EDGE_BATCH": EDGE_BATCH,
        "LAMBDA_FSCE": LAMBDA_FSCE,
        "WARMUP_EPOCHS": WARMUP_EPOCHS,
        "NOISE_P": NOISE_P,
        "NOISE_MODE": NOISE_MODE,
        "N_EXPERTS": N_EXPERTS,
        "ROUTER_TEMPERATURE": ROUTER_TEMPERATURE,
        "MOE_SCALE": MOE_SCALE,
        "LAMBDA_BALANCE": LAMBDA_BALANCE,
    })

    ensure_patches()
    data = Patches(PATCHES)
    generator = torch.Generator().manual_seed(SEED)
    permutation = torch.randperm(data.n, generator=generator)
    n_validation = int(data.n * VAL_FRAC)
    val_idx = permutation[:n_validation]
    train_idx = permutation[n_validation:]

    # Build FSCE only from training patches; returned indices are train-local.
    x_train = np.log1p(data.agg(train_idx).numpy())
    edge_i, edge_j, edge_weight, fsce_a, fsce_b = build_fsce_graph(
        x_train, n_neighbors=N_NEIGHBORS, metric=GRAPH_METRIC
    )
    log(
        f"{data.n} patches, train={len(train_idx)}, val={len(val_idx)}, "
        f"device={device}, train-only FSCE edges={len(edge_i)}"
    )

    model, best_epoch, best_dev = run(
        data,
        train_idx,
        val_idx,
        edge_i,
        edge_j,
        edge_weight,
        fsce_a,
        fsce_b,
        epochs=args.epochs,
        baseline_checkpoint=args.baseline_checkpoint,
        log=log,
    )
    z, error, gates = infer_all(model, data)
    if not all(np.isfinite(value).all() for value in (z, error, gates)):
        raise FloatingPointError("final inference contains NaN or infinity")

    if not args.no_save:
        checkpoint = {
            "model_state": model.state_dict(),
            "best_epoch": best_epoch,
            "best_val_deviance": best_dev,
            "baseline_checkpoint": os.path.abspath(args.baseline_checkpoint),
            "latent_dim": LATENT_DIM,
            "n_experts": N_EXPERTS,
            "router_temperature": ROUTER_TEMPERATURE,
            "moe_scale": MOE_SCALE,
        }
        torch.save(checkpoint, CKPT)
        is_train = np.zeros(data.n, dtype=bool)
        is_train[train_idx.numpy()] = True
        np.savez(
            OUT,
            n_poi=data.n_poi,
            lat=data.lat,
            lon=data.lon,
            z=z,
            err=error,
            gates=gates,
            expert_id=gates.argmax(axis=1),
            is_train=is_train,
            is_val=~is_train,
            best_epoch=np.int64(best_epoch),
            best_val_deviance=np.float32(best_dev),
        )
        log(f"saved checkpoint: {CKPT}")
        log(f"saved latents: {OUT}")


if __name__ == "__main__":
    main()
