"""Train a strict-2D baseline, then competitive residual Poisson experts.

Stage A learns a factorized single decoder and restores its best checkpoint.
Stage B freezes that baseline while experts learn fixed latent clusters, then
alternates hard assignments and low-LR joint fine-tuning. The final checkpoint
is the best of base, hard routing, and soft routing on validation deviance.
"""

from __future__ import annotations

import argparse
import copy
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.abspath(f"{os.path.dirname(__file__)}/../.."))
from ae import (  # noqa: E402
    EXPERT_INIT_NOISE,
    MODE_BASE,
    MODE_HARD,
    MODE_NAMES,
    MODE_SOFT,
    N_EXPERTS,
    ROUTER_TEMPERATURE,
    MLPAE,
    Patches,
    build_fsce_graph,
    corrupt,
    expert_deviances,
    expert_nlls,
    fsce_loss,
    poisson_deviance,
    poisson_nll,
)
from config.dataset import N_CAT, PATCHES, ensure_patches, result  # noqa: E402
from config.train_log import open_log  # noqa: E402


VERSION = "v3_ddae_moe_test2"
OUT = result(VERSION, "latents.npz")
CKPT = result(VERSION, "ae.pt")

LATENT_DIM = 2
BASE_EPOCHS = 600
MOE_EPOCHS = 500
SPECIALIZE_EPOCHS = 75
E_STEP_EVERY = 5
BATCH = 256

LR_BASE = 1e-3
LR_BACKBONE_FINETUNE = 1e-4
LR_EXPERT = 3e-4
LR_ROUTER = 1e-4
MIN_LR_FACTOR = 0.1
WEIGHT_DECAY = 1e-6
VAL_FRAC = 0.1
SEED = 0
GRAD_CLIP = 5.0

N_NEIGHBORS = 15
GRAPH_METRIC = "euclidean"
EDGE_BATCH = 256
LAMBDA_FSCE = 0.005
WARMUP_EPOCHS = 200
LAMBDA_ROUTER = 0.1
LAMBDA_RESIDUAL = 1e-4
RESPONSIBILITY_ROUTER_WEIGHT = 0.25
MIN_EXPERT_FRACTION = 0.10

NOISE_P = 0.3
NOISE_MODE = "thinning"
REPORT_EVERY = 5
SCHEDULER_PATIENCE = 12
BASE_EARLY_STOP_PATIENCE = 250
MOE_EARLY_STOP_PATIENCE = 250

device = (
    "mps"
    if torch.backends.mps.is_available()
    else "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-epochs", type=int, default=BASE_EPOCHS)
    parser.add_argument("--moe-epochs", type=int, default=MOE_EPOCHS)
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Run without writing ae.pt or latents.npz.",
    )
    return parser.parse_args()


def _format_vector(values, precision=5):
    return "[" + ",".join(f"{value:.{precision}f}" for value in values) + "]"


def _set_backbone_trainable(model, trainable):
    modules = [model.encoder, model.total_decoder, model.composition_decoder]
    for module in modules:
        for parameter in module.parameters():
            parameter.requires_grad_(trainable)


def _backbone_parameters(model):
    modules = [model.encoder, model.total_decoder, model.composition_decoder]
    return [parameter for module in modules for parameter in module.parameters()]


def _sample_fsce(
    model,
    data,
    train_idx,
    edge_i,
    edge_j,
    edge_weight,
    fsce_a,
    fsce_b,
    generator,
):
    sampled = torch.randint(
        0, len(edge_i), (EDGE_BATCH,), generator=generator
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
    left_patch = train_idx[torch.cat([positive_i, negative_i])]
    right_patch = train_idx[torch.cat([positive_j, negative_j])]
    left_count = corrupt(
        data.agg(left_patch), NOISE_P, NOISE_MODE, generator=generator
    ).to(device)
    right_count = corrupt(
        data.agg(right_patch), NOISE_P, NOISE_MODE, generator=generator
    ).to(device)
    pair_weight = torch.cat(
        [positive_weight, torch.zeros(EDGE_BATCH)]
    ).to(device)
    return fsce_loss(
        model.encode(left_count),
        model.encode(right_count),
        pair_weight,
        fsce_a,
        fsce_b,
    ).mean()


def evaluation_metrics(model, data, indices, null_log_lam):
    model.eval()
    values = {
        "base_nll": [],
        "base_dev": [],
        "hard_dev": [],
        "soft_dev": [],
        "expert_dev": [],
        "oracle_dev": [],
        "gates": [],
        "gate_entropy": [],
        "residual_size": [],
        "null_dev": [],
    }
    with torch.no_grad():
        for start in range(0, len(indices), BATCH):
            batch = indices[start : start + BATCH]
            x = data.agg(batch).to(device)
            _, outputs = model.forward_with_experts(x)
            per_expert = expert_deviances(outputs["expert_log_lam"], x)
            values["base_nll"].append(
                poisson_nll(outputs["base_log_lam"], x).cpu()
            )
            values["base_dev"].append(
                poisson_deviance(outputs["base_log_lam"], x).cpu()
            )
            values["hard_dev"].append(
                poisson_deviance(outputs["hard_log_lam"], x).cpu()
            )
            values["soft_dev"].append(
                poisson_deviance(outputs["soft_log_lam"], x).cpu()
            )
            values["expert_dev"].append(per_expert.cpu())
            values["oracle_dev"].append(per_expert.min(dim=1).values.cpu())
            values["gates"].append(outputs["gates"].cpu())
            values["gate_entropy"].append(
                -(
                    outputs["gates"]
                    * outputs["gates"].clamp_min(1e-8).log()
                ).sum(dim=1).cpu()
            )
            values["residual_size"].append(
                outputs["residuals"].abs().mean(dim=(1, 2)).cpu()
            )
            values["null_dev"].append(
                poisson_deviance(
                    null_log_lam.expand(len(batch), -1), x
                ).cpu()
            )

    return {
        "base_nll": torch.cat(values["base_nll"]).mean().item(),
        "base_dev": torch.cat(values["base_dev"]).mean().item(),
        "hard_dev": torch.cat(values["hard_dev"]).mean().item(),
        "soft_dev": torch.cat(values["soft_dev"]).mean().item(),
        "expert_dev": torch.cat(values["expert_dev"]).mean(dim=0).numpy(),
        "oracle_dev": torch.cat(values["oracle_dev"]).mean().item(),
        "gate_usage": torch.cat(values["gates"]).mean(dim=0).numpy(),
        "gate_entropy": torch.cat(values["gate_entropy"]).mean().item(),
        "residual_size": torch.cat(values["residual_size"]).mean().item(),
        "null_dev": torch.cat(values["null_dev"]).mean().item(),
    }


def train_baseline(
    model,
    data,
    train_idx,
    val_idx,
    edge_i,
    edge_j,
    edge_weight,
    fsce_a,
    fsce_b,
    null_log_lam,
    epochs,
    log,
):
    _set_backbone_trainable(model, True)
    optimizer = torch.optim.Adam(
        _backbone_parameters(model), lr=LR_BASE, weight_decay=WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=SCHEDULER_PATIENCE,
        min_lr=LR_BASE * MIN_LR_FACTOR,
    )
    best_dev = float("inf")
    best_epoch = 0
    best_state = None
    last_improvement = 0

    for epoch in range(epochs):
        try:
            model.train()
            fsce_weight = LAMBDA_FSCE * min(
                1.0, (epoch + 1) / max(WARMUP_EPOCHS, 1)
            )
            generator = torch.Generator().manual_seed(SEED + epoch)
            permutation = train_idx[
                torch.randperm(len(train_idx), generator=generator)
            ]
            total_nll = 0.0
            total_fsce = 0.0

            for start in range(0, len(permutation), BATCH):
                batch = permutation[start : start + BATCH]
                clean_x = data.agg(batch)
                noisy_x = corrupt(
                    clean_x, NOISE_P, NOISE_MODE, generator=generator
                )
                clean_x = clean_x.to(device)
                noisy_x = noisy_x.to(device)
                _, base_log_lam = model.forward_base(noisy_x)
                reconstruction = poisson_nll(base_log_lam, clean_x).mean()
                fuzzy = _sample_fsce(
                    model,
                    data,
                    train_idx,
                    edge_i,
                    edge_j,
                    edge_weight,
                    fsce_a,
                    fsce_b,
                    generator,
                )
                loss = reconstruction + fsce_weight * fuzzy
                if not torch.isfinite(loss):
                    raise FloatingPointError(
                        f"non-finite baseline loss at epoch {epoch + 1}"
                    )
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    _backbone_parameters(model), GRAD_CLIP
                )
                optimizer.step()
                total_nll += reconstruction.item() * len(batch)
                total_fsce += fuzzy.item() * len(batch)

            should_report = (
                (epoch + 1) % REPORT_EVERY == 0
                or epoch == 0
                or epoch + 1 == epochs
            )
            if should_report:
                metrics = evaluation_metrics(model, data, val_idx, null_log_lam)
                val_dev = metrics["base_dev"]
                scheduler.step(val_dev)
                improved = val_dev < best_dev
                if improved:
                    best_dev = val_dev
                    best_epoch = epoch + 1
                    best_state = copy.deepcopy(model.state_dict())
                    last_improvement = epoch + 1
                explained = 1.0 - val_dev / metrics["null_dev"]
                marker = " | BEST" if improved else ""
                log(
                    f"base epoch {epoch + 1:4d} | "
                    f"train_nll {total_nll / len(train_idx):.5f} | "
                    f"train_fsce {total_fsce / len(train_idx):.5f} | "
                    f"lr {optimizer.param_groups[0]['lr']:.2e} | "
                    f"val_nll {metrics['base_nll']:.5f} | "
                    f"val_dev {val_dev:.5f} | expl_dev {explained:.5f}{marker}"
                )
                if epoch + 1 - last_improvement >= BASE_EARLY_STOP_PATIENCE:
                    log(
                        f"base early stop at epoch {epoch + 1}; no improvement "
                        f"for {BASE_EARLY_STOP_PATIENCE} epochs"
                    )
                    break
        except KeyboardInterrupt:
            log(f"[interrupted] base epoch {epoch + 1}")
            break

    if best_state is None:
        raise RuntimeError("baseline ended before producing a checkpoint")
    model.load_state_dict(best_state)
    model.set_inference_mode(MODE_BASE)
    log(f"restored baseline epoch {best_epoch}, val_dev={best_dev:.5f}")
    return best_epoch, best_dev


def initial_cluster_assignments(model, data, train_idx):
    model.eval()
    latent_values = []
    with torch.no_grad():
        for start in range(0, len(train_idx), BATCH):
            batch = train_idx[start : start + BATCH]
            latent_values.append(model.encode(data.agg(batch).to(device)).cpu())
    latent = torch.cat(latent_values).numpy()
    latent = (latent - latent.mean(axis=0)) / np.maximum(
        latent.std(axis=0), 1e-6
    )
    labels = KMeans(
        n_clusters=N_EXPERTS, random_state=SEED, n_init=20
    ).fit_predict(latent)
    return torch.from_numpy(labels).long()


def hard_em_assignments(model, data, train_idx):
    model.eval()
    labels = []
    with torch.no_grad():
        for start in range(0, len(train_idx), BATCH):
            batch = train_idx[start : start + BATCH]
            x = data.agg(batch).to(device)
            _, outputs = model.forward_with_experts(x)
            per_expert_nll = expert_nlls(outputs["expert_log_lam"], x)
            joint_score = (
                -N_CAT * per_expert_nll
                + RESPONSIBILITY_ROUTER_WEIGHT
                * outputs["gates"].clamp_min(1e-8).log()
            )
            labels.append(joint_score.argmax(dim=1).cpu())
    return torch.cat(labels)


def _assignment_lookup(data_size, train_idx, labels):
    lookup = torch.full((data_size,), -1, dtype=torch.long)
    lookup[train_idx] = labels
    return lookup


def _make_moe_optimizer(model, include_backbone):
    groups = []
    if include_backbone:
        groups.append(
            {"params": _backbone_parameters(model), "lr": LR_BACKBONE_FINETUNE}
        )
    groups.extend(
        [
            {"params": model.residual_experts.parameters(), "lr": LR_EXPERT},
            {"params": model.router.parameters(), "lr": LR_ROUTER},
        ]
    )
    return torch.optim.Adam(groups, weight_decay=WEIGHT_DECAY)


def train_competitive_moe(
    model,
    data,
    train_idx,
    val_idx,
    edge_i,
    edge_j,
    edge_weight,
    fsce_a,
    fsce_b,
    null_log_lam,
    epochs,
    baseline_epoch,
    baseline_dev,
    log,
):
    fixed_labels = initial_cluster_assignments(model, data, train_idx)
    labels = fixed_labels.clone()
    counts = torch.bincount(labels, minlength=N_EXPERTS)
    log(f"initial latent-cluster assignments: {counts.tolist()}")

    _set_backbone_trainable(model, False)
    optimizer = _make_moe_optimizer(model, include_backbone=False)
    scheduler = None

    best_dev = baseline_dev
    best_epoch = 0
    best_mode = MODE_BASE
    model.set_inference_mode(MODE_BASE)
    best_state = copy.deepcopy(model.state_dict())
    last_improvement = 0

    for epoch in range(epochs):
        try:
            if epoch == SPECIALIZE_EPOCHS:
                _set_backbone_trainable(model, True)
                optimizer = _make_moe_optimizer(model, include_backbone=True)
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer,
                    mode="min",
                    factor=0.5,
                    patience=SCHEDULER_PATIENCE,
                    min_lr=[
                        LR_BACKBONE_FINETUNE * MIN_LR_FACTOR,
                        LR_EXPERT * MIN_LR_FACTOR,
                        LR_ROUTER * MIN_LR_FACTOR,
                    ],
                )
                log("unfroze baseline backbone for low-LR joint fine-tuning")

            if epoch >= SPECIALIZE_EPOCHS and (
                epoch == SPECIALIZE_EPOCHS or epoch % E_STEP_EVERY == 0
            ):
                candidate_labels = hard_em_assignments(model, data, train_idx)
                candidate_counts = torch.bincount(
                    candidate_labels, minlength=N_EXPERTS
                )
                if candidate_counts.min().item() / len(train_idx) >= MIN_EXPERT_FRACTION:
                    labels = candidate_labels
                else:
                    log(
                        "rejected collapsed E-step assignments "
                        f"{candidate_counts.tolist()}; keeping previous assignments"
                    )

            assignment_lookup = _assignment_lookup(data.n, train_idx, labels)
            model.train()
            generator = torch.Generator().manual_seed(
                SEED + BASE_EPOCHS + epoch
            )
            permutation = train_idx[
                torch.randperm(len(train_idx), generator=generator)
            ]
            total_nll = 0.0
            total_router = 0.0
            total_fsce = 0.0
            total_residual = 0.0

            for start in range(0, len(permutation), BATCH):
                batch = permutation[start : start + BATCH]
                assignment = assignment_lookup[batch].to(device)
                clean_x = data.agg(batch)
                noisy_x = corrupt(
                    clean_x, NOISE_P, NOISE_MODE, generator=generator
                )
                clean_x = clean_x.to(device)
                noisy_x = noisy_x.to(device)
                z, outputs = model.forward_with_experts(noisy_x)
                row = torch.arange(len(batch), device=device)
                selected_log_lam = outputs["expert_log_lam"][row, assignment]
                reconstruction = poisson_nll(selected_log_lam, clean_x).mean()
                router_loss = F.cross_entropy(
                    model.router(z.detach()) / ROUTER_TEMPERATURE,
                    assignment,
                )
                residual_penalty = outputs["residuals"].pow(2).mean()

                if epoch >= SPECIALIZE_EPOCHS:
                    fuzzy = _sample_fsce(
                        model,
                        data,
                        train_idx,
                        edge_i,
                        edge_j,
                        edge_weight,
                        fsce_a,
                        fsce_b,
                        generator,
                    )
                else:
                    fuzzy = torch.zeros((), device=device)

                loss = (
                    reconstruction
                    + LAMBDA_ROUTER * router_loss
                    + LAMBDA_RESIDUAL * residual_penalty
                    + LAMBDA_FSCE * fuzzy
                )
                if not torch.isfinite(loss):
                    raise FloatingPointError(
                        f"non-finite MoE loss at epoch {epoch + 1}"
                    )
                optimizer.zero_grad()
                loss.backward()
                trainable_parameters = [
                    parameter for parameter in model.parameters()
                    if parameter.requires_grad
                ]
                torch.nn.utils.clip_grad_norm_(trainable_parameters, GRAD_CLIP)
                optimizer.step()

                batch_size = len(batch)
                total_nll += reconstruction.item() * batch_size
                total_router += router_loss.item() * batch_size
                total_fsce += fuzzy.item() * batch_size
                total_residual += residual_penalty.item() * batch_size

            should_report = (
                (epoch + 1) % REPORT_EVERY == 0
                or epoch == 0
                or epoch + 1 == epochs
            )
            if should_report:
                metrics = evaluation_metrics(model, data, val_idx, null_log_lam)
                candidates = {
                    MODE_BASE: metrics["base_dev"],
                    MODE_HARD: metrics["hard_dev"],
                    MODE_SOFT: metrics["soft_dev"],
                }
                candidate_mode = min(candidates, key=candidates.get)
                candidate_dev = candidates[candidate_mode]
                improved = candidate_dev < best_dev
                if improved:
                    best_dev = candidate_dev
                    best_epoch = epoch + 1
                    best_mode = candidate_mode
                    best_state = copy.deepcopy(model.state_dict())
                    last_improvement = epoch + 1
                if scheduler is not None:
                    scheduler.step(candidate_dev)

                assignment_counts = torch.bincount(
                    labels, minlength=N_EXPERTS
                ).tolist()
                marker = " | BEST" if improved else ""
                phase = "cluster" if epoch < SPECIALIZE_EPOCHS else "hard-em"
                log(
                    f"moe epoch {epoch + 1:4d} ({phase}) | "
                    f"train_nll {total_nll / len(train_idx):.5f} | "
                    f"router_ce {total_router / len(train_idx):.5f} | "
                    f"train_fsce {total_fsce / len(train_idx):.5f} | "
                    f"residual_l2 {total_residual / len(train_idx):.5f} | "
                    f"assign {assignment_counts} | "
                    f"base_dev {metrics['base_dev']:.5f} | "
                    f"hard_dev {metrics['hard_dev']:.5f} | "
                    f"soft_dev {metrics['soft_dev']:.5f} | "
                    f"expert_dev {_format_vector(metrics['expert_dev'])} | "
                    f"oracle_dev {metrics['oracle_dev']:.5f} | "
                    f"gate_usage {_format_vector(metrics['gate_usage'])} | "
                    f"gate_entropy {metrics['gate_entropy']:.5f} | "
                    f"residual_size {metrics['residual_size']:.5f}{marker}"
                )
                if (
                    epoch >= SPECIALIZE_EPOCHS
                    and epoch + 1 - last_improvement >= MOE_EARLY_STOP_PATIENCE
                ):
                    log(
                        f"MoE early stop at epoch {epoch + 1}; no improvement "
                        f"for {MOE_EARLY_STOP_PATIENCE} epochs"
                    )
                    break
        except KeyboardInterrupt:
            log(f"[interrupted] MoE epoch {epoch + 1}")
            break

    model.load_state_dict(best_state)
    model.set_inference_mode(best_mode)
    log(
        f"selected {MODE_NAMES[best_mode]} checkpoint: "
        f"base_epoch={baseline_epoch}, moe_epoch={best_epoch}, "
        f"val_dev={best_dev:.5f}"
    )
    return best_epoch, best_dev, best_mode


def infer_all(model, data):
    model.eval()
    values = {
        "z": [],
        "base_log_lam": [],
        "hard_log_lam": [],
        "soft_log_lam": [],
        "expert_log_lam": [],
        "base_err": [],
        "hard_err": [],
        "soft_err": [],
        "expert_err": [],
        "gates": [],
    }
    mode = int(model.inference_mode.item())
    selected_error_key = {
        MODE_BASE: "base_err",
        MODE_HARD: "hard_err",
        MODE_SOFT: "soft_err",
    }[mode]

    with torch.no_grad():
        for start in range(0, data.n, BATCH):
            idx = torch.arange(start, min(start + BATCH, data.n))
            x = data.agg(idx).to(device)
            z, outputs = model.forward_with_experts(x)
            values["z"].append(z.cpu())
            for name in (
                "base_log_lam",
                "hard_log_lam",
                "soft_log_lam",
                "expert_log_lam",
            ):
                values[name].append(outputs[name].cpu())
            values["base_err"].append(
                poisson_deviance(outputs["base_log_lam"], x).cpu()
            )
            values["hard_err"].append(
                poisson_deviance(outputs["hard_log_lam"], x).cpu()
            )
            values["soft_err"].append(
                poisson_deviance(outputs["soft_log_lam"], x).cpu()
            )
            values["expert_err"].append(
                expert_deviances(outputs["expert_log_lam"], x).cpu()
            )
            values["gates"].append(outputs["gates"].cpu())

    arrays = {name: torch.cat(parts).numpy() for name, parts in values.items()}
    arrays["oracle_err"] = arrays["expert_err"].min(axis=1)
    arrays["err"] = arrays[selected_error_key]
    return arrays


def main():
    args = parse_args()
    if args.base_epochs <= 0 or args.moe_epochs <= 0:
        raise ValueError("--base-epochs and --moe-epochs must be positive")

    log = open_log(
        VERSION,
        {
            "LATENT_DIM": LATENT_DIM,
            "ARCHITECTURE": "shared total + competitive residual composition experts",
            "BASE_EPOCHS": args.base_epochs,
            "MOE_EPOCHS": args.moe_epochs,
            "SPECIALIZE_EPOCHS": SPECIALIZE_EPOCHS,
            "E_STEP_EVERY": E_STEP_EVERY,
            "BATCH": BATCH,
            "LR_BASE": LR_BASE,
            "LR_BACKBONE_FINETUNE": LR_BACKBONE_FINETUNE,
            "LR_EXPERT": LR_EXPERT,
            "LR_ROUTER": LR_ROUTER,
            "WEIGHT_DECAY": WEIGHT_DECAY,
            "VAL_FRAC": VAL_FRAC,
            "SEED": SEED,
            "LAMBDA_FSCE": LAMBDA_FSCE,
            "LAMBDA_ROUTER": LAMBDA_ROUTER,
            "LAMBDA_RESIDUAL": LAMBDA_RESIDUAL,
            "RESPONSIBILITY_ROUTER_WEIGHT": RESPONSIBILITY_ROUTER_WEIGHT,
            "MIN_EXPERT_FRACTION": MIN_EXPERT_FRACTION,
            "NOISE_P": NOISE_P,
            "NOISE_MODE": NOISE_MODE,
            "N_EXPERTS": N_EXPERTS,
            "ROUTER_TEMPERATURE": ROUTER_TEMPERATURE,
            "EXPERT_INIT_NOISE": EXPERT_INIT_NOISE,
            "FALLBACK": "minimum validation deviance among base/hard/soft",
        },
    )

    torch.manual_seed(SEED)
    ensure_patches()
    data = Patches(PATCHES)
    generator = torch.Generator().manual_seed(SEED)
    permutation = torch.randperm(data.n, generator=generator)
    n_validation = int(data.n * VAL_FRAC)
    val_idx = permutation[:n_validation]
    train_idx = permutation[n_validation:]

    x_train = np.log1p(data.agg(train_idx).numpy())
    edge_i, edge_j, edge_weight, fsce_a, fsce_b = build_fsce_graph(
        x_train, n_neighbors=N_NEIGHBORS, metric=GRAPH_METRIC
    )
    null_log_lam = (
        data.agg(train_idx)
        .mean(dim=0, keepdim=True)
        .clamp_min(1e-8)
        .log()
        .to(device)
    )
    log(
        f"{data.n} patches, train={len(train_idx)}, val={len(val_idx)}, "
        f"device={device}, train-only FSCE edges={len(edge_i)}"
    )

    model = MLPAE(
        latent_dim=LATENT_DIM,
        n_experts=N_EXPERTS,
        router_temperature=ROUTER_TEMPERATURE,
        expert_init_noise=EXPERT_INIT_NOISE,
    ).to(device)
    baseline_epoch, baseline_dev = train_baseline(
        model,
        data,
        train_idx,
        val_idx,
        edge_i,
        edge_j,
        edge_weight,
        fsce_a,
        fsce_b,
        null_log_lam,
        args.base_epochs,
        log,
    )
    moe_epoch, best_dev, best_mode = train_competitive_moe(
        model,
        data,
        train_idx,
        val_idx,
        edge_i,
        edge_j,
        edge_weight,
        fsce_a,
        fsce_b,
        null_log_lam,
        args.moe_epochs,
        baseline_epoch,
        baseline_dev,
        log,
    )

    inference = infer_all(model, data)
    if not all(np.isfinite(value).all() for value in inference.values()):
        raise FloatingPointError("final inference contains NaN or infinity")

    if not args.no_save:
        torch.save(model.state_dict(), CKPT)
        is_train = np.zeros(data.n, dtype=bool)
        is_train[train_idx.numpy()] = True
        np.savez(
            OUT,
            n_poi=data.n_poi,
            lat=data.lat,
            lon=data.lon,
            **inference,
            expert_id=inference["gates"].argmax(axis=1),
            is_train=is_train,
            is_val=~is_train,
            baseline_epoch=np.int64(baseline_epoch),
            baseline_val_deviance=np.float32(baseline_dev),
            best_moe_epoch=np.int64(moe_epoch),
            best_val_deviance=np.float32(best_dev),
            selected_mode=np.asarray(MODE_NAMES[best_mode]),
        )
        log(f"saved checkpoint: {CKPT}")
        log(f"saved diagnostics and strict-2D latents: {OUT}")


if __name__ == "__main__":
    main()
