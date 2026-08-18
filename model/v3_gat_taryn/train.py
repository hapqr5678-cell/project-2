"""Train the DDAE + FSCE + POI OD-GAT fusion model.

Training leakage controls:
  * the validation split is made before either training graph is built;
  * the FSCE graph contains training patches only;
  * the GAT training graph contains only OD edges whose two patches are train;
  * OD feature normalization uses training nodes/edges only;
  * clean validation counts are used only for reporting metrics.

The full OD graph is used only in eval mode for validation/final inference.  It
is treated as an input available at inference time and never updates weights.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch


sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.abspath(f"{os.path.dirname(__file__)}/../.."))
from ae import (  # noqa: E402
    CountODFusedAE,
    Patches,
    build_fsce_graph,
    corrupt,
    fsce_loss,
    load_od_graph,
    poisson_deviance,
    poisson_nll,
)
from config.dataset import CSV as POI_CSV  # noqa: E402
from config.dataset import PATCHES, ensure_patches, result  # noqa: E402


VERSION = "v2_ddae_fsce_gat"
OD_CSV = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../data/odGraph.csv")
)
OUT = result(VERSION, "latents.npz")
CKPT = result(VERSION, "ae.pt")

LATENT_DIM = 2
EPOCHS = 300
BATCH = 256
LR = 1e-3
WEIGHT_DECAY = 0
VAL_FRAC = 0.1
SEED = 0

N_NEIGHBORS = 15
GRAPH_METRIC = "cosine"
EDGE_BATCH = 256
FSCE_WEIGHT = 0.5
WARMUP_EPOCHS = 500

NOISE_P = 0.3
NOISE_MODE = "thinning"
ALPHA_OD = 0.3
GRAD_CLIP = 5.0

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
    parser.add_argument("--alpha-od", type=float, default=ALPHA_OD)
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Run training and validation without writing checkpoint/latents.",
    )
    return parser.parse_args()


def validation_metrics(model, graph, data, val_idx):
    model.eval()
    nll_values, deviance_values = [], []
    with torch.no_grad():
        z_od_all = model.encode_od(graph, training_graph=False)
        for start in range(0, len(val_idx), BATCH):
            patch_idx = val_idx[start : start + BATCH]
            x = data.agg(patch_idx).to(device)
            patch_device = patch_idx.to(device)
            z_count = model.encode_count(x)
            z = model.fuse(
                z_count,
                z_od_all[patch_device],
                graph.patch_has_od[patch_device],
            )
            log_lam = model.decode(z)
            nll_values.append(poisson_nll(log_lam, x).cpu())
            deviance_values.append(poisson_deviance(log_lam, x).cpu())
    return (
        torch.cat(nll_values).mean().item(),
        torch.cat(deviance_values).mean().item(),
    )


def infer_all(model, graph, data):
    model.eval()
    z_values, z_count_values, z_od_values, error_values = [], [], [], []
    with torch.no_grad():
        z_od_all = model.encode_od(graph, training_graph=False)
        for start in range(0, data.n, BATCH):
            patch_idx = torch.arange(start, min(start + BATCH, data.n))
            x = data.agg(patch_idx).to(device)
            patch_device = patch_idx.to(device)
            z_count = model.encode_count(x)
            z_od = z_od_all[patch_device]
            z = model.fuse(
                z_count,
                z_od,
                graph.patch_has_od[patch_device],
            )
            log_lam = model.decode(z)
            z_values.append(z.cpu())
            z_count_values.append(z_count.cpu())
            z_od_values.append(z_od.cpu())
            error_values.append(poisson_deviance(log_lam, x).cpu())
    return (
        torch.cat(z_values).numpy(),
        torch.cat(z_count_values).numpy(),
        torch.cat(z_od_values).numpy(),
        torch.cat(error_values).numpy(),
    )


def run(
    data,
    graph,
    train_idx,
    val_idx,
    fsce_i,
    fsce_j,
    fsce_edge_weight,
    fsce_a,
    fsce_b,
    epochs,
    alpha_od,
):
    torch.manual_seed(SEED)
    model = CountODFusedAE(
        latent_dim=LATENT_DIM,
        node_feature_dim=graph.node_feature_dim,
        edge_feature_dim=graph.edge_feature_dim,
        alpha_od=alpha_od,
    ).to(device)
    graph = graph.to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY
    )
    n_fsce_edges = len(fsce_i)

    for epoch in range(epochs):
        model.train()
        fsce_weight_t = FSCE_WEIGHT * min(
            1.0, (epoch + 1) / max(WARMUP_EPOCHS, 1)
        )
        generator = torch.Generator().manual_seed(SEED + epoch)
        permutation = train_idx[
            torch.randperm(len(train_idx), generator=generator)
        ]
        total_reconstruction = 0.0
        total_fsce = 0.0
        total_loss = 0.0

        for start in range(0, len(permutation), BATCH):
            batch = permutation[start : start + BATCH]
            x_clean = data.agg(batch)
            x_noisy = corrupt(
                x_clean, NOISE_P, NOISE_MODE, generator=generator
            )
            x_clean = x_clean.to(device)
            x_noisy = x_noisy.to(device)
            batch_device = batch.to(device)

            # Recomputed once per optimizer step so gradients reach every GAT layer.
            z_od_all = model.encode_od(graph, training_graph=True)
            z_count = model.encode_count(x_noisy)
            z = model.fuse(
                z_count,
                z_od_all[batch_device],
                graph.patch_has_od[batch_device],
            )
            reconstruction = poisson_nll(model.decode(z), x_clean).mean()

            sampled_edges = torch.randint(
                0, n_fsce_edges, (EDGE_BATCH,), generator=generator
            )
            positive_i = fsce_i[sampled_edges]
            positive_j = fsce_j[sampled_edges]
            positive_weight = fsce_edge_weight[sampled_edges]
            negative_i = torch.randint(
                0, len(train_idx), (EDGE_BATCH,), generator=generator
            )
            negative_j = torch.randint(
                0, len(train_idx), (EDGE_BATCH,), generator=generator
            )
            left_local = torch.cat([positive_i, negative_i])
            right_local = torch.cat([positive_j, negative_j])
            left_patch = train_idx[left_local]
            right_patch = train_idx[right_local]

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
            left_device = left_patch.to(device)
            right_device = right_patch.to(device)
            z_left = model.fuse(
                model.encode_count(left_count),
                z_od_all[left_device],
                graph.patch_has_od[left_device],
            )
            z_right = model.fuse(
                model.encode_count(right_count),
                z_od_all[right_device],
                graph.patch_has_od[right_device],
            )
            pair_weight = torch.cat(
                [positive_weight, torch.zeros(EDGE_BATCH)]
            ).to(device)
            fuzzy = fsce_loss(
                z_left, z_right, pair_weight, fsce_a, fsce_b
            ).mean()

            loss = reconstruction + fsce_weight_t * fuzzy
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"non-finite loss at epoch {epoch + 1}: "
                    f"recon={reconstruction.item()} fsce={fuzzy.item()}"
                )
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()

            batch_size = len(batch)
            total_reconstruction += reconstruction.item() * batch_size
            total_fsce += fuzzy.item() * batch_size
            total_loss += loss.item() * batch_size

        should_report = (epoch + 1) % 20 == 0 or epoch == 0 or epoch + 1 == epochs
        if should_report:
            val_nll, val_deviance = validation_metrics(
                model, graph, data, val_idx
            )
            print(
                f"epoch {epoch + 1:4d}  "
                f"train total {total_loss / len(train_idx):.5f}  "
                f"recon {total_reconstruction / len(train_idx):.5f}  "
                f"FSCE {total_fsce / len(train_idx):.5f}  "
                f"FSCE weight {fsce_weight_t:.3f}  alpha_od {alpha_od:.3f}  "
                f"clean val NLL {val_nll:.5f}  "
                f"clean val deviance {val_deviance:.5f}"
            )

    return model, graph


def main():
    args = parse_args()
    if args.epochs <= 0:
        raise ValueError("--epochs must be positive")
    if not 0.0 <= args.alpha_od <= 1.0:
        raise ValueError("--alpha-od must be in [0, 1]")
    if not os.path.exists(OD_CSV):
        raise FileNotFoundError(
            f"missing {OD_CSV}; run data/preprocess/build_od.py first"
        )

    ensure_patches()
    data = Patches(PATCHES)
    generator = torch.Generator().manual_seed(SEED)
    permutation = torch.randperm(data.n, generator=generator)
    n_validation = int(data.n * VAL_FRAC)
    val_idx, train_idx = permutation[:n_validation], permutation[n_validation:]

    x_train = np.log1p(data.agg(train_idx).numpy())
    fsce_i, fsce_j, fsce_edge_weight, fsce_a, fsce_b = build_fsce_graph(
        x_train, n_neighbors=N_NEIGHBORS, metric=GRAPH_METRIC
    )
    graph = load_od_graph(
        OD_CSV,
        POI_CSV,
        n_patches=data.n,
        train_idx=train_idx.numpy(),
    )
    print(
        f"{data.n} patches, device={device}, noise={NOISE_MODE} p={NOISE_P}, "
        f"weight_decay={WEIGHT_DECAY}, alpha_od={args.alpha_od}"
    )
    print(
        f"FSCE training-only edges={len(fsce_i)}, "
        f"OD raw edges={graph.n_raw_edges}, "
        f"OD training-only raw edges={graph.n_train_raw_edges}, "
        f"POI nodes={len(graph.node_features)}"
    )

    model, graph = run(
        data,
        graph,
        train_idx,
        val_idx,
        fsce_i,
        fsce_j,
        fsce_edge_weight,
        fsce_a,
        fsce_b,
        epochs=args.epochs,
        alpha_od=args.alpha_od,
    )
    z, z_count, z_od, error = infer_all(model, graph, data)
    if not all(np.isfinite(value).all() for value in (z, z_count, z_od, error)):
        raise FloatingPointError("final inference contains NaN or infinity")

    if not args.no_save:
        checkpoint = {
            "model_state": model.state_dict(),
            "alpha_od": args.alpha_od,
            "latent_dim": LATENT_DIM,
            "node_feature_dim": graph.node_feature_dim,
            "edge_feature_dim": graph.edge_feature_dim,
            "weight_decay": WEIGHT_DECAY,
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
            z_count=z_count,
            z_od=z_od,
            err=error,
            is_train=is_train,
            is_val=~is_train,
            alpha_od=np.float32(args.alpha_od),
        )
        print(f"saved checkpoint: {CKPT}")
        print(f"saved latents: {OUT}")


if __name__ == "__main__":
    main()
