"""Train DDAE + OD-GAT with balanced soft-composition fuzzy pairs.

Unlike the failed composition-head experiment, this objective acts directly
on latent distances.  It replaces the original log-count kNN FSCE graph with
an implicit training-only graph whose categories are sampled uniformly and
whose pair memberships remain continuous for mixed patches.
"""

from __future__ import annotations

import argparse
import os
import re
import sys

import numpy as np
import torch
from sklearn.metrics import f1_score
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.preprocessing import StandardScaler


sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.abspath(f"{os.path.dirname(__file__)}/../.."))
from ae import (  # noqa: E402
    CountODFusedAE,
    Patches,
    build_balanced_composition_pairs,
    corrupt,
    count_composition,
    fsce_loss,
    load_od_graph,
    poisson_deviance,
    poisson_nll,
    sample_balanced_composition_pairs,
)
from config.dataset import CSV as POI_CSV  # noqa: E402
from config.dataset import (  # noqa: E402
    CATEGORIES,
    N_CAT,
    PATCHES,
    ensure_patches,
    result,
)


VERSION = "v2_ddae_gat_balanced_fsce"
OD_CSV = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../data/odGraph.csv")
)

LATENT_DIM = 2
EPOCHS = 300
BATCH = 256
LR = 1e-3
WEIGHT_DECAY = 0
VAL_FRAC = 0.1
SEED = 0

NOISE_P = 0.3
NOISE_MODE = "thinning"
ALPHA_OD = 0.3

PAIR_BATCH = 256
BETA_PAIR = 0.1
PAIR_WARMUP_EPOCHS = 100
SAMPLING_POWER = 0.75
MEMBERSHIP_POWER = 2.0
MIXED_MARGIN = 0.15
K = 15
GRAD_CLIP = 5.0

DINING = CATEGORIES.index("Dining and Drinking")

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
    parser.add_argument("--beta-pair", type=float, default=BETA_PAIR)
    parser.add_argument(
        "--membership-power", type=float, default=MEMBERSHIP_POWER
    )
    parser.add_argument("--mixed-margin", type=float, default=MIXED_MARGIN)
    parser.add_argument(
        "--output-tag",
        default="",
        help="Optional suffix, e.g. beta005 -> latents_beta005.npz.",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Run training and validation without writing checkpoint/latents.",
    )
    return parser.parse_args()


def _macro_f1(target, prediction):
    labels = np.union1d(target, prediction)
    return f1_score(
        target,
        prediction,
        labels=labels,
        average="macro",
        zero_division=0,
    )


def infer_all(model, graph, data):
    model.eval()
    z_values = []
    z_count_values = []
    z_od_values = []
    nll_values = []
    deviance_values = []
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
            log_rate = model.decode(z)
            z_values.append(z.cpu())
            z_count_values.append(z_count.cpu())
            z_od_values.append(z_od.cpu())
            nll_values.append(poisson_nll(log_rate, x).cpu())
            deviance_values.append(poisson_deviance(log_rate, x).cpu())
    return tuple(
        torch.cat(values).numpy()
        for values in (
            z_values,
            z_count_values,
            z_od_values,
            nll_values,
            deviance_values,
        )
    )


def geometry_metrics(
    z,
    nll,
    deviance,
    all_counts,
    is_train,
    is_val,
    mixed_margin,
):
    composition = all_counts / np.maximum(
        all_counts.sum(axis=1, keepdims=True), 1.0
    )
    label = composition.argmax(axis=1)
    ordered = np.sort(composition, axis=1)
    margin = ordered[:, -1] - ordered[:, -2]

    scaler = StandardScaler().fit(z[is_train])
    train_z = scaler.transform(z[is_train])
    val_z = scaler.transform(z[is_val])
    classifier = KNeighborsClassifier(n_neighbors=K, weights="distance")
    classifier.fit(train_z, label[is_train])
    prediction = classifier.predict(val_z)
    confident = margin[is_val] >= mixed_margin

    neighbors = NearestNeighbors(n_neighbors=K).fit(train_z)
    neighbor_index = neighbors.kneighbors(val_z, return_distance=False)
    neighbor_label = label[is_train][neighbor_index]
    non_dining = label[is_val] != DINING

    eigenvalues = np.linalg.eigvalsh(np.cov(z.T))
    return {
        "nll": float(nll[is_val].mean()),
        "deviance": float(deviance[is_val].mean()),
        "macro_f1": float(_macro_f1(label[is_val], prediction)),
        "confident_macro_f1": float(
            _macro_f1(label[is_val][confident], prediction[confident])
        ),
        "non_dining_neighbor_dining_fraction": float(
            (neighbor_label[non_dining] == DINING).mean()
        ),
        "non_dining_predicted_as_dining": float(
            (prediction[non_dining] == DINING).mean()
        ),
        "pc1_variance_ratio": float(eigenvalues[-1] / eigenvalues.sum()),
    }


def run(
    data,
    graph,
    train_idx,
    val_idx,
    pair_sampler,
    all_counts,
    epochs,
    alpha_od,
    beta_pair,
    mixed_margin,
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
    is_train = np.zeros(data.n, dtype=bool)
    is_train[train_idx.numpy()] = True
    is_val = ~is_train

    for epoch in range(epochs):
        model.train()
        pair_weight_t = beta_pair * min(
            1.0, (epoch + 1) / max(PAIR_WARMUP_EPOCHS, 1)
        )
        generator = torch.Generator().manual_seed(SEED + epoch)
        permutation = train_idx[
            torch.randperm(len(train_idx), generator=generator)
        ]
        total_reconstruction = 0.0
        total_pair = 0.0
        total_loss = 0.0

        for start in range(0, len(permutation), BATCH):
            batch = permutation[start : start + BATCH]
            x_clean = data.agg(batch)
            x_noisy = corrupt(
                x_clean, NOISE_P, NOISE_MODE, generator=generator
            ).to(device)
            x_clean = x_clean.to(device)
            batch_device = batch.to(device)

            # Training view contains no validation-to-validation or cross-split
            # OD edges. Recompute per optimizer step so gradients reach the GAT.
            z_od_all = model.encode_od(graph, training_graph=True)
            z_count = model.encode_count(x_noisy)
            z = model.fuse(
                z_count,
                z_od_all[batch_device],
                graph.patch_has_od[batch_device],
            )
            reconstruction = poisson_nll(model.decode(z), x_clean).mean()

            left_local, right_local, membership = (
                sample_balanced_composition_pairs(
                    pair_sampler,
                    n_pairs=PAIR_BATCH,
                    generator=generator,
                )
            )
            left_patch = train_idx[left_local]
            right_patch = train_idx[right_local]
            pair_patch = torch.cat([left_patch, right_patch])
            pair_count = corrupt(
                data.agg(pair_patch),
                NOISE_P,
                NOISE_MODE,
                generator=generator,
            ).to(device)
            pair_device = pair_patch.to(device)
            pair_z = model.fuse(
                model.encode_count(pair_count),
                z_od_all[pair_device],
                graph.patch_has_od[pair_device],
            )
            z_left, z_right = pair_z.chunk(2, dim=0)
            pair_loss = fsce_loss(
                z_left,
                z_right,
                membership.to(device),
                pair_sampler.a,
                pair_sampler.b,
            ).mean()

            loss = reconstruction + pair_weight_t * pair_loss
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"non-finite loss at epoch {epoch + 1}: "
                    f"recon={reconstruction.item()} pair={pair_loss.item()}"
                )
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()

            batch_size = len(batch)
            total_reconstruction += reconstruction.item() * batch_size
            total_pair += pair_loss.item() * batch_size
            total_loss += loss.item() * batch_size

        should_report = (epoch + 1) % 20 == 0 or epoch == 0 or epoch + 1 == epochs
        if should_report:
            z_eval, _, _, nll_eval, deviance_eval = infer_all(model, graph, data)
            metrics = geometry_metrics(
                z_eval,
                nll_eval,
                deviance_eval,
                all_counts,
                is_train,
                is_val,
                mixed_margin,
            )
            print(
                f"epoch {epoch + 1:4d}  "
                f"train total {total_loss / len(train_idx):.5f}  "
                f"recon {total_reconstruction / len(train_idx):.5f}  "
                f"balanced pair {total_pair / len(train_idx):.5f}  "
                f"pair weight {pair_weight_t:.3f}  "
                f"clean val NLL {metrics['nll']:.5f}  "
                f"clean val deviance {metrics['deviance']:.5f}  "
                f"val macro F1 {metrics['macro_f1']:.3f}  "
                f"confident F1 {metrics['confident_macro_f1']:.3f}  "
                f"non-dining neighbor dining "
                f"{metrics['non_dining_neighbor_dining_fraction']:.3f}  "
                f"PC1 ratio {metrics['pc1_variance_ratio']:.3f}"
            )

    return model, graph, is_train, is_val


def main():
    args = parse_args()
    if args.epochs <= 0:
        raise ValueError("--epochs must be positive")
    if not 0.0 <= args.alpha_od <= 1.0:
        raise ValueError("--alpha-od must be in [0, 1]")
    if args.beta_pair < 0.0:
        raise ValueError("--beta-pair must be non-negative")
    if args.membership_power <= 0.0:
        raise ValueError("--membership-power must be positive")
    if not 0.0 <= args.mixed_margin <= 1.0:
        raise ValueError("--mixed-margin must be in [0, 1]")
    if args.output_tag and not re.fullmatch(r"[A-Za-z0-9_-]+", args.output_tag):
        raise ValueError("--output-tag may contain only letters, digits, _ and -")
    if not os.path.exists(OD_CSV):
        raise FileNotFoundError(
            f"missing {OD_CSV}; run data/preprocess/build_od.py first"
        )

    suffix = f"_{args.output_tag}" if args.output_tag else ""
    checkpoint_path = result(VERSION, f"ae{suffix}.pt")
    latent_path = result(VERSION, f"latents{suffix}.npz")

    ensure_patches()
    data = Patches(PATCHES)
    generator = torch.Generator().manual_seed(SEED)
    permutation = torch.randperm(data.n, generator=generator)
    n_validation = int(data.n * VAL_FRAC)
    val_idx, train_idx = permutation[:n_validation], permutation[n_validation:]
    all_counts_tensor = data.agg(torch.arange(data.n))
    all_counts = all_counts_tensor.numpy()

    pair_sampler = build_balanced_composition_pairs(
        all_counts_tensor[train_idx],
        sampling_power=SAMPLING_POWER,
        membership_power=args.membership_power,
    )
    graph = load_od_graph(
        OD_CSV,
        POI_CSV,
        n_patches=data.n,
        train_idx=train_idx.numpy(),
    )

    diagnostic_generator = torch.Generator().manual_seed(SEED)
    _, _, diagnostic_membership = sample_balanced_composition_pairs(
        pair_sampler,
        n_pairs=5000,
        generator=diagnostic_generator,
    )
    same = diagnostic_membership[:5000]
    cross = diagnostic_membership[5000:]
    print(
        f"{data.n} patches, device={device}, noise={NOISE_MODE} p={NOISE_P}, "
        f"weight_decay={WEIGHT_DECAY}, alpha_od={args.alpha_od}, "
        f"beta_pair={args.beta_pair}, membership_power={args.membership_power}"
    )
    print(
        f"balanced pair membership median: same-category={same.median():.3f}, "
        f"cross-category={cross.median():.3f}; "
        f"OD raw edges={graph.n_raw_edges}, "
        f"OD training-only raw edges={graph.n_train_raw_edges}, "
        f"POI nodes={len(graph.node_features)}"
    )
    print(
        "training-only composition prior: "
        + ", ".join(
            f"{name}={value:.4f}"
            for name, value in zip(CATEGORIES, pair_sampler.prior.tolist())
        )
    )

    model, graph, is_train, is_val = run(
        data,
        graph,
        train_idx,
        val_idx,
        pair_sampler,
        all_counts,
        epochs=args.epochs,
        alpha_od=args.alpha_od,
        beta_pair=args.beta_pair,
        mixed_margin=args.mixed_margin,
    )
    z, z_count, z_od, nll, deviance = infer_all(model, graph, data)
    arrays = (z, z_count, z_od, nll, deviance)
    if not all(np.isfinite(value).all() for value in arrays):
        raise FloatingPointError("final inference contains NaN or infinity")

    metrics = geometry_metrics(
        z,
        nll,
        deviance,
        all_counts,
        is_train,
        is_val,
        args.mixed_margin,
    )
    print("final validation metrics: " + repr(metrics))

    if not args.no_save:
        checkpoint = {
            "model_state": model.state_dict(),
            "alpha_od": args.alpha_od,
            "beta_pair": args.beta_pair,
            "sampling_power": SAMPLING_POWER,
            "membership_power": args.membership_power,
            "mixed_margin": args.mixed_margin,
            "latent_dim": LATENT_DIM,
            "node_feature_dim": graph.node_feature_dim,
            "edge_feature_dim": graph.edge_feature_dim,
            "weight_decay": WEIGHT_DECAY,
            "composition_prior": pair_sampler.prior,
        }
        torch.save(checkpoint, checkpoint_path)
        composition = count_composition(all_counts_tensor).numpy()
        ordered = np.sort(composition, axis=1)
        target_margin = ordered[:, -1] - ordered[:, -2]
        np.savez(
            latent_path,
            n_poi=data.n_poi,
            lat=data.lat,
            lon=data.lon,
            z=z,
            z_count=z_count,
            z_od=z_od,
            nll=nll,
            err=deviance,
            composition_target=composition,
            dominant_target=composition.argmax(axis=1),
            target_margin=target_margin,
            mixed_target=target_margin < args.mixed_margin,
            is_train=is_train,
            is_val=is_val,
            alpha_od=np.float32(args.alpha_od),
            beta_pair=np.float32(args.beta_pair),
            sampling_power=np.float32(SAMPLING_POWER),
            membership_power=np.float32(args.membership_power),
            mixed_margin=np.float32(args.mixed_margin),
            composition_prior=pair_sampler.prior.numpy(),
        )
        print(f"saved checkpoint: {checkpoint_path}")
        print(f"saved latents: {latent_path}")


if __name__ == "__main__":
    main()
