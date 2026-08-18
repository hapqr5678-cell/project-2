"""Train DDAE + FSCE + OD-GAT with soft category-composition supervision.

Leakage controls:
  * split patches before estimating any graph or class weight;
  * build FSCE and GAT training views from training patches only;
  * estimate composition class weights from training counts only;
  * use clean validation counts only for reporting metrics;
  * use the full OD graph only in eval mode, never for optimizer updates.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch
from sklearn.metrics import f1_score


sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.abspath(f"{os.path.dirname(__file__)}/../.."))
from ae import (  # noqa: E402
    CountODCompositionAE,
    Patches,
    build_fsce_graph,
    class_balanced_soft_ce,
    composition_js,
    corrupt,
    count_composition,
    fsce_loss,
    load_od_graph,
    make_class_weights,
    poisson_deviance,
    poisson_nll,
)
from config.dataset import CSV as POI_CSV  # noqa: E402
from config.dataset import (  # noqa: E402
    CATEGORIES,
    N_CAT,
    PATCHES,
    ensure_patches,
    result,
)


VERSION = "v2_ddae_gat_composition"
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
FSCE_WARMUP_EPOCHS = 500

NOISE_P = 0.3
NOISE_MODE = "thinning"
ALPHA_OD = 0.3
BETA_COMPOSITION = 0.5
COMPOSITION_WARMUP_EPOCHS = 50
MIXED_MARGIN = 0.15
CLASS_WEIGHT_POWER = 0.5
CLASS_WEIGHT_MIN = 0.25
CLASS_WEIGHT_MAX = 4.0
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
    parser.add_argument(
        "--beta-composition", type=float, default=BETA_COMPOSITION
    )
    parser.add_argument("--mixed-margin", type=float, default=MIXED_MARGIN)
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Run training and validation without writing checkpoint/latents.",
    )
    return parser.parse_args()


def _margin(probability):
    top_two = probability.topk(k=2, dim=1).values
    return top_two[:, 0] - top_two[:, 1]


def _macro_f1(target, prediction):
    target = target.numpy()
    prediction = prediction.numpy()
    labels = np.union1d(target, prediction)
    return f1_score(
        target,
        prediction,
        labels=labels,
        average="macro",
        zero_division=0,
    )


def validation_metrics(model, graph, data, val_idx, mixed_margin):
    model.eval()
    nll_values = []
    deviance_values = []
    js_values = []
    target_probability_values = []
    predicted_probability_values = []
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
            log_rate = model.decode(z)
            target_probability = count_composition(x)
            predicted_probability = model.composition_probabilities(z)

            nll_values.append(poisson_nll(log_rate, x).cpu())
            deviance_values.append(poisson_deviance(log_rate, x).cpu())
            js_values.append(
                composition_js(predicted_probability, target_probability).cpu()
            )
            target_probability_values.append(target_probability.cpu())
            predicted_probability_values.append(predicted_probability.cpu())

    target_probability = torch.cat(target_probability_values)
    predicted_probability = torch.cat(predicted_probability_values)
    target_label = target_probability.argmax(dim=1)
    predicted_label = predicted_probability.argmax(dim=1)
    target_margin = _margin(target_probability)
    predicted_margin = _margin(predicted_probability)
    confident = target_margin >= mixed_margin

    hard_macro_f1 = _macro_f1(target_label, predicted_label)
    confident_macro_f1 = (
        _macro_f1(target_label[confident], predicted_label[confident])
        if confident.any()
        else float("nan")
    )
    mixed_target = target_label.clone()
    mixed_prediction = predicted_label.clone()
    mixed_target[target_margin < mixed_margin] = N_CAT
    mixed_prediction[predicted_margin < mixed_margin] = N_CAT
    mixed_macro_f1 = _macro_f1(mixed_target, mixed_prediction)

    non_dining = target_label != DINING
    dining_false_positive = (
        (predicted_label[non_dining] == DINING).float().mean().item()
        if non_dining.any()
        else float("nan")
    )
    return {
        "nll": torch.cat(nll_values).mean().item(),
        "deviance": torch.cat(deviance_values).mean().item(),
        "composition_js": torch.cat(js_values).mean().item(),
        "composition_mae": (
            predicted_probability - target_probability
        ).abs().mean().item(),
        "hard_macro_f1": hard_macro_f1,
        "confident_macro_f1": confident_macro_f1,
        "mixed_macro_f1": mixed_macro_f1,
        "true_mixed_fraction": (target_margin < mixed_margin).float().mean().item(),
        "predicted_mixed_fraction": (
            predicted_margin < mixed_margin
        ).float().mean().item(),
        "non_dining_predicted_as_dining": dining_false_positive,
    }


def infer_all(model, graph, data):
    model.eval()
    z_values = []
    z_count_values = []
    z_od_values = []
    error_values = []
    target_composition_values = []
    predicted_composition_values = []
    js_values = []
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
            target_composition = count_composition(x)
            predicted_composition = model.composition_probabilities(z)

            z_values.append(z.cpu())
            z_count_values.append(z_count.cpu())
            z_od_values.append(z_od.cpu())
            error_values.append(poisson_deviance(log_rate, x).cpu())
            target_composition_values.append(target_composition.cpu())
            predicted_composition_values.append(predicted_composition.cpu())
            js_values.append(
                composition_js(predicted_composition, target_composition).cpu()
            )
    return tuple(
        torch.cat(values).numpy()
        for values in (
            z_values,
            z_count_values,
            z_od_values,
            error_values,
            target_composition_values,
            predicted_composition_values,
            js_values,
        )
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
    class_weights,
    epochs,
    alpha_od,
    beta_composition,
    mixed_margin,
):
    torch.manual_seed(SEED)
    model = CountODCompositionAE(
        latent_dim=LATENT_DIM,
        node_feature_dim=graph.node_feature_dim,
        edge_feature_dim=graph.edge_feature_dim,
        n_categories=N_CAT,
        alpha_od=alpha_od,
    ).to(device)
    graph = graph.to(device)
    class_weights = class_weights.to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY
    )
    n_fsce_edges = len(fsce_i)

    for epoch in range(epochs):
        model.train()
        fsce_weight_t = FSCE_WEIGHT * min(
            1.0, (epoch + 1) / max(FSCE_WARMUP_EPOCHS, 1)
        )
        composition_weight_t = beta_composition * min(
            1.0, (epoch + 1) / max(COMPOSITION_WARMUP_EPOCHS, 1)
        )
        generator = torch.Generator().manual_seed(SEED + epoch)
        permutation = train_idx[
            torch.randperm(len(train_idx), generator=generator)
        ]
        total_reconstruction = 0.0
        total_fsce = 0.0
        total_composition = 0.0
        total_loss = 0.0

        for start in range(0, len(permutation), BATCH):
            batch = permutation[start : start + BATCH]
            x_clean = data.agg(batch)
            x_noisy = corrupt(
                x_clean, NOISE_P, NOISE_MODE, generator=generator
            )
            target_composition = count_composition(x_clean).to(device)
            x_clean = x_clean.to(device)
            x_noisy = x_noisy.to(device)
            batch_device = batch.to(device)

            # Recompute once per optimizer step so gradients reach the GAT.
            z_od_all = model.encode_od(graph, training_graph=True)
            z_count = model.encode_count(x_noisy)
            z = model.fuse(
                z_count,
                z_od_all[batch_device],
                graph.patch_has_od[batch_device],
            )
            reconstruction = poisson_nll(model.decode(z), x_clean).mean()
            composition = class_balanced_soft_ce(
                model.composition_logits(z),
                target_composition,
                class_weights,
            ).mean()

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

            loss = (
                reconstruction
                + fsce_weight_t * fuzzy
                + composition_weight_t * composition
            )
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"non-finite loss at epoch {epoch + 1}: "
                    f"recon={reconstruction.item()} FSCE={fuzzy.item()} "
                    f"composition={composition.item()}"
                )
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()

            batch_size = len(batch)
            total_reconstruction += reconstruction.item() * batch_size
            total_fsce += fuzzy.item() * batch_size
            total_composition += composition.item() * batch_size
            total_loss += loss.item() * batch_size

        should_report = (epoch + 1) % 20 == 0 or epoch == 0 or epoch + 1 == epochs
        if should_report:
            metrics = validation_metrics(
                model, graph, data, val_idx, mixed_margin
            )
            print(
                f"epoch {epoch + 1:4d}  "
                f"train total {total_loss / len(train_idx):.5f}  "
                f"recon {total_reconstruction / len(train_idx):.5f}  "
                f"FSCE {total_fsce / len(train_idx):.5f}  "
                f"composition {total_composition / len(train_idx):.5f}  "
                f"FSCE weight {fsce_weight_t:.3f}  "
                f"composition weight {composition_weight_t:.3f}  "
                f"clean val NLL {metrics['nll']:.5f}  "
                f"clean val deviance {metrics['deviance']:.5f}  "
                f"val JS {metrics['composition_js']:.5f}  "
                f"hard F1 {metrics['hard_macro_f1']:.3f}  "
                f"confident F1 {metrics['confident_macro_f1']:.3f}  "
                f"mixed F1 {metrics['mixed_macro_f1']:.3f}  "
                f"non-dining->dining {metrics['non_dining_predicted_as_dining']:.3f}"
            )

    return model, graph


def main():
    args = parse_args()
    if args.epochs <= 0:
        raise ValueError("--epochs must be positive")
    if not 0.0 <= args.alpha_od <= 1.0:
        raise ValueError("--alpha-od must be in [0, 1]")
    if args.beta_composition < 0.0:
        raise ValueError("--beta-composition must be non-negative")
    if not 0.0 <= args.mixed_margin <= 1.0:
        raise ValueError("--mixed-margin must be in [0, 1]")
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

    training_counts = data.agg(train_idx)
    x_train_graph = np.log1p(training_counts.numpy())
    fsce_i, fsce_j, fsce_edge_weight, fsce_a, fsce_b = build_fsce_graph(
        x_train_graph, n_neighbors=N_NEIGHBORS, metric=GRAPH_METRIC
    )
    class_weights = make_class_weights(
        training_counts,
        power=CLASS_WEIGHT_POWER,
        minimum=CLASS_WEIGHT_MIN,
        maximum=CLASS_WEIGHT_MAX,
    )
    graph = load_od_graph(
        OD_CSV,
        POI_CSV,
        n_patches=data.n,
        train_idx=train_idx.numpy(),
    )
    print(
        f"{data.n} patches, device={device}, noise={NOISE_MODE} p={NOISE_P}, "
        f"weight_decay={WEIGHT_DECAY}, alpha_od={args.alpha_od}, "
        f"beta_composition={args.beta_composition}, "
        f"mixed_margin={args.mixed_margin}"
    )
    print(
        f"FSCE training-only edges={len(fsce_i)}, "
        f"OD raw edges={graph.n_raw_edges}, "
        f"OD training-only raw edges={graph.n_train_raw_edges}, "
        f"POI nodes={len(graph.node_features)}"
    )
    print(
        "training-only composition class weights: "
        + ", ".join(
            f"{name}={weight:.3f}"
            for name, weight in zip(CATEGORIES, class_weights.tolist())
        )
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
        class_weights,
        epochs=args.epochs,
        alpha_od=args.alpha_od,
        beta_composition=args.beta_composition,
        mixed_margin=args.mixed_margin,
    )
    (
        z,
        z_count,
        z_od,
        error,
        target_composition,
        predicted_composition,
        composition_error,
    ) = infer_all(model, graph, data)
    arrays = (
        z,
        z_count,
        z_od,
        error,
        target_composition,
        predicted_composition,
        composition_error,
    )
    if not all(np.isfinite(value).all() for value in arrays):
        raise FloatingPointError("final inference contains NaN or infinity")

    if not args.no_save:
        checkpoint = {
            "model_state": model.state_dict(),
            "alpha_od": args.alpha_od,
            "beta_composition": args.beta_composition,
            "mixed_margin": args.mixed_margin,
            "latent_dim": LATENT_DIM,
            "node_feature_dim": graph.node_feature_dim,
            "edge_feature_dim": graph.edge_feature_dim,
            "n_categories": N_CAT,
            "weight_decay": WEIGHT_DECAY,
            "class_weights": class_weights,
        }
        torch.save(checkpoint, CKPT)
        is_train = np.zeros(data.n, dtype=bool)
        is_train[train_idx.numpy()] = True
        target_label = target_composition.argmax(axis=1)
        predicted_label = predicted_composition.argmax(axis=1)
        target_margin = np.sort(target_composition, axis=1)[:, -1] - np.sort(
            target_composition, axis=1
        )[:, -2]
        predicted_margin = np.sort(predicted_composition, axis=1)[:, -1] - np.sort(
            predicted_composition, axis=1
        )[:, -2]
        np.savez(
            OUT,
            n_poi=data.n_poi,
            lat=data.lat,
            lon=data.lon,
            z=z,
            z_count=z_count,
            z_od=z_od,
            err=error,
            composition_target=target_composition,
            composition_pred=predicted_composition,
            composition_js=composition_error,
            dominant_target=target_label,
            dominant_pred=predicted_label,
            target_margin=target_margin,
            predicted_margin=predicted_margin,
            mixed_target=target_margin < args.mixed_margin,
            mixed_pred=predicted_margin < args.mixed_margin,
            is_train=is_train,
            is_val=~is_train,
            alpha_od=np.float32(args.alpha_od),
            beta_composition=np.float32(args.beta_composition),
            mixed_margin=np.float32(args.mixed_margin),
            class_weights=class_weights.numpy(),
        )
        print(f"saved checkpoint: {CKPT}")
        print(f"saved latents: {OUT}")


if __name__ == "__main__":
    main()
