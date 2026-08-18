"""Explain what v2_ddae_fsce_copy's two-dimensional latent encodes.

The figure colors the same latent coordinates by density, reconstruction
deviance, category diversity, dominant category, latitude, and longitude.
Metrics are evaluated from validation points to training-point neighbors so
the report does not train or evaluate a predictor on the same patch.
"""

import os
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, os.path.abspath(f"{os.path.dirname(__file__)}/../../.."))
from config.dataset import (  # noqa: E402
    CAT_COLORS,
    CAT_ZH,
    N_CAT,
    PATCHES,
    result,
)

VERSION = "v2_ddae_gat_composition"
LATENTS = result(VERSION, "latents.npz")
OUT = result(VERSION, "latent_analysisoverture.png")

SEED = 0
VAL_FRAC = 0.1
K = 15
DOT = 7.0

mpl.rcParams["font.family"] = ["Microsoft JhengHei", "Noto Sans CJK TC", "DejaVu Sans"]
mpl.rcParams["axes.unicode_minus"] = False
mpl.rcParams["figure.dpi"] = 130


def aggregate_counts(path):
    """Reconstruct the (N,N_CAT) clean count matrix from patches.npz."""
    data = np.load(path)
    cat = data["cat"].astype(np.int64)
    offsets = data["offsets"]
    counts = np.zeros((len(offsets) - 1, N_CAT), dtype=np.float32)
    for i in range(len(counts)):
        counts[i] = np.bincount(
            cat[offsets[i]:offsets[i + 1]], minlength=N_CAT
        )
    return counts


def pairwise_euclidean(query, reference):
    q2 = np.square(query).sum(axis=1, keepdims=True)
    r2 = np.square(reference).sum(axis=1, keepdims=True).T
    return np.sqrt(np.maximum(q2 + r2 - 2.0 * query @ reference.T, 0.0))


def pairwise_cosine(query, reference):
    query = query / np.maximum(np.linalg.norm(query, axis=1, keepdims=True), 1e-12)
    reference = reference / np.maximum(
        np.linalg.norm(reference, axis=1, keepdims=True), 1e-12
    )
    return 1.0 - query @ reference.T


def nearest_indices(distances, k):
    return np.argpartition(distances, kth=k - 1, axis=1)[:, :k]


def r2_score(y_true, y_pred):
    residual = np.square(y_true - y_pred).sum(axis=0)
    total = np.square(y_true - y_true.mean(axis=0)).sum(axis=0)
    return 1.0 - residual / np.maximum(total, 1e-12)


def validation_metrics(z, graph_features, values):
    """Measure information in held-out latents using training neighbors only."""
    generator = torch.Generator().manual_seed(SEED)
    perm = torch.randperm(len(z), generator=generator).numpy()
    n_val = int(len(z) * VAL_FRAC)
    val_idx, train_idx = perm[:n_val], perm[n_val:]

    high_knn = nearest_indices(
        pairwise_cosine(graph_features[val_idx], graph_features[train_idx]), K
    )
    latent_knn = nearest_indices(
        pairwise_euclidean(z[val_idx], z[train_idx]), K
    )
    recall = np.mean([
        len(set(high_knn[i]) & set(latent_knn[i])) / K
        for i in range(n_val)
    ])

    scores = {}
    for name, value in values.items():
        prediction = value[train_idx][latent_knn].mean(axis=1)
        score = r2_score(value[val_idx], prediction)
        scores[name] = float(np.mean(score))

    dominant = values["proportion"].argmax(axis=1)
    neighbor_classes = dominant[train_idx][latent_knn]
    prediction = np.array([
        np.bincount(row, minlength=N_CAT).argmax()
        for row in neighbor_classes
    ])
    dominant_accuracy = float(np.mean(prediction == dominant[val_idx]))
    return val_idx, recall, scores, dominant_accuracy


def continuous_panel(fig, ax, z, value, title, colorbar_label):
    points = ax.scatter(
        z[:, 0], z[:, 1], c=value, cmap="viridis", s=DOT,
        linewidths=0, alpha=0.72, rasterized=True,
    )
    ax.set_title(title, fontsize=10)
    colorbar = fig.colorbar(points, ax=ax, fraction=0.046, pad=0.03)
    colorbar.set_label(colorbar_label, fontsize=8)
    colorbar.ax.tick_params(labelsize=7)


def finish_axis(ax, z, val_idx):
    ax.scatter(
        z[val_idx, 0], z[val_idx, 1], s=13, facecolors="none",
        edgecolors="#202020", linewidths=0.35, alpha=0.55,
        rasterized=True,
    )
    ax.set_xlabel("z1", fontsize=8)
    ax.set_ylabel("z2", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.grid(alpha=0.12, linewidth=0.4)
    ax.set_aspect("equal", adjustable="box")


def main():
    latent_data = np.load(LATENTS)
    z = latent_data["z"]
    n_poi = latent_data["n_poi"].astype(np.float32)
    err = latent_data["err"]
    lat = latent_data["lat"]
    lon = latent_data["lon"]
    if not all(np.isfinite(v).all() for v in (z, n_poi, err, lat, lon)):
        raise ValueError("latents.npz contains NaN or infinity")

    counts = aggregate_counts(PATCHES)
    proportion = counts / np.maximum(counts.sum(axis=1, keepdims=True), 1.0)
    entropy = -np.where(
        proportion > 0,
        proportion * np.log(proportion + 1e-12),
        0.0,
    ).sum(axis=1) / np.log(N_CAT)
    dominant = proportion.argmax(axis=1)
    graph_features = np.log1p(counts)

    val_idx, recall, scores, dominant_accuracy = validation_metrics(
        z,
        graph_features,
        {
            "POI total": n_poi,
            "deviance": err,
            "entropy": entropy,
            "latitude": lat,
            "longitude": lon,
            "proportion": proportion,
        },
    )

    fig, axes = plt.subplots(2, 3, figsize=(14, 9), constrained_layout=True)
    continuous_panel(fig, axes[0, 0], z, n_poi, "POI total", "count")
    continuous_panel(fig, axes[0, 1], z, err, "Poisson deviance", "deviance")
    continuous_panel(fig, axes[0, 2], z, entropy, "Category diversity", "normalized entropy")

    categorical = axes[1, 0]
    for category in range(N_CAT):
        mask = dominant == category
        categorical.scatter(
            z[mask, 0], z[mask, 1], s=DOT,
            color=CAT_COLORS[category], label=CAT_ZH[category],
            linewidths=0, alpha=0.72, rasterized=True,
        )
    categorical.set_title("Dominant POI category", fontsize=10)
    categorical.legend(fontsize=6, ncol=2, frameon=False, markerscale=1.5)

    continuous_panel(fig, axes[1, 1], z, lat, "Geographic latitude", "degrees")
    continuous_panel(fig, axes[1, 2], z, lon, "Geographic longitude", "degrees")
    for ax in axes.flat:
        finish_axis(ax, z, val_idx)

    fig.suptitle(f"{VERSION}: what the latent space encodes", fontsize=13)
    fig.text(
        0.5,
        -0.015,
        f"held-out validation: graph-neighbor recall@{K}={recall:.3f} | "
        f"kNN R2 total={scores['POI total']:.3f}, "
        f"composition={scores['proportion']:.3f}, "
        f"deviance={scores['deviance']:.3f}, "
        f"dominant-category accuracy={dominant_accuracy:.3f}",
        ha="center",
        fontsize=8,
    )
    fig.savefig(OUT, bbox_inches="tight")

    print(f"held-out graph-neighbor recall@{K}: {recall:.4f}")
    for name, score in scores.items():
        print(f"held-out latent kNN R2 ({name}): {score:.4f}")
    print(f"held-out dominant-category accuracy: {dominant_accuracy:.4f}")
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
