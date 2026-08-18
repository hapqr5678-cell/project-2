"""Analyze soft-composition prediction and mixed patches in the new latent."""

from __future__ import annotations

import json
import os
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import f1_score
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.preprocessing import StandardScaler


sys.path.insert(0, os.path.abspath(f"{os.path.dirname(__file__)}/../../.."))
from config.dataset import (  # noqa: E402
    CAT_COLORS,
    CAT_ZH,
    CATEGORIES,
    N_CAT,
    result,
)


VERSION = "v2_ddae_gat_composition"
LATENTS = result(VERSION, "latents.npz")
OUT = result(VERSION, "latent_composition_analysis.png")
METRICS_OUT = result(VERSION, "composition_metrics.json")
BASELINE = result("v2_ddae_gat", "latents.npz")

K = 15
DINING = CATEGORIES.index("Dining and Drinking")
DOT = 8

mpl.rcParams["font.family"] = "Microsoft JhengHei"
mpl.rcParams["axes.unicode_minus"] = False
mpl.rcParams["figure.dpi"] = 140


def macro_f1(target, prediction):
    labels = np.union1d(target, prediction)
    return f1_score(
        target,
        prediction,
        labels=labels,
        average="macro",
        zero_division=0,
    )


def latent_knn_metrics(z, target_label, target_margin, is_train, is_val, margin):
    scaler = StandardScaler().fit(z[is_train])
    train_z = scaler.transform(z[is_train])
    val_z = scaler.transform(z[is_val])
    classifier = KNeighborsClassifier(n_neighbors=K, weights="distance")
    classifier.fit(train_z, target_label[is_train])
    prediction = classifier.predict(val_z)

    confident = target_margin[is_val] >= margin
    nearest = NearestNeighbors(n_neighbors=K).fit(train_z)
    neighbor_indices = nearest.kneighbors(val_z, return_distance=False)
    neighbor_labels = target_label[is_train][neighbor_indices]
    non_dining = target_label[is_val] != DINING
    return {
        "macro_f1": float(macro_f1(target_label[is_val], prediction)),
        "confident_macro_f1": float(
            macro_f1(
                target_label[is_val][confident],
                prediction[confident],
            )
        ),
        "non_dining_neighbor_dining_fraction": float(
            (neighbor_labels[non_dining] == DINING).mean()
        ),
        "non_dining_predicted_as_dining": float(
            (prediction[non_dining] == DINING).mean()
        ),
    }


def head_metrics(data):
    target_probability = data["composition_target"]
    predicted_probability = data["composition_pred"]
    target_label = data["dominant_target"]
    predicted_label = data["dominant_pred"]
    target_margin = data["target_margin"]
    predicted_margin = data["predicted_margin"]
    is_val = data["is_val"].astype(bool)
    margin = float(data["mixed_margin"])

    confident = is_val & (target_margin >= margin)
    target_mixed = target_label.copy()
    predicted_mixed = predicted_label.copy()
    target_mixed[target_margin < margin] = N_CAT
    predicted_mixed[predicted_margin < margin] = N_CAT
    non_dining = is_val & (target_label != DINING)
    return {
        "validation_composition_js": float(data["composition_js"][is_val].mean()),
        "validation_composition_mae": float(
            np.abs(
                predicted_probability[is_val] - target_probability[is_val]
            ).mean()
        ),
        "validation_hard_macro_f1": float(
            macro_f1(target_label[is_val], predicted_label[is_val])
        ),
        "validation_confident_macro_f1": float(
            macro_f1(target_label[confident], predicted_label[confident])
        ),
        "validation_mixed_aware_macro_f1": float(
            macro_f1(target_mixed[is_val], predicted_mixed[is_val])
        ),
        "validation_true_mixed_fraction": float(
            (target_margin[is_val] < margin).mean()
        ),
        "validation_predicted_mixed_fraction": float(
            (predicted_margin[is_val] < margin).mean()
        ),
        "validation_non_dining_predicted_as_dining": float(
            (predicted_label[non_dining] == DINING).mean()
        ),
        "validation_clean_deviance": float(data["err"][is_val].mean()),
    }


def categorical_panel(ax, z, label, mixed, title):
    # Dining is drawn first; rare categories and Mixed remain visible above it.
    for category in range(N_CAT):
        mask = (label == category) & ~mixed
        ax.scatter(
            z[mask, 0],
            z[mask, 1],
            s=DOT,
            color=CAT_COLORS[category],
            label=CAT_ZH[category],
            alpha=0.7,
            linewidths=0,
            rasterized=True,
        )
    ax.scatter(
        z[mixed, 0],
        z[mixed, 1],
        s=DOT + 2,
        color="#777777",
        label="混合",
        alpha=0.65,
        linewidths=0,
        rasterized=True,
    )
    ax.set_title(title)
    ax.legend(fontsize=6, ncol=2, frameon=False, markerscale=1.5)


def continuous_panel(fig, ax, z, value, title, label, cmap="viridis"):
    points = ax.scatter(
        z[:, 0],
        z[:, 1],
        c=value,
        cmap=cmap,
        s=DOT,
        alpha=0.72,
        linewidths=0,
        rasterized=True,
    )
    ax.set_title(title)
    colorbar = fig.colorbar(points, ax=ax, fraction=0.046, pad=0.03)
    colorbar.set_label(label, fontsize=8)


def finish_axis(ax):
    ax.set_xlabel("z1")
    ax.set_ylabel("z2")
    ax.grid(alpha=0.12, linewidth=0.4)
    ax.set_aspect("equal", adjustable="box")


def main():
    if not os.path.exists(LATENTS):
        raise FileNotFoundError(f"missing {LATENTS}; run train.py first")
    with np.load(LATENTS) as loaded:
        data = {key: loaded[key] for key in loaded.files}

    required = {
        "z",
        "composition_target",
        "composition_pred",
        "composition_js",
        "dominant_target",
        "dominant_pred",
        "target_margin",
        "predicted_margin",
        "is_train",
        "is_val",
        "mixed_margin",
        "err",
    }
    missing = required - set(data)
    if missing:
        raise ValueError(f"latents.npz missing fields: {sorted(missing)}")
    if not all(np.isfinite(data[key]).all() for key in required if key != "is_train" and key != "is_val"):
        raise ValueError("latents.npz contains NaN or infinity")

    z = data["z"]
    target_probability = data["composition_target"]
    predicted_probability = data["composition_pred"]
    target_label = data["dominant_target"]
    predicted_label = data["dominant_pred"]
    target_margin = data["target_margin"]
    predicted_margin = data["predicted_margin"]
    is_train = data["is_train"].astype(bool)
    is_val = data["is_val"].astype(bool)
    margin = float(data["mixed_margin"])
    target_mixed = target_margin < margin
    predicted_mixed = predicted_margin < margin

    metrics = {
        "mixed_margin": margin,
        "composition_head": head_metrics(data),
        "new_latent_knn": latent_knn_metrics(
            z,
            target_label,
            target_margin,
            is_train,
            is_val,
            margin,
        ),
    }
    if os.path.exists(BASELINE):
        with np.load(BASELINE) as baseline:
            baseline_z = baseline["z"]
        if baseline_z.shape == z.shape and np.isfinite(baseline_z).all():
            metrics["baseline_gat_latent_knn"] = latent_knn_metrics(
                baseline_z,
                target_label,
                target_margin,
                is_train,
                is_val,
                margin,
            )

    fig, axes = plt.subplots(2, 3, figsize=(16, 10), constrained_layout=True)
    categorical_panel(
        axes[0, 0], z, target_label, target_mixed, "Target：dominant + 混合"
    )
    categorical_panel(
        axes[0, 1], z, predicted_label, predicted_mixed, "Composition head 預測"
    )
    continuous_panel(
        fig,
        axes[0, 2],
        z,
        target_probability[:, DINING],
        "真實餐飲比例",
        "target proportion",
        cmap="Reds",
    )
    continuous_panel(
        fig,
        axes[1, 0],
        z,
        predicted_probability[:, DINING],
        "預測餐飲比例",
        "predicted proportion",
        cmap="Reds",
    )
    continuous_panel(
        fig,
        axes[1, 1],
        z,
        data["composition_js"],
        "Composition Jensen-Shannon error",
        "JS divergence",
        cmap="magma",
    )
    continuous_panel(
        fig,
        axes[1, 2],
        z,
        data["err"],
        "Clean Poisson reconstruction deviance",
        "deviance",
        cmap="viridis",
    )
    for ax in axes.flat:
        finish_axis(ax)
    fig.suptitle(
        f"{VERSION}: soft composition and mixed patches (margin={margin:.2f})",
        fontsize=14,
    )
    fig.savefig(OUT, bbox_inches="tight")
    plt.close(fig)

    with open(METRICS_OUT, "w", encoding="utf-8") as stream:
        json.dump(metrics, stream, ensure_ascii=False, indent=2)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"saved {OUT}")
    print(f"saved {METRICS_OUT}")


if __name__ == "__main__":
    main()
