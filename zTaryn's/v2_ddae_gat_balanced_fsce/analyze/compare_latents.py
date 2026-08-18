"""Compare baseline, failed composition-head, and balanced-pair latents."""

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
from config.dataset import CAT_COLORS, CAT_ZH, CATEGORIES, N_CAT, result  # noqa: E402


VERSION = "v2_ddae_gat_balanced_fsce"
LATENTS = result(VERSION, "latents.npz")
OUT = result(VERSION, "balanced_fsce_comparison.png")
METRICS_OUT = result(VERSION, "balanced_fsce_metrics.json")

MODEL_FILES = {
    "Original GAT": result("v2_ddae_gat", "latents.npz"),
    "Composition head": result("v2_ddae_gat_composition", "latents.npz"),
    "Balanced pair FSCE": LATENTS,
}
K = 15
DINING = CATEGORIES.index("Dining and Drinking")
DOT = 7

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


def latent_metrics(z, deviance, label, margin, is_train, is_val, mixed_margin):
    scaler = StandardScaler().fit(z[is_train])
    train_z = scaler.transform(z[is_train])
    val_z = scaler.transform(z[is_val])
    classifier = KNeighborsClassifier(n_neighbors=K, weights="distance")
    classifier.fit(train_z, label[is_train])
    prediction = classifier.predict(val_z)
    confident = margin[is_val] >= mixed_margin

    nearest = NearestNeighbors(n_neighbors=K).fit(train_z)
    neighbor_index = nearest.kneighbors(val_z, return_distance=False)
    neighbor_label = label[is_train][neighbor_index]
    non_dining = label[is_val] != DINING
    eigenvalues = np.linalg.eigvalsh(np.cov(z.T))
    return {
        "heldout_macro_f1": float(macro_f1(label[is_val], prediction)),
        "heldout_confident_macro_f1": float(
            macro_f1(label[is_val][confident], prediction[confident])
        ),
        "heldout_non_dining_neighbor_dining_fraction": float(
            (neighbor_label[non_dining] == DINING).mean()
        ),
        "heldout_non_dining_predicted_as_dining": float(
            (prediction[non_dining] == DINING).mean()
        ),
        "validation_clean_deviance": float(deviance[is_val].mean()),
        "pc1_variance_ratio": float(eigenvalues[-1] / eigenvalues.sum()),
    }


def categorical_panel(ax, z, label, mixed, title):
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
    ax.set_xlabel("z1")
    ax.set_ylabel("z2")
    ax.grid(alpha=0.12, linewidth=0.4)
    ax.set_aspect("equal", adjustable="box")


def bar_panel(ax, names, values, title, x_label, colors):
    y = np.arange(len(names))
    ax.barh(y, values, color=colors)
    ax.set_yticks(y, names)
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.grid(axis="x", alpha=0.15, linewidth=0.5)
    for row, value in enumerate(values):
        ax.text(value, row, f" {value:.3f}", va="center", fontsize=8)


def main():
    if not os.path.exists(LATENTS):
        raise FileNotFoundError(f"missing {LATENTS}; run train.py first")
    with np.load(LATENTS) as loaded:
        reference = {key: loaded[key] for key in loaded.files}

    composition = reference["composition_target"]
    label = reference["dominant_target"]
    margin = reference["target_margin"]
    mixed_margin = float(reference["mixed_margin"])
    mixed = margin < mixed_margin
    is_train = reference["is_train"].astype(bool)
    is_val = reference["is_val"].astype(bool)

    arrays = {}
    metrics = {}
    skipped = {}
    for name, path in MODEL_FILES.items():
        if not os.path.exists(path):
            skipped[name] = "missing file"
            continue
        with np.load(path) as loaded:
            z = loaded["z"]
            deviance = loaded["err"]
        if z.shape != reference["z"].shape or not np.isfinite(z).all():
            skipped[name] = f"invalid latent shape/content: {z.shape}"
            continue
        arrays[name] = z
        metrics[name] = latent_metrics(
            z,
            deviance,
            label,
            margin,
            is_train,
            is_val,
            mixed_margin,
        )

    fig, axes = plt.subplots(2, 3, figsize=(17, 10), constrained_layout=True)
    display_order = [
        "Original GAT",
        "Composition head",
        "Balanced pair FSCE",
    ]
    for ax, name in zip(axes[0], display_order):
        if name in arrays:
            categorical_panel(ax, arrays[name], label, mixed, name)
        else:
            ax.axis("off")
            ax.set_title(f"{name} unavailable")
    if "Balanced pair FSCE" in arrays:
        axes[0, 2].legend(
            fontsize=6,
            ncol=2,
            frameon=False,
            markerscale=1.5,
            loc="best",
        )

    names = [name for name in display_order if name in metrics]
    colors = ["#4c78a8", "#e45756", "#54a24b"][: len(names)]
    bar_panel(
        axes[1, 0],
        names,
        [metrics[name]["heldout_macro_f1"] for name in names],
        "Held-out 15-NN macro-F1",
        "macro-F1 (higher is better)",
        colors,
    )
    bar_panel(
        axes[1, 1],
        names,
        [
            metrics[name]["heldout_non_dining_neighbor_dining_fraction"]
            for name in names
        ],
        "非餐飲 validation 點的餐飲鄰居比例",
        "fraction (lower is better)",
        colors,
    )
    bar_panel(
        axes[1, 2],
        names,
        [metrics[name]["validation_clean_deviance"] for name in names],
        "Clean validation Poisson deviance",
        "deviance (lower is better)",
        colors,
    )
    fig.suptitle(
        f"Balanced composition pair geometry (mixed margin={mixed_margin:.2f})",
        fontsize=14,
    )
    fig.savefig(OUT, bbox_inches="tight")
    plt.close(fig)

    report = {
        "mixed_margin": mixed_margin,
        "n_train": int(is_train.sum()),
        "n_validation": int(is_val.sum()),
        "metrics": metrics,
        "skipped": skipped,
    }
    with open(METRICS_OUT, "w", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"saved {OUT}")
    print(f"saved {METRICS_OUT}")


if __name__ == "__main__":
    main()
