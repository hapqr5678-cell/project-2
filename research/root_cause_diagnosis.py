"""Diagnose why POI dominant categories overlap before changing the model.

This script performs no training.  It isolates five possible causes:

1. source category imbalance and unstable dominant labels;
2. information lost by compressing raw features to two dimensions;
3. information lost by the learned count/OD/fused latent branches;
4. secondary non-Dining structure hidden by the dominant-label plot;
5. whether the OD graph is category-homophilous enough to help separation.

Outputs are written to ``research/root_cause_results``.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    r2_score,
)
from sklearn.neighbors import (
    KNeighborsClassifier,
    KNeighborsRegressor,
    NearestNeighbors,
)
from sklearn.preprocessing import StandardScaler
from umap import UMAP


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from config.dataset import CAT_COLORS, CAT_ZH, CATEGORIES, N_CAT, PATCHES  # noqa: E402


OUTPUT = ROOT / "research" / "root_cause_results"
GAT_LATENTS = ROOT / "model" / "v2_ddae_gat" / "result" / "latents.npz"
OD_CSV = ROOT / "data" / "odGraph.csv"
K = 15
DINING = CATEGORIES.index("Dining and Drinking")
MIXED_MARGIN = 0.15
RANDOM_PERMUTATIONS = 100

mpl.rcParams["font.family"] = "Microsoft JhengHei"
mpl.rcParams["axes.unicode_minus"] = False
mpl.rcParams["figure.dpi"] = 140


def aggregate_counts(path):
    with np.load(path) as data:
        category = data["cat"].astype(np.int64)
        offsets = data["offsets"].astype(np.int64)
    counts = np.zeros((len(offsets) - 1, N_CAT), dtype=np.float64)
    for patch in range(len(counts)):
        counts[patch] = np.bincount(
            category[offsets[patch] : offsets[patch + 1]],
            minlength=N_CAT,
        )
    return counts


def macro_f1(target, prediction):
    labels = np.union1d(target, prediction)
    return f1_score(
        target,
        prediction,
        labels=labels,
        average="macro",
        zero_division=0,
    )


def heldout_knn_metrics(
    features,
    labels,
    is_train,
    is_val,
    valid=None,
    dining_label=None,
):
    if valid is None:
        valid = np.ones(len(labels), dtype=bool)
    train_mask = is_train & valid
    val_mask = is_val & valid
    scaler = StandardScaler().fit(features[train_mask])
    train_x = scaler.transform(features[train_mask])
    val_x = scaler.transform(features[val_mask])
    classifier = KNeighborsClassifier(n_neighbors=K, weights="distance")
    classifier.fit(train_x, labels[train_mask])
    prediction = classifier.predict(val_x)

    nearest = NearestNeighbors(n_neighbors=K).fit(train_x)
    neighbor_index = nearest.kneighbors(val_x, return_distance=False)
    neighbor_labels = labels[train_mask][neighbor_index]
    same_neighbor_fraction = (neighbor_labels == labels[val_mask, None]).mean(axis=1)
    present = np.unique(labels[val_mask])
    macro_neighbor_purity = np.mean(
        [same_neighbor_fraction[labels[val_mask] == category].mean() for category in present]
    )

    result = {
        "n_train": int(train_mask.sum()),
        "n_validation": int(val_mask.sum()),
        "accuracy": float(accuracy_score(labels[val_mask], prediction)),
        "balanced_accuracy": float(
            balanced_accuracy_score(labels[val_mask], prediction)
        ),
        "macro_f1": float(macro_f1(labels[val_mask], prediction)),
        "macro_neighbor_purity": float(macro_neighbor_purity),
    }
    if dining_label is not None:
        non_dining = labels[val_mask] != dining_label
        result["non_dining_neighbor_dining_fraction"] = float(
            (neighbor_labels[non_dining] == dining_label).mean()
        )
        result["non_dining_predicted_as_dining"] = float(
            (prediction[non_dining] == dining_label).mean()
        )
    return result


def pca_two_dimensions(features, is_train):
    scaler = StandardScaler().fit(features[is_train])
    standardized = scaler.transform(features)
    pca = PCA(n_components=2).fit(standardized[is_train])
    return pca.transform(standardized), float(pca.explained_variance_ratio_.sum())


def umap_two_dimensions(features, is_train, metric):
    model = UMAP(
        n_components=2,
        n_neighbors=15,
        min_dist=0.1,
        metric=metric,
        random_state=0,
        transform_seed=0,
        n_jobs=1,
    )
    embedding = np.empty((len(features), 2), dtype=np.float32)
    embedding[is_train] = model.fit_transform(features[is_train])
    embedding[~is_train] = model.transform(features[~is_train])
    return embedding


def heldout_information_metrics(features, counts, proportion, is_train, is_val):
    scaler = StandardScaler().fit(features[is_train])
    train_x = scaler.transform(features[is_train])
    val_x = scaler.transform(features[is_val])
    regressor = KNeighborsRegressor(n_neighbors=K, weights="distance")
    regressor.fit(train_x, np.log1p(counts.sum(axis=1))[is_train])
    total_prediction = regressor.predict(val_x)
    regressor.fit(train_x, proportion[is_train])
    composition_prediction = regressor.predict(val_x)
    return {
        "total_log_count_r2": float(
            r2_score(
                np.log1p(counts.sum(axis=1))[is_val],
                total_prediction,
            )
        ),
        "composition_r2": float(
            r2_score(
                proportion[is_val],
                composition_prediction,
                multioutput="variance_weighted",
            )
        ),
    }


def make_representations(counts, proportion, is_train):
    nonzero = counts + 0.5
    log_fraction = np.log(nonzero / nonzero.sum(axis=1, keepdims=True))
    clr = log_fraction - log_fraction.mean(axis=1, keepdims=True)
    base = {
        "log1p counts (10D)": np.log1p(counts),
        "composition (10D)": proportion,
        "Hellinger composition (10D)": np.sqrt(proportion),
        "CLR composition (10D)": clr,
    }
    representations = dict(base)
    pca_variance = {}
    for name, features in base.items():
        short = name.replace(" (10D)", "") + " PCA-2D"
        representations[short], pca_variance[short] = pca_two_dimensions(
            features, is_train
        )
    representations["composition UMAP-2D"] = umap_two_dimensions(
        proportion, is_train, metric="cosine"
    )
    representations["Hellinger composition UMAP-2D"] = umap_two_dimensions(
        np.sqrt(proportion), is_train, metric="euclidean"
    )
    representations["log1p counts UMAP-2D"] = umap_two_dimensions(
        np.log1p(counts), is_train, metric="cosine"
    )
    return representations, pca_variance


def representation_table(
    representations,
    labels,
    margin,
    is_train,
    is_val,
    pca_variance,
    kind,
    counts,
    proportion,
):
    rows = []
    confident = margin >= MIXED_MARGIN
    supported_categories = np.flatnonzero(
        np.bincount(labels, minlength=N_CAT) >= 10
    )
    supported = np.isin(labels, supported_categories)
    for name, features in representations.items():
        metrics = heldout_knn_metrics(
            features,
            labels,
            is_train,
            is_val,
            dining_label=DINING,
        )
        confident_metrics = heldout_knn_metrics(
            features,
            labels,
            is_train,
            is_val,
            valid=confident,
            dining_label=DINING,
        )
        supported_metrics = heldout_knn_metrics(
            features,
            labels,
            is_train,
            is_val,
            valid=supported,
            dining_label=DINING,
        )
        information = heldout_information_metrics(
            features,
            counts,
            proportion,
            is_train,
            is_val,
        )
        rows.append(
            {
                "representation": name,
                "kind": kind,
                "dimensions": features.shape[1],
                "pca_explained_variance": pca_variance.get(name, np.nan),
                **metrics,
                **information,
                "confident_macro_f1": confident_metrics["macro_f1"],
                "supported_class_macro_f1": supported_metrics["macro_f1"],
            }
        )
    return rows


def secondary_category_metrics(representations, proportion, is_train, is_val):
    non_dining_mass = proportion[:, 1:].sum(axis=1)
    valid = non_dining_mass > 0
    conditional = proportion[:, 1:] / np.maximum(non_dining_mass[:, None], 1e-12)
    secondary = conditional.argmax(axis=1)
    rows = []
    for name, features in representations.items():
        metrics = heldout_knn_metrics(
            features,
            secondary,
            is_train,
            is_val,
            valid=valid,
        )
        rows.append(
            {
                "representation": name,
                "dimensions": features.shape[1],
                **metrics,
            }
        )
    return pd.DataFrame(rows), conditional, secondary, valid


def weighted_mean(values, weights):
    return float(np.average(values, weights=weights))


def od_graph_metrics(edges, proportion, labels):
    category_lookup = {name: index for index, name in enumerate(CATEGORIES)}
    origin_category = edges["origin_category"].map(category_lookup).to_numpy()
    destination_category = edges["destination_category"].map(category_lookup).to_numpy()
    if np.isnan(origin_category).any() or np.isnan(destination_category).any():
        raise ValueError("OD graph contains a category outside config.dataset")
    origin_category = origin_category.astype(np.int64)
    destination_category = destination_category.astype(np.int64)
    weight = edges["trip_count"].to_numpy(dtype=np.float64)

    transition = np.zeros((N_CAT, N_CAT), dtype=np.float64)
    np.add.at(transition, (origin_category, destination_category), weight)
    origin_prior = transition.sum(axis=1) / transition.sum()
    destination_prior = transition.sum(axis=0) / transition.sum()
    observed_same_category = transition.trace() / transition.sum()
    expected_same_category = float((origin_prior * destination_prior).sum())

    origin_patch = edges["origin_patch_id"].to_numpy(dtype=np.int64)
    destination_patch = edges["destination_patch_id"].to_numpy(dtype=np.int64)
    patch_embedding = np.sqrt(proportion)
    patch_similarity = (
        patch_embedding[origin_patch] * patch_embedding[destination_patch]
    ).sum(axis=1)
    observed_patch_similarity = weighted_mean(patch_similarity, weight)
    observed_same_dominant = weighted_mean(
        labels[origin_patch] == labels[destination_patch], weight
    )

    rng = np.random.default_rng(0)
    null_similarity = []
    null_same_dominant = []
    for _ in range(RANDOM_PERMUTATIONS):
        shuffled = rng.permutation(destination_patch)
        similarity = (
            patch_embedding[origin_patch] * patch_embedding[shuffled]
        ).sum(axis=1)
        null_similarity.append(weighted_mean(similarity, weight))
        null_same_dominant.append(
            weighted_mean(labels[origin_patch] == labels[shuffled], weight)
        )

    row_sum = transition.sum(axis=1, keepdims=True)
    normalized_transition = np.divide(
        transition,
        row_sum,
        out=np.zeros_like(transition),
        where=row_sum > 0,
    )
    dining_involved = (origin_category == DINING) | (
        destination_category == DINING
    )
    return {
        "n_edges": int(len(edges)),
        "trip_count_sum": int(weight.sum()),
        "same_category_edge_fraction_unweighted": float(
            (origin_category == destination_category).mean()
        ),
        "same_category_trip_fraction": float(observed_same_category),
        "expected_same_category_trip_fraction_from_marginals": expected_same_category,
        "category_assortativity_lift": float(
            observed_same_category / max(expected_same_category, 1e-12)
        ),
        "dining_involved_trip_fraction": weighted_mean(dining_involved, weight),
        "observed_patch_hellinger_similarity": observed_patch_similarity,
        "random_patch_hellinger_similarity_mean": float(np.mean(null_similarity)),
        "random_patch_hellinger_similarity_std": float(np.std(null_similarity)),
        "patch_similarity_lift": float(
            observed_patch_similarity / max(np.mean(null_similarity), 1e-12)
        ),
        "observed_same_dominant_trip_fraction": observed_same_dominant,
        "random_same_dominant_trip_fraction_mean": float(
            np.mean(null_same_dominant)
        ),
        "random_same_dominant_trip_fraction_std": float(
            np.std(null_same_dominant)
        ),
        "transition": normalized_transition,
    }


def plot_results(
    category_frame,
    representation_frame,
    secondary_frame,
    od_metrics,
):
    fig, axes = plt.subplots(2, 3, figsize=(18, 10), constrained_layout=True)
    x = np.arange(N_CAT)
    width = 0.38
    axes[0, 0].bar(
        x - width / 2,
        category_frame["poi_count_share"],
        width,
        label="POI count share",
    )
    axes[0, 0].bar(
        x + width / 2,
        category_frame["dominant_patch_share"],
        width,
        label="dominant patch share",
    )
    axes[0, 0].set_title("資料與 dominant label 不平衡")
    axes[0, 0].set_xticks(x, CAT_ZH, rotation=45, ha="right")
    axes[0, 0].set_ylabel("fraction")
    axes[0, 0].legend(frameon=False, fontsize=8)

    raw = representation_frame[representation_frame["kind"] == "raw"].copy()
    raw = raw.sort_values("macro_f1")
    axes[0, 1].barh(raw["representation"], raw["macro_f1"], color="#4c78a8")
    axes[0, 1].set_title("原始特徵 vs PCA-2D：held-out macro-F1")
    axes[0, 1].set_xlabel("macro-F1")

    model = representation_frame[representation_frame["kind"] == "model"].copy()
    model = model.sort_values("macro_f1")
    axes[0, 2].barh(model["representation"], model["macro_f1"], color="#54a24b")
    axes[0, 2].set_title("Original GAT 各分支：held-out macro-F1")
    axes[0, 2].set_xlabel("macro-F1")

    transition = od_metrics["transition"]
    image = axes[1, 0].imshow(transition, vmin=0, vmax=transition.max(), cmap="Blues")
    axes[1, 0].set_title("OD trip-weighted category transition")
    axes[1, 0].set_xlabel("destination")
    axes[1, 0].set_ylabel("origin")
    axes[1, 0].set_xticks(x, CAT_ZH, rotation=45, ha="right", fontsize=7)
    axes[1, 0].set_yticks(x, CAT_ZH, fontsize=7)
    fig.colorbar(image, ax=axes[1, 0], fraction=0.046, pad=0.03)

    names = ["Observed OD", "Random destination"]
    patch_values = [
        od_metrics["observed_patch_hellinger_similarity"],
        od_metrics["random_patch_hellinger_similarity_mean"],
    ]
    dominant_values = [
        od_metrics["observed_same_dominant_trip_fraction"],
        od_metrics["random_same_dominant_trip_fraction_mean"],
    ]
    positions = np.arange(2)
    axes[1, 1].bar(
        positions - width / 2,
        patch_values,
        width,
        label="patch composition similarity",
    )
    axes[1, 1].bar(
        positions + width / 2,
        dominant_values,
        width,
        label="same dominant fraction",
    )
    axes[1, 1].set_xticks(positions, names)
    axes[1, 1].set_ylim(0, 1)
    axes[1, 1].set_title("OD edge 是否連接相似 POI 組成")
    axes[1, 1].legend(frameon=False, fontsize=8)

    secondary = secondary_frame.sort_values("macro_f1")
    axes[1, 2].barh(
        secondary["representation"],
        secondary["macro_f1"],
        color="#f58518",
    )
    axes[1, 2].set_title("排除餐飲後的 secondary category")
    axes[1, 2].set_xlabel("held-out macro-F1")

    for axis in axes.flat:
        axis.grid(alpha=0.12, linewidth=0.5)
    fig.suptitle("POI latent overlap root-cause diagnosis", fontsize=14)
    fig.savefig(OUTPUT / "root_cause_diagnosis.png", bbox_inches="tight")
    plt.close(fig)


def plot_geometry_tradeoff(gat_z, composition_z, labels, margin, total_count):
    fig, axes = plt.subplots(2, 2, figsize=(13, 10), constrained_layout=True)
    mixed = margin < MIXED_MARGIN
    for axis, z, title in (
        (axes[0, 0], gat_z, "Original GAT：dominant + mixed"),
        (axes[0, 1], composition_z, "Training-only composition UMAP-2D"),
    ):
        for category in range(N_CAT):
            mask = (labels == category) & ~mixed
            axis.scatter(
                z[mask, 0],
                z[mask, 1],
                s=7,
                color=CAT_COLORS[category],
                label=CAT_ZH[category],
                alpha=0.7,
                linewidths=0,
                rasterized=True,
            )
        axis.scatter(
            z[mixed, 0],
            z[mixed, 1],
            s=8,
            color="#777777",
            label="混合",
            alpha=0.55,
            linewidths=0,
            rasterized=True,
        )
        axis.set_title(title)
        axis.set_xlabel("dimension 1")
        axis.set_ylabel("dimension 2")
        axis.grid(alpha=0.12, linewidth=0.4)
    axes[0, 1].legend(fontsize=6, ncol=2, frameon=False, markerscale=1.5)

    for axis, z, title in (
        (axes[1, 0], gat_z, "Original GAT：log(1 + total POI)"),
        (axes[1, 1], composition_z, "Composition UMAP：log(1 + total POI)"),
    ):
        points = axis.scatter(
            z[:, 0],
            z[:, 1],
            c=np.log1p(total_count),
            cmap="viridis",
            s=7,
            alpha=0.72,
            linewidths=0,
            rasterized=True,
        )
        axis.set_title(title)
        axis.set_xlabel("dimension 1")
        axis.set_ylabel("dimension 2")
        axis.grid(alpha=0.12, linewidth=0.4)
        colorbar = fig.colorbar(points, ax=axis, fraction=0.046, pad=0.03)
        colorbar.set_label("log1p total count", fontsize=8)
    fig.suptitle("Composition geometry and count-reconstruction trade-off", fontsize=14)
    fig.savefig(OUTPUT / "composition_geometry_tradeoff.png", bbox_inches="tight")
    plt.close(fig)


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    if not GAT_LATENTS.exists():
        raise FileNotFoundError(f"missing original GAT latents: {GAT_LATENTS}")
    if not OD_CSV.exists():
        raise FileNotFoundError(f"missing OD graph: {OD_CSV}")

    counts = aggregate_counts(PATCHES)
    total = counts.sum(axis=1, keepdims=True)
    proportion = counts / np.maximum(total, 1.0)
    labels = proportion.argmax(axis=1)
    ordered = np.sort(proportion, axis=1)
    margin = ordered[:, -1] - ordered[:, -2]

    with np.load(GAT_LATENTS) as latent_data:
        is_train = latent_data["is_train"].astype(bool)
        is_val = latent_data["is_val"].astype(bool)
        model_representations = {
            "Original GAT fused": latent_data["z"],
            "Original GAT count branch": latent_data["z_count"],
            "Original GAT OD branch": latent_data["z_od"],
        }

    raw_representations, pca_variance = make_representations(
        counts, proportion, is_train
    )
    rows = representation_table(
        raw_representations,
        labels,
        margin,
        is_train,
        is_val,
        pca_variance,
        kind="raw",
        counts=counts,
        proportion=proportion,
    )
    rows += representation_table(
        model_representations,
        labels,
        margin,
        is_train,
        is_val,
        {},
        kind="model",
        counts=counts,
        proportion=proportion,
    )
    representation_frame = pd.DataFrame(rows)
    representation_frame.to_csv(
        OUTPUT / "representation_metrics.csv", index=False
    )

    secondary_representations = {
        "Non-Dining composition (9D)": (
            proportion[:, 1:]
            / np.maximum(proportion[:, 1:].sum(axis=1, keepdims=True), 1e-12)
        ),
        **model_representations,
    }
    secondary_representations["Non-Dining composition PCA-2D"], _ = (
        pca_two_dimensions(
            secondary_representations["Non-Dining composition (9D)"],
            is_train,
        )
    )
    secondary_frame, _, secondary_label, secondary_valid = (
        secondary_category_metrics(
            secondary_representations,
            proportion,
            is_train,
            is_val,
        )
    )
    secondary_frame.to_csv(OUTPUT / "secondary_category_metrics.csv", index=False)

    poi_share = counts.sum(axis=0) / counts.sum()
    dominant_count = np.bincount(labels, minlength=N_CAT)
    category_frame = pd.DataFrame(
        {
            "category": CATEGORIES,
            "category_zh": CAT_ZH,
            "poi_count": counts.sum(axis=0).astype(int),
            "poi_count_share": poi_share,
            "dominant_patch_count": dominant_count,
            "dominant_patch_share": dominant_count / len(counts),
            "patch_presence_share": (counts > 0).mean(axis=0),
        }
    )
    # The secondary count belongs to categories 1..N_CAT-1; shift it explicitly.
    category_frame["secondary_patch_count"] = np.concatenate(
        [
            [0],
            np.bincount(
                secondary_label[secondary_valid], minlength=N_CAT - 1
            ),
        ]
    )
    category_frame.to_csv(OUTPUT / "category_support.csv", index=False)

    edges = pd.read_csv(OD_CSV)
    od_metrics = od_graph_metrics(edges, proportion, labels)
    pd.DataFrame(
        od_metrics["transition"], index=CAT_ZH, columns=CAT_ZH
    ).to_csv(OUTPUT / "od_transition_weighted.csv")

    raw_comp = representation_frame.loc[
        representation_frame["representation"] == "composition (10D)"
    ].iloc[0]
    raw_comp_pca = representation_frame.loc[
        representation_frame["representation"] == "composition PCA-2D"
    ].iloc[0]
    raw_comp_umap = representation_frame.loc[
        representation_frame["representation"] == "composition UMAP-2D"
    ].iloc[0]
    gat_fused = representation_frame.loc[
        representation_frame["representation"] == "Original GAT fused"
    ].iloc[0]
    best_secondary = secondary_frame.loc[secondary_frame["macro_f1"].idxmax()]
    report = {
        "n_patches": len(counts),
        "dominant_label": {
            "dining_poi_count_share": float(poi_share[DINING]),
            "dining_dominant_patch_share": float(
                dominant_count[DINING] / len(counts)
            ),
            "mixed_patch_share_margin_below_0_15": float(
                (margin < MIXED_MARGIN).mean()
            ),
            "zero_dominant_categories": [
                CATEGORIES[index]
                for index in np.flatnonzero(dominant_count == 0)
            ],
        },
        "two_dimensional_bottleneck": {
            "raw_composition_10d_macro_f1": float(raw_comp["macro_f1"]),
            "raw_composition_pca2_macro_f1": float(raw_comp_pca["macro_f1"]),
            "pca2_explained_variance": float(
                raw_comp_pca["pca_explained_variance"]
            ),
            "macro_f1_loss_from_pca2": float(
                raw_comp["macro_f1"] - raw_comp_pca["macro_f1"]
            ),
            "raw_composition_umap2_macro_f1": float(raw_comp_umap["macro_f1"]),
            "umap2_total_log_count_r2": float(
                raw_comp_umap["total_log_count_r2"]
            ),
            "umap2_composition_r2": float(raw_comp_umap["composition_r2"]),
        },
        "original_gat": {
            "fused_macro_f1": float(gat_fused["macro_f1"]),
            "fused_non_dining_neighbor_dining_fraction": float(
                gat_fused["non_dining_neighbor_dining_fraction"]
            ),
            "macro_f1_gap_from_raw_composition": float(
                raw_comp["macro_f1"] - gat_fused["macro_f1"]
            ),
            "total_log_count_r2": float(gat_fused["total_log_count_r2"]),
            "composition_r2": float(gat_fused["composition_r2"]),
            "macro_f1_gap_from_composition_umap2": float(
                raw_comp_umap["macro_f1"] - gat_fused["macro_f1"]
            ),
        },
        "secondary_non_dining": {
            "valid_patch_count": int(secondary_valid.sum()),
            "best_representation": str(best_secondary["representation"]),
            "best_macro_f1": float(best_secondary["macro_f1"]),
            "original_gat_fused_macro_f1": float(
                secondary_frame.loc[
                    secondary_frame["representation"] == "Original GAT fused",
                    "macro_f1",
                ].iloc[0]
            ),
        },
        "od_graph": {
            key: value
            for key, value in od_metrics.items()
            if key != "transition"
        },
    }
    with open(OUTPUT / "summary.json", "w", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)

    plot_results(
        category_frame,
        representation_frame,
        secondary_frame,
        od_metrics,
    )
    composition_umap = raw_representations["composition UMAP-2D"]
    plot_geometry_tradeoff(
        model_representations["Original GAT fused"],
        composition_umap,
        labels,
        margin,
        counts.sum(axis=1),
    )
    np.savez(
        OUTPUT / "diagnostic_embeddings.npz",
        original_gat=model_representations["Original GAT fused"],
        composition_umap=composition_umap,
        dominant_label=labels,
        margin=margin,
        total_count=counts.sum(axis=1),
        is_train=is_train,
        is_val=is_val,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print("\nDominant-category representations:")
    print(
        representation_frame.sort_values("macro_f1", ascending=False).to_string(
            index=False
        )
    )
    print("\nSecondary non-Dining category:")
    print(secondary_frame.sort_values("macro_f1", ascending=False).to_string(index=False))
    print(f"saved results to {OUTPUT}")


if __name__ == "__main__":
    main()
