"""Reproducible, read-only diagnostics for the 2-D count autoencoders.

The script deliberately uses only NumPy and the Python standard library so it
can run independently of the training environment.  It never overwrites model
artifacts.  All generated files are written next to this script in ``output``.
"""

from __future__ import annotations

import csv
import json
import math
import re
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
PATCHES = ROOT / "data" / "patch" / "patches.npz"
OUTPUT = Path(__file__).resolve().parent / "output"
N_BOOTSTRAP = 10_000
K_NEIGHBORS = 15
RNG_SEED = 20260818


def aggregate_counts(path: Path) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    with np.load(path) as data:
        categories = data["cat"].astype(np.int64)
        offsets = data["offsets"].astype(np.int64)
        metadata = {
            "n_poi": data["n_poi"].copy(),
            "lat": data["center_lat"].copy(),
            "lon": data["center_lon"].copy(),
        }
    n = len(offsets) - 1
    n_categories = int(categories.max()) + 1
    counts = np.zeros((n, n_categories), dtype=np.float64)
    for row in range(n):
        counts[row] = np.bincount(
            categories[offsets[row] : offsets[row + 1]],
            minlength=n_categories,
        )
    return counts, metadata


def discover_latents(n: int) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for path in sorted((ROOT / "model").glob("*/result/latents.npz")):
        with np.load(path) as data:
            if "err" in data and len(data["err"]) == n:
                paths[path.parents[1].name] = path
    return paths


def common_split(paths: dict[str, Path], n: int) -> tuple[np.ndarray, np.ndarray, str]:
    preferred = ["v3_ddae_moe_test2", "v3_ddae_moe_test", "v3_ddae_mot"]
    for name in preferred:
        path = paths.get(name)
        if path is None:
            continue
        with np.load(path) as data:
            if "is_train" in data and "is_val" in data:
                train = data["is_train"].astype(bool)
                val = data["is_val"].astype(bool)
                if len(train) == n and len(val) == n and not np.any(train & val):
                    return train, val, f"saved masks from {name}"
    rng = np.random.default_rng(0)
    permutation = rng.permutation(n)
    val = np.zeros(n, dtype=bool)
    val[permutation[: int(0.1 * n)]] = True
    return ~val, val, "NumPy seed-0 fallback (not PyTorch-identical)"


def poisson_deviance(counts: np.ndarray, rates: np.ndarray) -> np.ndarray:
    rates = np.clip(rates, 1e-10, 1e10)
    log_ratio = np.zeros_like(counts, dtype=np.float64)
    positive = counts > 0
    log_ratio[positive] = np.log(counts[positive] / rates[positive])
    cells = 2.0 * (counts * log_ratio - counts + rates)
    return cells.mean(axis=1)


def paired_bootstrap(
    candidate: np.ndarray,
    baseline: np.ndarray,
    rng: np.random.Generator,
) -> dict[str, float]:
    delta = candidate - baseline
    n = len(delta)
    sampled = np.empty(N_BOOTSTRAP, dtype=np.float64)
    batch = 500
    for start in range(0, N_BOOTSTRAP, batch):
        size = min(batch, N_BOOTSTRAP - start)
        indices = rng.integers(0, n, size=(size, n))
        sampled[start : start + size] = delta[indices].mean(axis=1)
    low, high = np.quantile(sampled, [0.025, 0.975])
    return {
        "mean_delta": float(delta.mean()),
        "ci_low": float(low),
        "ci_high": float(high),
        "probability_better": float((sampled < 0).mean()),
    }


def squared_distances(x: np.ndarray) -> np.ndarray:
    norms = np.sum(x * x, axis=1)
    result = norms[:, None] + norms[None, :] - 2.0 * x @ x.T
    np.maximum(result, 0.0, out=result)
    np.fill_diagonal(result, 0.0)
    return result


def intrinsic_dimension(x: np.ndarray, ks=(5, 10, 15, 20, 30)) -> dict[int, float]:
    distances = np.sqrt(squared_distances(x))
    ordered = np.sort(distances, axis=1)[:, 1:]
    estimates: dict[int, float] = {}
    for k in ks:
        radius = np.maximum(ordered[:, k - 1], 1e-12)
        inner = np.maximum(ordered[:, : k - 1], 1e-12)
        denominator = np.log(radius[:, None] / inner).mean(axis=1)
        local = 1.0 / np.maximum(denominator, 1e-12)
        finite = local[np.isfinite(local) & (local < 100)]
        estimates[k] = float(np.median(finite))
    return estimates


def pca_rate_distortion(
    counts: np.ndarray,
    train: np.ndarray,
    val: np.ndarray,
) -> list[dict[str, float]]:
    transformed = np.log1p(counts)
    mean = transformed[train].mean(axis=0)
    scale = transformed[train].std(axis=0)
    scale[scale < 1e-8] = 1.0
    standardized = (transformed - mean) / scale
    _, singular, vt = np.linalg.svd(standardized[train], full_matrices=False)
    total_variance = np.sum(singular**2)
    rows = []
    for dimension in range(1, counts.shape[1] + 1):
        components = vt[:dimension]
        reconstructed_standard = (
            standardized[val] @ components.T @ components
        )
        reconstructed_log = reconstructed_standard * scale + mean
        reconstructed = np.maximum(np.expm1(reconstructed_log), 0.0)
        deviance = poisson_deviance(counts[val], reconstructed).mean()
        explained = np.sum(singular[:dimension] ** 2) / total_variance
        rows.append(
            {
                "dimension": dimension,
                "explained_variance": float(explained),
                "val_poisson_deviance": float(deviance),
            }
        )
    return rows


def neighbor_metrics(high_distance: np.ndarray, z: np.ndarray, k: int) -> dict[str, float]:
    n = len(z)
    low_distance = squared_distances((z - z.mean(axis=0)) / np.maximum(z.std(axis=0), 1e-8))
    high_order = np.argsort(high_distance, axis=1)
    low_order = np.argsort(low_distance, axis=1)
    high_knn = high_order[:, 1 : k + 1]
    low_knn = low_order[:, 1 : k + 1]
    high_ranks = np.empty_like(high_order)
    low_ranks = np.empty_like(low_order)
    row_index = np.arange(n)[:, None]
    high_ranks[row_index, high_order] = np.arange(n)[None, :]
    low_ranks[row_index, low_order] = np.arange(n)[None, :]

    trust_penalty = 0.0
    continuity_penalty = 0.0
    overlaps = []
    for row in range(n):
        high_set = set(high_knn[row].tolist())
        low_set = set(low_knn[row].tolist())
        overlaps.append(len(high_set & low_set) / k)
        for point in low_set - high_set:
            trust_penalty += high_ranks[row, point] - k
        for point in high_set - low_set:
            continuity_penalty += low_ranks[row, point] - k
    normalizer = 2.0 / (n * k * (2 * n - 3 * k - 1))
    return {
        "trustworthiness": float(1.0 - normalizer * trust_penalty),
        "continuity": float(1.0 - normalizer * continuity_penalty),
        "knn_overlap": float(np.mean(overlaps)),
    }


def parse_training_logs() -> list[dict[str, float | int | str]]:
    pattern = re.compile(r"epoch\s+(\d+).*?val_dev\s+([0-9]+(?:\.[0-9]+)?)")
    rows = []
    for path in sorted((ROOT / "model").glob("*/result/*.log")):
        values = [(int(epoch), float(value)) for epoch, value in pattern.findall(path.read_text(encoding="utf-8", errors="ignore"))]
        if not values:
            continue
        epochs = np.asarray([item[0] for item in values])
        deviance = np.asarray([item[1] for item in values])
        best = int(np.argmin(deviance))
        tail = deviance[epochs >= max(800, int(0.8 * epochs.max()))]
        if not len(tail):
            tail = deviance[-max(1, len(deviance) // 5) :]
        rows.append(
            {
                "model": path.parents[1].name,
                "log": path.name,
                "n_evaluations": len(values),
                "best_epoch": int(epochs[best]),
                "best_val_deviance": float(deviance[best]),
                "final_val_deviance": float(deviance[-1]),
                "tail_mean": float(tail.mean()),
                "tail_std": float(tail.std(ddof=1)) if len(tail) > 1 else 0.0,
            }
        )
    return rows


def count_diagnostics(counts: np.ndarray, train: np.ndarray) -> dict[str, object]:
    train_counts = counts[train]
    means = train_counts.mean(axis=0)
    variance = train_counts.var(axis=0, ddof=1)
    ratios = variance / np.maximum(means, 1e-12)
    total = train_counts.sum(axis=1)
    return {
        "category_mean": means.tolist(),
        "category_variance_to_mean": ratios.tolist(),
        "zero_fraction": (train_counts == 0).mean(axis=0).tolist(),
        "total_variance_to_mean": float(total.var(ddof=1) / total.mean()),
    }


def graph_scope_diagnostics(features: np.ndarray, train: np.ndarray, val: np.ndarray, k: int) -> dict[str, float]:
    """Approximate the leakage surface of an all-data Euclidean kNN graph.

    The production fuzzy graph has weighted/symmetrized edges, so these directed
    kNN figures are an interpretable lower-level diagnostic, not an exact edge
    count from UMAP.
    """
    order = np.argsort(squared_distances(features), axis=1)[:, 1 : k + 1]
    source_is_val = np.repeat(val[:, None], k, axis=1)
    target_is_val = val[order]
    touching_val = source_is_val | target_is_val
    train_targets = target_is_val[train]
    return {
        "directed_edges": int(len(features) * k),
        "fraction_edges_touching_validation": float(touching_val.mean()),
        "fraction_train_neighbors_in_validation": float(train_targets.mean()),
        "fraction_validation_neighbors_in_train": float((~target_is_val[val]).mean()),
        "negative_pair_probability_touches_validation": float(1.0 - (train.mean() ** 2)),
    }


def moe_diagnostics(path: Path, counts: np.ndarray, train: np.ndarray, val: np.ndarray) -> dict[str, object]:
    with np.load(path) as data:
        result: dict[str, object] = {}
        z = data["z"].astype(np.float64)
        total = np.log1p(counts.sum(axis=1))
        result["latent_total_correlations"] = [
            float(np.corrcoef(z[:, column], total)[0, 1])
            for column in range(z.shape[1])
        ]
        if "gates" in data:
            gates = np.clip(data["gates"].astype(np.float64), 1e-12, 1.0)
            result["gate_usage"] = gates[val].mean(axis=0).tolist()
            result["gate_entropy"] = float((-(gates[val] * np.log(gates[val])).sum(axis=1)).mean())
            result["gate_max_probability"] = float(gates[val].max(axis=1).mean())
        if "routed_log_lam" in data:
            rates = np.exp(data["routed_log_lam"].astype(np.float64))
            val_rates = rates[val]
            val_counts = counts[val]
            predicted_total = val_rates.sum(axis=1)
            observed_total = val_counts.sum(axis=1)
            exact_total_rates = val_rates * (
                observed_total / np.maximum(predicted_total, 1e-12)
            )[:, None]
            result["selected_val_deviance"] = float(poisson_deviance(val_counts, val_rates).mean())
            result["exact_total_rescaled_val_deviance"] = float(poisson_deviance(val_counts, exact_total_rates).mean())
            result["total_log_correlation"] = float(np.corrcoef(np.log1p(observed_total), np.log1p(predicted_total))[0, 1])
            result["total_relative_mae"] = float(np.mean(np.abs(predicted_total - observed_total) / np.maximum(observed_total, 1.0)))
            pearson = ((val_counts - val_rates) ** 2 / np.maximum(val_rates, 1e-10)).mean(axis=0)
            result["conditional_pearson_dispersion"] = pearson.tolist()
            result["observed_zero_fraction"] = (val_counts == 0).mean(axis=0).tolist()
            result["poisson_expected_zero_fraction"] = np.exp(-val_rates).mean(axis=0).tolist()
            total_dev = poisson_deviance(observed_total[:, None], predicted_total[:, None])
            full_dev_sum = poisson_deviance(val_counts, val_rates) * counts.shape[1]
            result["deviance_share_total"] = float(total_dev.sum() / full_dev_sum.sum())
            result["deviance_share_composition"] = float(1.0 - total_dev.sum() / full_dev_sum.sum())
        for key in ("err", "uniform_err", "oracle_err"):
            if key in data:
                result[f"{key}_val_mean"] = float(data[key][val].mean())
        if "expert_err" in data:
            result["expert_err_val_mean"] = data["expert_err"][val].mean(axis=0).tolist()
        return result


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def svg_line_chart(path: Path, rows: list[dict[str, float]], x_key: str, y_key: str, title: str) -> None:
    width, height = 760, 440
    left, right, top, bottom = 75, 30, 55, 65
    xs = np.asarray([row[x_key] for row in rows], dtype=float)
    ys = np.asarray([row[y_key] for row in rows], dtype=float)
    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()
    y_pad = max((y_max - y_min) * 0.08, 1e-3)
    y_min, y_max = y_min - y_pad, y_max + y_pad

    def sx(value: float) -> float:
        return left + (value - x_min) / max(x_max - x_min, 1e-12) * (width - left - right)

    def sy(value: float) -> float:
        return top + (y_max - value) / max(y_max - y_min, 1e-12) * (height - top - bottom)

    points = " ".join(f"{sx(x):.1f},{sy(y):.1f}" for x, y in zip(xs, ys))
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width/2}" y="30" text-anchor="middle" font-family="sans-serif" font-size="20">{title}</text>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height-bottom}" stroke="#333"/>',
        f'<line x1="{left}" y1="{height-bottom}" x2="{width-right}" y2="{height-bottom}" stroke="#333"/>',
    ]
    for tick in np.linspace(y_min, y_max, 6):
        y = sy(tick)
        parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{width-right}" y2="{y:.1f}" stroke="#ddd"/>')
        parts.append(f'<text x="{left-8}" y="{y+4:.1f}" text-anchor="end" font-family="sans-serif" font-size="12">{tick:.3f}</text>')
    for x in xs:
        parts.append(f'<text x="{sx(x):.1f}" y="{height-bottom+22}" text-anchor="middle" font-family="sans-serif" font-size="12">{int(x)}</text>')
    parts.append(f'<polyline points="{points}" fill="none" stroke="#275dad" stroke-width="3"/>')
    for x, y in zip(xs, ys):
        parts.append(f'<circle cx="{sx(x):.1f}" cy="{sy(y):.1f}" r="4" fill="#e4572e"/>')
    parts.append(f'<text x="{width/2}" y="{height-18}" text-anchor="middle" font-family="sans-serif" font-size="14">latent dimension (diagnostic only)</text>')
    parts.append('</svg>')
    path.write_text("\n".join(parts), encoding="utf-8")


def build_report(summary: dict[str, object]) -> str:
    pca = summary["pca_rate_distortion"]
    comparisons = summary["model_comparison"]
    intrinsic = summary["intrinsic_dimension"]
    moe = summary.get("moe_test2", {})
    geometry = summary["latent_geometry"]
    count = summary["count_diagnostics"]
    graph = summary["all_data_graph_scope"]
    lines = [
        "# AE quantitative diagnostic report",
        "",
        f"Generated from {summary['n_patches']} patches; validation n={summary['n_validation']} ({summary['split_source']}).",
        "",
        "## Decision summary",
        "",
        "1. Treat the current 0.61387 result as a transductive/single-run reference, not an inductive target, until it is reproduced with train-only graph construction and multiple seeds.",
        f"2. The data intrinsic dimension is about {intrinsic['standardized_log1p']['15']:.2f} at k=15, while the required latent is 2-D.",
        f"3. Linear log-count PCA improves from {pca[1]['val_poisson_deviance']:.4f} at 2-D to {pca[3]['val_poisson_deviance']:.4f} at 4-D. This is evidence for a bottleneck before the decoder.",
    ]
    if moe:
        lines.append(
            f"4. In old test2, exact observed-total rescaling changes deviance from {moe.get('selected_val_deviance', math.nan):.4f} to {moe.get('exact_total_rescaled_val_deviance', math.nan):.4f}; total and composition errors should be modeled and reported separately."
        )
        lines.append(
            f"5. The saved MoE expert oracle is {moe.get('oracle_err_val_mean', math.nan):.4f}, versus routed {moe.get('err_val_mean', math.nan):.4f}; these experts do not contain a hidden better solution for the router to discover."
        )
    lines.extend(["", "## Existing model comparison (same saved validation mask)", "", "Lower deviance is better. CI is a paired patch bootstrap against the currently saved v2 checkpoint.", "", "| model | val deviance | delta vs v2 | 95% CI | P(better) | trust | continuity | kNN overlap |", "|---|---:|---:|---:|---:|---:|---:|---:|"])
    for row in comparisons:
        ci = f"[{row['ci_low']:.4f}, {row['ci_high']:.4f}]" if row["model"] != "v2_ddae_fsce" else "reference"
        probability = f"{row['probability_better']:.3f}" if row["model"] != "v2_ddae_fsce" else "-"
        metrics = geometry[row["model"]]
        lines.append(
            f"| {row['model']} | {row['val_deviance']:.5f} | {row['mean_delta']:+.5f} | {ci} | {probability} | {metrics['trustworthiness']:.3f} | {metrics['continuity']:.3f} | {metrics['knn_overlap']:.3f} |"
        )
    lines.extend(["", "## Rate-distortion diagnostic", "", "Other dimensions are diagnostics only; the production representation can remain 2-D.", "", "| dimension | explained variance | validation Poisson deviance |", "|---:|---:|---:|"])
    for row in pca:
        lines.append(f"| {row['dimension']} | {row['explained_variance']:.3f} | {row['val_poisson_deviance']:.5f} |")
    lines.extend(["", "## Count diagnostics", "", f"- Category marginal variance/mean range: {min(count['category_variance_to_mean']):.3f}–{max(count['category_variance_to_mean']):.3f}.", f"- Total-count variance/mean: {count['total_variance_to_mean']:.3f}."])
    if moe:
        lines.extend([
            f"- Conditional Pearson dispersion range under old test2: {min(moe['conditional_pearson_dispersion']):.3f}–{max(moe['conditional_pearson_dispersion']):.3f}.",
            f"- Deviance share: total {moe['deviance_share_total']:.1%}, composition {moe['deviance_share_composition']:.1%}.",
            f"- Validation gate usage: {np.array2string(np.asarray(moe['gate_usage']), precision=3)}; entropy {moe['gate_entropy']:.3f}.",
        ])
    lines.extend([
        "",
        "## All-data graph contamination surface",
        "",
        f"- {graph['fraction_edges_touching_validation']:.1%} of directed kNN edges touch validation.",
        f"- {graph['fraction_train_neighbors_in_validation']:.1%} of neighbors attached to a training source are validation points.",
        f"- With the old all-data negative sampler, a random pair touches validation with probability {graph['negative_pair_probability_touches_validation']:.1%}.",
        "- These are valid transductive edges only if the intended task is embedding this one fixed complete dataset. They are contamination for inductive validation.",
    ])
    lines.extend([
        "",
        "## Next experiment gate",
        "",
        "- First reproduce the train-only-graph 2-D baseline with 5 seeds and one untouched test split. Report mean, SD, and paired CI; do not select models by the minimum of the same validation curve.",
        "- Add a dimension sweep to the same trainer. If the 2-D to 4-D gap persists across seeds, decoder MoE is not the next priority; test a multi-chart 2-D representation.",
        "- Measure encoder gradient cosine between reconstruction and FSCE before adding PCGrad. Only use gradient surgery if negative-gradient frequency is substantial.",
        "- Compare Poisson and negative-binomial/factorized total-composition decoders only after the protocol above is fixed.",
        "",
        "## Important limitations",
        "",
        "- Saved checkpoints come from different training protocols and some from different devices. This report can diagnose failure modes but cannot certify a winning architecture.",
        "- Exact-total rescaling uses the observed target total and is therefore an oracle diagnostic, not a fair deployable model unless total count is explicitly allowed as an input side channel.",
        "- Marginal variance/mean does not by itself prove conditional negative-binomial dispersion.",
    ])
    return "\n".join(lines) + "\n"


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    counts, _ = aggregate_counts(PATCHES)
    paths = discover_latents(len(counts))
    if "v2_ddae_fsce" not in paths:
        raise FileNotFoundError("v2_ddae_fsce/result/latents.npz is required")
    train, val, split_source = common_split(paths, len(counts))
    transformed = np.log1p(counts)
    mean = transformed[train].mean(axis=0)
    scale = np.maximum(transformed[train].std(axis=0), 1e-8)
    standardized = (transformed - mean) / scale
    composition = (counts + 0.5) / (counts + 0.5).sum(axis=1, keepdims=True)
    clr = np.log(composition)
    clr -= clr.mean(axis=1, keepdims=True)
    high_distance = squared_distances(standardized)

    rng = np.random.default_rng(RNG_SEED)
    with np.load(paths["v2_ddae_fsce"]) as baseline_data:
        baseline_error = baseline_data["err"][val].astype(np.float64)

    comparison_rows = []
    geometry: dict[str, dict[str, float]] = {}
    for name, path in paths.items():
        with np.load(path) as data:
            error = data["err"][val].astype(np.float64)
            z = data["z"].astype(np.float64)
        boot = paired_bootstrap(error, baseline_error, rng)
        if name == "v2_ddae_fsce":
            boot = {"mean_delta": 0.0, "ci_low": 0.0, "ci_high": 0.0, "probability_better": 0.0}
        comparison_rows.append({"model": name, "val_deviance": float(error.mean()), **boot})
        geometry[name] = neighbor_metrics(high_distance, z, K_NEIGHBORS)
    comparison_rows.sort(key=lambda row: row["val_deviance"])

    pca = pca_rate_distortion(counts, train, val)
    log_rows = parse_training_logs()
    summary: dict[str, object] = {
        "n_patches": len(counts),
        "n_train": int(train.sum()),
        "n_validation": int(val.sum()),
        "split_source": split_source,
        "intrinsic_dimension": {
            "standardized_log1p": {str(k): v for k, v in intrinsic_dimension(standardized).items()},
            "clr_composition": {str(k): v for k, v in intrinsic_dimension(clr).items()},
        },
        "pca_rate_distortion": pca,
        "count_diagnostics": count_diagnostics(counts, train),
        "all_data_graph_scope": graph_scope_diagnostics(transformed, train, val, K_NEIGHBORS),
        "model_comparison": comparison_rows,
        "latent_geometry": geometry,
        "training_logs": log_rows,
    }
    if "v3_ddae_moe_test2" in paths:
        summary["moe_test2"] = moe_diagnostics(paths["v3_ddae_moe_test2"], counts, train, val)

    (OUTPUT / "diagnostics.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    write_csv(OUTPUT / "model_comparison.csv", comparison_rows)
    write_csv(OUTPUT / "pca_rate_distortion.csv", pca)
    write_csv(OUTPUT / "training_log_stability.csv", log_rows)
    svg_line_chart(OUTPUT / "rate_distortion.svg", pca, "dimension", "val_poisson_deviance", "Rate-distortion diagnostic")
    (OUTPUT / "report.md").write_text(build_report(summary), encoding="utf-8")
    print(f"Wrote diagnostics to {OUTPUT}")


if __name__ == "__main__":
    main()
