"""Compare baseline and residual-MoE training logs."""

import os
import re

import matplotlib.pyplot as plt
import numpy as np


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
BASELINE_LOG = os.path.join(
    ROOT, "model", "v2_ddae_fsce", "result", "result_base.log"
)
MOE_LOG = os.path.join(
    ROOT, "model", "v3_ddae_moe_test", "result", "result.log"
)
OUT = os.path.join(
    ROOT, "model", "v3_ddae_moe_test", "result", "compare.png"
)
PATTERN = re.compile(
    r"epoch\s+(\d+).*?train_nll\s+([-\d.]+).*?val_dev\s+([-\d.]+)"
)
INITIAL_PATTERN = re.compile(
    r"epoch\s+0.*?val_dev\s+([-\d.]+)"
)


def parse(path):
    rows = []
    with open(path, encoding="utf-8", errors="replace") as handle:
        for line in handle:
            match = PATTERN.search(line)
            if match:
                rows.append(tuple(map(float, match.groups())))
    if not rows:
        raise ValueError(f"no metrics found in {path}")
    return np.asarray(rows)


def parse_initial_baseline(path):
    with open(path, encoding="utf-8", errors="replace") as handle:
        for line in handle:
            match = INITIAL_PATTERN.search(line)
            if match:
                return float(match.group(1))
    raise ValueError(f"epoch-0 baseline metric not found in {path}")


def summarize(label, values):
    best = values[np.argmin(values[:, 2])]
    tail = values[values[:, 0] >= max(values[-1, 0] - 100, 0)]
    print(
        f"{label}: best val_dev={best[2]:.5f} at epoch={int(best[0])}; "
        f"last-100 mean={tail[:, 2].mean():.5f}"
    )


def main():
    baseline = parse(BASELINE_LOG)
    moe = parse(MOE_LOG)
    same_run_baseline = parse_initial_baseline(MOE_LOG)
    summarize("baseline", baseline)
    summarize("residual MoE", moe)
    best_moe = moe[:, 2].min()
    print(
        f"same-run epoch-0 baseline={same_run_baseline:.5f}; "
        f"best MoE delta={best_moe - same_run_baseline:+.5f}"
    )

    figure, (train_axis, val_axis) = plt.subplots(
        2, 1, figsize=(8, 8), sharex=True
    )
    for label, values in (("baseline", baseline), ("residual MoE", moe)):
        train_axis.plot(values[:, 0], values[:, 1], label=label)
        val_axis.plot(values[:, 0], values[:, 2], label=label)
    val_axis.axhline(
        same_run_baseline,
        color="black",
        linestyle="--",
        linewidth=1,
        label="same-run epoch-0 baseline",
    )
    train_axis.set_ylabel("train_nll")
    val_axis.set_ylabel("val_dev")
    val_axis.set_xlabel("epoch")
    for axis in (train_axis, val_axis):
        axis.grid(True, alpha=0.3)
        axis.legend()
    figure.tight_layout()
    figure.savefig(OUT, dpi=150)
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
