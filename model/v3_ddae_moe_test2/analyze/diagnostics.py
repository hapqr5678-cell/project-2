"""Summarize the selected, base, hard, soft, and oracle reconstructions."""

import os

import numpy as np


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
PATH = os.path.join(
    ROOT, "model", "v3_ddae_moe_test2", "result", "latents.npz"
)


def summarize(label, mask, data):
    selected = data["err"][mask]
    base = data["base_err"][mask]
    hard = data["hard_err"][mask]
    soft = data["soft_err"][mask]
    expert = data["expert_err"][mask]
    oracle = data["oracle_err"][mask]
    print(
        f"{label}: n={mask.sum()}, selected={selected.mean():.5f}, "
        f"base={base.mean():.5f}, hard={hard.mean():.5f}, "
        f"soft={soft.mean():.5f}, "
        f"experts={np.array2string(expert.mean(axis=0), precision=5)}, "
        f"oracle={oracle.mean():.5f}"
    )


def main():
    data = np.load(PATH)
    required = {
        "base_err",
        "hard_err",
        "soft_err",
        "selected_mode",
        "baseline_epoch",
    }
    missing = sorted(required.difference(data.files))
    if missing:
        raise RuntimeError(
            "latents.npz belongs to the previous test2 architecture; retrain "
            f"before running this diagnostic (missing: {missing})"
        )
    gates = data["gates"]
    entropy = -(gates * np.log(np.clip(gates, 1e-12, None))).sum(axis=1)
    expert_id = gates.argmax(axis=1)
    mode = str(data["selected_mode"].item())

    print(
        f"selected_mode={mode}, baseline_epoch={data['baseline_epoch'].item()}, "
        f"moe_epoch={data['best_moe_epoch'].item()}, "
        f"best_val={data['best_val_deviance'].item():.5f}"
    )
    summarize("all", np.ones(len(gates), dtype=bool), data)
    summarize("train", data["is_train"], data)
    summarize("validation", data["is_val"], data)
    print(
        "gate usage:",
        np.array2string(gates.mean(axis=0), precision=5),
        "argmax counts:",
        np.bincount(expert_id, minlength=gates.shape[1]),
    )
    print(
        f"gate entropy mean={entropy.mean():.5f}, "
        f"max-probability mean={gates.max(axis=1).mean():.5f}"
    )
    for expert in range(gates.shape[1]):
        mask = expert_id == expert
        print(
            f"expert {expert} region: n={mask.sum()}, "
            f"mean_n_poi={data['n_poi'][mask].mean():.3f}, "
            f"hard_dev={data['hard_err'][mask].mean():.5f}"
        )


if __name__ == "__main__":
    main()
