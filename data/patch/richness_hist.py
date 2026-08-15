"""不經任何模型，純粹統計 patches.npz：每個網格有幾種不同類別的分布。
x 軸是類別數（整數刻度），y 軸是網格數。
"""

import os
import sys

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

sys.path.insert(0, os.path.abspath(f"{os.path.dirname(__file__)}/../.."))
from config.dataset import N_CAT, PATCHES, result  # noqa: E402

OUT = result("v0", "richness_hist.png")

mpl.rcParams["font.family"] = ["Heiti TC"]
mpl.rcParams["axes.unicode_minus"] = False
mpl.rcParams["figure.dpi"] = 130


def main():
    p = np.load(PATCHES)
    cat, offsets, n = p["cat"], p["offsets"], len(p["n_poi"])

    owner = np.repeat(np.arange(n), np.diff(offsets))
    counts = np.zeros((n, N_CAT), dtype=np.int64)
    np.add.at(counts, (owner, cat), 1)

    richness = (counts > 0).sum(axis=1)  # 每個網格出現過的類別數

    fig, ax = plt.subplots(figsize=(7, 5))
    bins = np.arange(-0.5, N_CAT + 1.5)
    ax.hist(richness, bins=bins, color="#3a6ea5", alpha=0.8, edgecolor="white", linewidth=0.3)
    ax.set_xlim(-0.5, N_CAT + 0.5)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(1))
    ax.set_xlabel("網格內出現的類別數", fontsize=9)
    ax.set_ylabel("網格數", fontsize=9)
    ax.tick_params(labelsize=8)
    ax.grid(alpha=0.15, linewidth=0.5)
    for s in ax.spines.values():
        s.set_alpha(0.3)

    fig.suptitle(f"每網格類別數量分布（{n} 個網格，共 {N_CAT} 類）", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT, bbox_inches="tight")
    print(f"已存 {OUT}")


main()
