"""不經任何模型，純粹統計 patches.npz：每個類別各畫一張直方圖，
x 軸是該類別在網格內的數量（整數刻度），y 軸是有幾個網格落在該數量。
10 個類別排成 2x5 網格存成一張圖，所有子圖共用同一個 x 軸範圍以方便互相比較。
"""

import os
import sys

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

sys.path.insert(0, os.path.abspath(f"{os.path.dirname(__file__)}/../.."))
from config.dataset import CAT_ZH, N_CAT, PATCHES, result  # noqa: E402

OUT = result("v0", "count_hist.png")

mpl.rcParams["font.family"] = ["Heiti TC"]
mpl.rcParams["axes.unicode_minus"] = False
mpl.rcParams["figure.dpi"] = 130


def main():
    p = np.load(PATCHES)
    cat, offsets, n = p["cat"], p["offsets"], len(p["n_poi"])

    owner = np.repeat(np.arange(n), np.diff(offsets))
    counts = np.zeros((n, N_CAT), dtype=np.int64)
    np.add.at(counts, (owner, cat), 1)

    x_max = counts.max()
    bins = np.arange(-0.5, x_max + 1.5)

    fig, axes = plt.subplots(2, 5, figsize=(18, 7))
    for c, ax in enumerate(axes.flat):
        ax.hist(counts[:, c], bins=bins, color="#3a6ea5", alpha=0.8,
                 edgecolor="white", linewidth=0.3)
        ax.set_xlim(-0.5, x_max + 0.5)
        ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
        ax.set_title(CAT_ZH[c], fontsize=10)
        ax.set_xlabel("數量", fontsize=8)
        ax.set_ylabel("網格數", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.15, linewidth=0.5)
        for s in ax.spines.values():
            s.set_alpha(0.3)

    fig.suptitle(f"每類別數量分布（{n} 個網格，x 軸範圍統一為 0~{x_max}）", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT, bbox_inches="tight")
    print(f"已存 {OUT}")


main()
