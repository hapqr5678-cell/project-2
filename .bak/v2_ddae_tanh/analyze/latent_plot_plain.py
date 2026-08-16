"""畫 latent space 的純散點圖（不上色）。v2_ddae_tanh 的 z 本來就被 tanh
夾在 (-1,1) 附近，這裡一樣做 min-max 正規化到 [-1,1]，跟 v2_dae_tanh 的
畫法保持一致，才能直接比較深/淺兩種容量下的分群形狀。
"""

import os
import sys

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(f"{os.path.dirname(__file__)}/../../.."))
from config.dataset import result  # noqa: E402

VERSION = "v2_ddae_tanh"
LATENTS = result(VERSION, "latents.npz")
OUT = result(VERSION, "latent_plot_plain.png")

DOT = 4.0

mpl.rcParams["font.family"] = ["Heiti TC"]
mpl.rcParams["axes.unicode_minus"] = False
mpl.rcParams["figure.dpi"] = 130


def normalize(v):
    """min-max 正規化到 [-1,1]。"""
    lo, hi = v.min(), v.max()
    return 2 * (v - lo) / (hi - lo) - 1


def main():
    z = np.load(LATENTS)["z"]
    zn = np.column_stack([normalize(z[:, 0]), normalize(z[:, 1])])

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(zn[:, 0], zn[:, 1], s=DOT, c="#3a6ea5", linewidths=0,
               alpha=0.6, rasterized=True)
    ax.set_title(f"{VERSION} latent space", fontsize=11)
    ax.set_xlabel("z1", fontsize=9)
    ax.set_ylabel("z2", fontsize=9)
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.tick_params(labelsize=8)
    ax.grid(alpha=0.15, linewidth=0.5)
    for s in ax.spines.values():
        s.set_alpha(0.3)
    ax.set_aspect("equal", adjustable="box")

    fig.tight_layout()
    fig.savefig(OUT, bbox_inches="tight")
    print(f"已存 {OUT}")


main()
