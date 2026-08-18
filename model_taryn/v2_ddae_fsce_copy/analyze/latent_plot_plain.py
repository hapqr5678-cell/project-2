"""畫 latent space 的純散點圖（不上色），直接用 encoder 輸出的原始座標，
不做任何正規化——座標軸上的數字就是 latent 的真實尺度。

跟 v2_dae_fsce 的同名腳本一樣刻意不正規化：v2_ddae_fsce_copy 的 encoder 出口
沒有 tanh，latent 能跑多遠本身就是要看的資訊，正規化會把這件事整個抹掉。
"""

import os
import sys

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(f"{os.path.dirname(__file__)}/../../.."))
from config.dataset import result  # noqa: E402

VERSION = "v2_ddae_fsce_copy"
LATENTS = result(VERSION, "latents.npz")
OUT = result(VERSION, "latent_plot_plain.png")

DOT = 4.0

mpl.rcParams["font.family"] = ["Heiti TC"]
mpl.rcParams["axes.unicode_minus"] = False
mpl.rcParams["figure.dpi"] = 130


def main():
    z = np.load(LATENTS)["z"]

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(z[:, 0], z[:, 1], s=DOT, c="#3a6ea5", linewidths=0,
               alpha=0.6, rasterized=True)
    ax.set_title(f"{VERSION} latent space（原始座標）", fontsize=11)
    ax.set_xlabel("z1", fontsize=9)
    ax.set_ylabel("z2", fontsize=9)
    ax.tick_params(labelsize=8)
    ax.grid(alpha=0.15, linewidth=0.5)
    for s in ax.spines.values():
        s.set_alpha(0.3)
    ax.set_aspect("equal", adjustable="box")

    fig.tight_layout()
    fig.savefig(OUT, bbox_inches="tight")
    print(f"已存 {OUT}")


main()
