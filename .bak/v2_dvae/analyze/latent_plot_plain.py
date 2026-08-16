"""畫 latent space 的純散點圖（不上色），跟其他版本的同名圖對照形狀。

K=3 時 theta 落在一個三角形（2-simplex）上，自由度剛好 2，跟 v2_vae /
v2_gvae 的 latent_dim=2 同一個容量，所以這張圖跟它們的同名圖是可以直接
並排比較的。三角座標轉換沒有丟資訊（見 ternary.py），不是投影。

不上色是刻意的：跟 v2_vae 的同一張圖對照，就能看出 Dirichlet 先驗有沒有
真的把點推向三角形的角落（稀疏 alpha 的預期效果），而不是靠顏色暗示。
分群結果要看 cluster_plot.py。
"""

import os
import sys

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.abspath(f"{os.path.dirname(__file__)}/../../.."))
from ternary import frame, to_xy  # noqa: E402
from config.dataset import result  # noqa: E402

VERSION = "v2_dvae"
LATENTS = result(VERSION, "latents.npz")
OUT = result(VERSION, "latent_plot_plain.png")

DOT = 4.0

mpl.rcParams["font.family"] = ["Heiti TC"]
mpl.rcParams["axes.unicode_minus"] = False
mpl.rcParams["figure.dpi"] = 130


def main():
    d = np.load(LATENTS)
    theta, alpha = d["z"], float(d["alpha"])
    if theta.shape[1] != 3:
        raise SystemExit(f"這張圖只支援 K=3，latents.npz 的 K={theta.shape[1]}")
    xy = to_xy(theta)

    conf = theta.max(1)
    print(f"alpha={alpha:.3f}，主成分佔比中位數 {np.median(conf):.3f}，"
          f"平均每個 patch 用到 {(theta > 0.05).sum(1).mean():.2f} 個 archetype")

    fig, ax = plt.subplots(figsize=(6, 6))
    frame(ax)
    ax.scatter(xy[:, 0], xy[:, 1], s=DOT, c="#3a6ea5", linewidths=0,
               alpha=0.6, rasterized=True, zorder=2)
    ax.set_title(f"{VERSION} latent space（2-simplex，alpha={alpha:.2f}）",
                 fontsize=11)

    fig.tight_layout()
    fig.savefig(OUT, bbox_inches="tight")
    print(f"已存 {OUT}")


main()
