"""每個類別各畫一張直方圖：x 軸是這一類的 Poisson deviance（0~該類別的最大
觀測值），y 軸是有幾個 patch 落在該區間。10 個類別排成 2x5 網格存成一張圖。

用全部 patch（不是單一 --n）算每個 patch、每個類別各自的 deviance
（不對類別取平均，跟 latents.npz 裡存的 err 是「跨類別平均後」的版本不同）。
"""

import os
import sys

import numpy as np
import torch
import matplotlib as mpl
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(f"{os.path.dirname(__file__)}/../../.."))
from ae import VAE, N_CAT, Patches  # noqa: E402
from config.dataset import CAT_ZH, PATCHES, result  # noqa: E402

VERSION = "v2_vae"
CKPT = result(VERSION, "ae.pt")
OUT = result(VERSION, "loss_hist.png")

LATENT_DIM = 2
BINS = 40

mpl.rcParams["font.family"] = ["Heiti TC"]
mpl.rcParams["axes.unicode_minus"] = False
mpl.rcParams["figure.dpi"] = 130


def main():
    data = Patches(PATCHES)
    idx = torch.arange(data.n)
    x = data.agg(idx)

    model = VAE(LATENT_DIM)
    model.load_state_dict(torch.load(CKPT, map_location="cpu"))
    model.eval()
    with torch.no_grad():
        _, _, _, log_lam = model(x)   # eval 模式下 z = mu，決定性

    lam = torch.exp(log_lam)
    dev = (2 * (torch.xlogy(x, x) - x * log_lam - x + lam)).numpy()  # (N_patch, N_CAT)

    fig, axes = plt.subplots(2, 5, figsize=(18, 7))
    for c, ax in enumerate(axes.flat):
        d = dev[:, c]
        ax.hist(d, bins=BINS, color="#3a6ea5", alpha=0.8, edgecolor="white", linewidth=0.3)
        ax.set_title(CAT_ZH[c], fontsize=10)
        ax.set_xlabel("deviance", fontsize=8)
        ax.set_ylabel("patch 數", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.15, linewidth=0.5)
        for s in ax.spines.values():
            s.set_alpha(0.3)

    fig.suptitle(f"{VERSION}：每類別的重建 deviance 分布（{data.n} 個 patch）", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT, bbox_inches="tight")
    print(f"已存 {OUT}")


main()
