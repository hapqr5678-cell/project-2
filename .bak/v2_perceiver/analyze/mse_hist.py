"""每個類別各畫一張直方圖：x 軸是這一類的 MSE = (真實count - 重建λ)^2，
（0~該類別的最大觀測值），y 軸是有幾個 patch 落在該區間。
10 個類別排成 2x5 網格存成一張圖。
"""

import os
import sys

import numpy as np
import torch
import matplotlib as mpl
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(f"{os.path.dirname(__file__)}/../../.."))
from ae import PerceiverAE, N_CAT, Patches  # noqa: E402
from config.dataset import CAT_ZH, PATCHES, result  # noqa: E402

VERSION = "v2_perceiver"
CKPT = result(VERSION, "ae.pt")
OUT = result(VERSION, "mse_hist.png")

LATENT_DIM = 2
BINS = 40

mpl.rcParams["font.family"] = ["Heiti TC"]
mpl.rcParams["axes.unicode_minus"] = False
mpl.rcParams["figure.dpi"] = 130


def main():
    data = Patches(PATCHES)
    idx = torch.arange(data.n)
    x = data.agg(idx)
    tok, pad_mask = data.tokens(idx)

    model = PerceiverAE(LATENT_DIM)
    model.load_state_dict(torch.load(CKPT, map_location="cpu"))
    model.eval()
    with torch.no_grad():
        _, log_lam = model(tok, pad_mask)

    lam = torch.exp(log_lam)
    mse = ((x - lam) ** 2).numpy()  # (N_patch, N_CAT)

    fig, axes = plt.subplots(2, 5, figsize=(18, 7))
    for c, ax in enumerate(axes.flat):
        d = mse[:, c]
        ax.hist(d, bins=BINS, color="#3a6ea5", alpha=0.8, edgecolor="white", linewidth=0.3)
        ax.set_title(CAT_ZH[c], fontsize=10)
        ax.set_xlabel("MSE", fontsize=8)
        ax.set_ylabel("patch 數", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.15, linewidth=0.5)
        for s in ax.spines.values():
            s.set_alpha(0.3)

    fig.suptitle(f"{VERSION}：每類別的重建 MSE 分布（{data.n} 個 patch）", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT, bbox_inches="tight")
    print(f"已存 {OUT}")


main()
