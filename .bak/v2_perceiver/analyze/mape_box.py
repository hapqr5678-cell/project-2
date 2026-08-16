"""把每個類別的 MAPE 分布畫成箱型圖，10 個類別並排在同一張圖裡，
x 軸=類別，y 軸=MAPE(%)，方便直接比較類別之間的差異。

真實 count 為 0 的 patch 分母無意義，該類別計算 MAPE 時直接排除。
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
OUT = result(VERSION, "mape_box.png")

LATENT_DIM = 2

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
    x_np, lam_np = x.numpy(), lam.numpy()

    mape_per_cat = []
    for c in range(N_CAT):
        mask = x_np[:, c] > 0
        mape_per_cat.append(np.abs(x_np[mask, c] - lam_np[mask, c]) / x_np[mask, c] * 100)

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.boxplot(mape_per_cat, showfliers=True, patch_artist=True,
               boxprops=dict(facecolor="#3a6ea5", alpha=0.6),
               medianprops=dict(color="#1f2d3d"),
               flierprops=dict(markersize=3, markerfacecolor="#3a6ea5", markeredgewidth=0, alpha=0.4))
    ax.set_xticklabels(CAT_ZH, fontsize=9, rotation=20, ha="right")
    ax.set_ylabel("MAPE (%)", fontsize=9)
    ax.tick_params(labelsize=8)
    ax.grid(alpha=0.15, linewidth=0.5, axis="y")
    for s in ax.spines.values():
        s.set_alpha(0.3)

    fig.suptitle(f"{VERSION}：每類別的重建 MAPE 箱型圖（{data.n} 個 patch）", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT, bbox_inches="tight")
    print(f"已存 {OUT}")


main()
