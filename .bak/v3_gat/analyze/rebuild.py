"""重建單一 patch：載入這個版本的 checkpoint，畫「真實 count vs 重建 λ」
長條圖，用 --n 指定 patch 編號（0 起算）。同時印出這個 patch 的 alpha
（CrossAttentionFusion 學到的 GAT 分支混合比例）。
"""

import argparse
import os
import sys

import numpy as np
import torch
import matplotlib as mpl
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(f"{os.path.dirname(__file__)}/.."))
sys.path.insert(0, os.path.abspath(f"{os.path.dirname(__file__)}/../../.."))
from ae import GATAE, ODMatrices, Patches, poisson_deviance, poisson_nll  # noqa: E402
from config.dataset import CAT_COLORS, CAT_ZH, N_CAT, OD, PATCHES, result  # noqa: E402

VERSION = "v3_gat"
LATENT_DIM = 2   # v2/v3 系列固定 latent_dim=2

mpl.rcParams["font.family"] = ["Heiti TC"]
mpl.rcParams["axes.unicode_minus"] = False
mpl.rcParams["figure.dpi"] = 130


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=0, help="patch 編號（0 起算）")
    n = ap.parse_args().n

    data = Patches(PATCHES)
    od = ODMatrices(OD)
    assert 0 <= n < data.n, f"--n 要在 0~{data.n - 1}"

    model = GATAE(LATENT_DIM)
    model.load_state_dict(torch.load(result(VERSION, "ae.pt"), map_location="cpu"))
    model.eval()

    idx = torch.tensor([n])
    x = data.agg(idx)
    od_batch = od.get(idx)
    with torch.no_grad():
        z, log_lam, alpha = model(x, od_batch)
    cnt = x[0].numpy()
    lam = torch.exp(log_lam)[0].numpy()
    dev = poisson_deviance(log_lam, x).item()
    nll = poisson_nll(log_lam, x).item()

    err = np.load(result(VERSION, "latents.npz"))["err"]
    pct = (err < dev).mean() * 100

    print(f"patch {n}：POI {int(data.n_poi[n])}  "
          f"({float(data.lat[n]):.5f}, {float(data.lon[n]):.5f})")
    print(f"latent z = ({z[0, 0]:+.3f}, {z[0, 1]:+.3f})")
    print(f"alpha = {alpha.item():.4f}")
    print(f"deviance = {dev:.6f}（全體中位數 {np.median(err):.6f}，"
          f"此 patch 排在第 {pct:.1f} 百分位）")
    print(f"NLL = {nll:.6f}（省略 log(y!) 常數項，可能為負）")

    fig, ax = plt.subplots(figsize=(9, 5.5))
    idx_c = np.arange(N_CAT)
    w = 0.38
    ax.bar(idx_c - w / 2, cnt, width=w, color=CAT_COLORS, alpha=0.55,
           label="真實 count")
    ax.bar(idx_c + w / 2, np.round(lam), width=w, color=CAT_COLORS, alpha=0.95,
           hatch="//", edgecolor="white", label="重建 λ")
    ax.set_xticks(idx_c)
    ax.set_xticklabels(CAT_ZH, fontsize=9, rotation=20, ha="right")
    ax.set_ylabel("個數", fontsize=9)
    ax.legend(fontsize=9, framealpha=0.9)
    ax.grid(alpha=0.15, linewidth=0.5, axis="y")
    for s in ax.spines.values():
        s.set_alpha(0.3)
    ax.set_title(f"{VERSION} patch {n}（deviance {dev:.4f}，alpha {alpha.item():.3f}，"
                 f"POI {int(data.n_poi[n])} 個）", fontsize=11)

    fig.tight_layout()
    out = result(VERSION, "rebuild.png")
    fig.savefig(out, bbox_inches="tight")
    print(f"已存 {out}")


main()
