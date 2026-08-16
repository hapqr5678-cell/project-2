"""挑一個 patch 丟進訓練好的 v2_vae AE，比較「進去」與「出來」的類別 count。

用 --n 指定 patch 編號（0 起算）。

v2 系列的輸入沒有空間結構，「重建」比的不是點的位置，而是每一類的
期望個數：decoder 輸出 log λ，exp 之後就是「這一類的 POI 期望個數」，
直接跟真實 count 畫成長條圖比較即可，不需要 v0 那種螺旋排列的散點技巧。
"""

import argparse
import os
import sys

import numpy as np
import torch
import matplotlib as mpl
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(f"{os.path.dirname(__file__)}/../../.."))
from ae import VAE, N_CAT, Patches, poisson_deviance, poisson_nll  # noqa: E402
from config.dataset import CAT_COLORS, CAT_ZH, PATCHES, result  # noqa: E402

VERSION = "v2_vae"
LATENTS = result(VERSION, "latents.npz")
CKPT = result(VERSION, "ae.pt")
OUT = result(VERSION, "rebuild_test.png")

LATENT_DIM = 2     # v2 系列固定 latent_dim=2，三個版本才能互相比較

mpl.rcParams["font.family"] = ["Heiti TC"]
mpl.rcParams["axes.unicode_minus"] = False
mpl.rcParams["figure.dpi"] = 130


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=0, help="patch 編號（0 起算）")
    n = ap.parse_args().n

    data = Patches(PATCHES)
    assert 0 <= n < data.n, f"--n 要在 0~{data.n - 1}"

    model = VAE(LATENT_DIM)
    model.load_state_dict(torch.load(CKPT, map_location="cpu"))
    model.eval()

    x = data.agg(torch.tensor([n]))
    with torch.no_grad():
        _, _, z, log_lam = model(x)   # eval 模式下 z = mu，決定性
    dev = poisson_deviance(log_lam, x).item()
    nll = poisson_nll(log_lam, x).item()
    lam = np.round(torch.exp(log_lam)[0].numpy())
    cnt = x[0].numpy()

    err = np.load(LATENTS)["err"]
    pct = (err < dev).mean() * 100
    n_poi = int(data.n_poi[n])

    print(f"patch {n}：POI {n_poi}  ({data.lat[n]:.5f}, {data.lon[n]:.5f})")
    print(f"latent z = ({z[0, 0]:+.3f}, {z[0, 1]:+.3f})")
    print(f"deviance = {dev:.6f}  "
          f"（全體中位數 {np.median(err):.6f}，此 patch 排在第 {pct:.1f} 百分位）")
    print(f"NLL = {nll:.6f}（省略 log(y!) 常數項，可能為負）")

    print("\n逐類別 count 對帳：")
    for c, name in enumerate(CAT_ZH):
        print(f"  {name:<6}輸入 {cnt[c]:6.1f} 個 -> 重建 λ {lam[c]:6.2f}")

    fig, ax = plt.subplots(figsize=(10, 5.5))
    idx = np.arange(N_CAT)
    w = 0.38
    ax.bar(idx - w / 2, cnt, width=w, color=CAT_COLORS, alpha=0.55,
           label="真實 count")
    ax.bar(idx + w / 2, lam, width=w, color=CAT_COLORS, alpha=0.95,
           hatch="//", edgecolor="white", label="重建 λ")
    ax.set_xticks(idx)
    ax.set_xticklabels(CAT_ZH, fontsize=9, rotation=20, ha="right")
    ax.set_ylabel("個數", fontsize=9)
    ax.legend(fontsize=9, framealpha=0.9)
    ax.grid(alpha=0.15, linewidth=0.5, axis="y")
    for s in ax.spines.values():
        s.set_alpha(0.3)

    fig.suptitle(f"{VERSION} patch {n}（latent_dim={LATENT_DIM}，"
                 f"POI {n_poi} 個，deviance {dev:.4f}）", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT, bbox_inches="tight")
    print(f"\n已存 {OUT}")


main()
