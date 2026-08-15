"""挑一個 patch 丟進訓練好的 v3_multinomial，比較「進去」與「出來」的類別 count。

v3 的重建跟 v2 有一個結構性差異要先講清楚，不然圖會看錯：
decoder 的輸出經過 log_softmax 再加 log n，所以 Σλ = n 是硬約束——
「總數對得上」是設計保證的，不是模型的本事。這張圖唯一有資訊的是
**這 n 個 POI 被分配到各類別的比例對不對**，也就是長條之間的相對高低，
不是長條的絕對高度。

要看的患處是「真實有、λ 卻壓到接近 0」的類別：那代表這個 patch 的
類別組合在全市找不到同類，2 維 latent 沒有位置放它。這種 patch 正是
專題要找的離群點，重建失敗在這裡是特徵不是缺陷。

用 --n 指定 patch 編號（0 起算）；不給就自動挑一個 deviance 接近中位數的
典型 patch。圖上的重建 λ 會 round 成整數（POI 個數本來就是整數，
小數點只會讓長條圖難讀），但印出來的對帳保留兩位小數。
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
from ae import MultinomialAE, N_CAT, Patches, poisson_deviance, poisson_nll  # noqa: E402
from config.dataset import CAT_COLORS, CAT_ZH, PATCHES, result  # noqa: E402

VERSION = "v3_multinomial"
LATENTS = result(VERSION, "latents.npz")
CKPT = result(VERSION, "ae.pt")
OUT = result(VERSION, "rebuild_test.png")

LATENT_DIM = 2

mpl.rcParams["font.family"] = ["Heiti TC"]
mpl.rcParams["axes.unicode_minus"] = False
mpl.rcParams["figure.dpi"] = 130


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=None,
                    help="patch 編號（0 起算）；不給就挑 deviance 接近中位數的典型 patch")
    arg_n = ap.parse_args().n

    data = Patches(PATCHES)

    model = MultinomialAE(LATENT_DIM)
    model.load_state_dict(torch.load(CKPT, map_location="cpu"))
    model.eval()

    err = np.load(LATENTS)["err"]
    n = arg_n if arg_n is not None else int(
        np.argmin(np.abs(err - np.median(err))))
    assert 0 <= n < data.n, f"--n 要在 0~{data.n - 1}"

    x = data.agg(torch.tensor([n]))
    with torch.no_grad():
        z, log_lam = model(x)
    dev = poisson_deviance(log_lam, x).item()
    nll = poisson_nll(log_lam, x).item()
    lam = torch.exp(log_lam)[0].numpy()
    cnt = x[0].numpy()

    pct = (err < dev).mean() * 100
    n_poi = int(data.n_poi[n])

    print(f"patch {n}：POI {n_poi}  ({data.lat[n]:.5f}, {data.lon[n]:.5f})")
    print(f"latent z = ({z[0, 0]:+.3f}, {z[0, 1]:+.3f})")
    print(f"deviance = {dev:.6f}  "
          f"（全體中位數 {np.median(err):.6f}，此 patch 排在第 {pct:.1f} 百分位）")
    print(f"NLL = {nll:.6f}（省略 log(y!) 常數項，可能為負）")
    print(f"Σλ = {lam.sum():.2f}，真實總數 {cnt.sum():.0f}"
          f"（差 {abs(lam.sum() - cnt.sum()):.2e}，log_softmax + log n 的硬約束）")

    print("\n逐類別對帳（比例才是重點，總數是設計保證的）：")
    for c, name in enumerate(CAT_ZH):
        flag = ""
        if cnt[c] >= 3 and lam[c] < cnt[c] * 0.4:
            flag = "  <- 真實有、λ 壓不上去：組成在全市罕見"
        print(f"  {name:<6}輸入 {cnt[c]:6.0f} 個 -> 重建 λ {lam[c]:6.2f}{flag}")

    fig, ax = plt.subplots(figsize=(11, 5.5))
    idx = np.arange(N_CAT)
    w = 0.38
    ax.bar(idx - w / 2, cnt, width=w, color=CAT_COLORS, alpha=0.55,
           label="真實 count")
    ax.bar(idx + w / 2, np.round(lam), width=w, color=CAT_COLORS, alpha=0.95,
           hatch="//", edgecolor="white", label="重建 λ（四捨五入）")
    ax.set_xticks(idx)
    ax.set_xticklabels(CAT_ZH, fontsize=9, rotation=20, ha="right")
    ax.set_ylabel("個數", fontsize=9)
    ax.legend(fontsize=9, framealpha=0.9)
    ax.grid(alpha=0.15, linewidth=0.5, axis="y")
    for s in ax.spines.values():
        s.set_alpha(0.3)

    fig.suptitle(f"{VERSION} patch {n}（latent_dim={LATENT_DIM}，"
                 f"POI {n_poi} 個，deviance {dev:.4f}，第 {pct:.1f} 百分位）\n"
                 f"Σλ = n 為硬約束，看的是類別之間的相對比例", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT, bbox_inches="tight")
    print(f"\n已存 {OUT}")


main()
