"""挑一個 patch 丟進訓練好的 v0_poisson_nll AE，比較「進去」與「出來」的 POI 分布。

改 PATCH_N 換 patch（0 ~ 23699）。舊版本的 rebuild_test.py 用 --n，
新版依慣例改成檔案上方的常數。

左圖 = AE 前的真實 POI 點圖（半徑 300m 的圓，顏色 = 類別）。
右圖 = AE 後的重建：這一版 decoder 輸出的是 log λ，exp 之後就直接是
「該格該類的 POI 期望個數」，不必像 v0 那樣 expm1 反推——這是 Poisson 版
最好用的一點，重建強度本身有單位，可以直接跟真實 POI 數對帳
（見最後印的「重建 λ 總和 vs 真實 POI 數」）。

圓外的格子不畫也不列入統計：loss 有圓形遮罩，那裡的 λ 完全沒被約束，
畫出來只會看到模型的自由發揮，不是重建結果。

另外印出這個 patch 的 deviance / NLL、它在全體裡的百分位，以及逐類別的
deviance 與 λ 對帳。
"""

import os
import sys

import numpy as np
import torch
import matplotlib as mpl
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(f"{os.path.dirname(__file__)}/../../.."))
from ae import (GRID, CELL, MASK, RADIUS, ConvAE, Patches,  # noqa: E402
                poisson_deviance, poisson_nll)
from config.dataset import CAT_COLORS, CAT_ZH, PATCHES, result  # noqa: E402

LATENTS = result("v0_poisson_nll", "latents.npz")
CKPT = result("v0_poisson_nll", "ae.pt")
OUT = result("v0_poisson_nll", "rebuild_test.png")

PATCH_N = 0       # 要看哪個 patch
LATENT_DIM = 2
DOT = 18          # 點的基本大小
IN_COLOR = "#2c7fb8"
OUT_COLOR = "#c0392b"

mpl.rcParams["font.family"] = ["Heiti TC"]
mpl.rcParams["axes.unicode_minus"] = False
mpl.rcParams["figure.dpi"] = 130


def style(ax, title, edge):
    ax.add_patch(plt.Circle((0, 0), RADIUS, fill=False, lw=1.6,
                            color=edge, alpha=0.8))
    ax.set_xlim(-RADIUS * 1.05, RADIUS * 1.05)
    ax.set_ylim(-RADIUS * 1.05, RADIUS * 1.05)
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=10, color=edge)
    ax.tick_params(labelsize=7)
    ax.grid(alpha=0.15, linewidth=0.5)
    for s in ax.spines.values():
        s.set_alpha(0.3)


def draw_true(ax, p, i, title):
    """AE 前：真實 POI 一點一個。"""
    s, e = p["offsets"][i], p["offsets"][i + 1]
    dx, dy, cat = p["dx"][s:e], p["dy"][s:e], p["cat"][s:e].astype(np.int64)
    for k in range(len(CAT_ZH)):
        m = cat == k
        if m.any():
            ax.scatter(dx[m], dy[m], s=DOT, c=CAT_COLORS[k], linewidths=0,
                       alpha=0.85, label=CAT_ZH[k])
    style(ax, title, IN_COLOR)


def draw_recon(ax, lam, title):
    """AE 後：圓內每一格畫一點，顏色 = 該格 λ 最大的類別，
    大小/透明度 ∝ 該格的總 λ（λ 趨近 0 的格子自然看不見）。"""
    inten = lam[0].numpy()          # (10,40,40)，單位是「期望 POI 個數」
    total = inten.sum(0)
    top_cat = inten.argmax(0)
    inside = MASK[0, 0].numpy()
    rel = total / total[inside].max()

    # 格子中心換算回公尺
    gy, gx = np.mgrid[0:GRID, 0:GRID]
    x = (gx + 0.5 - GRID / 2) * CELL
    y = (gy + 0.5 - GRID / 2) * CELL
    for c in range(len(CAT_ZH)):
        m = (top_cat == c) & inside
        if m.any():
            ax.scatter(x[m], y[m], s=DOT * (0.15 + 1.5 * rel[m]),
                       c=CAT_COLORS[c], linewidths=0,
                       alpha=np.clip(rel[m], 0, 1) * 0.9, label=CAT_ZH[c])
    style(ax, title, OUT_COLOR)
    return total, inside


def main():
    data = Patches(PATCHES)
    n = PATCH_N
    assert 0 <= n < data.n, f"PATCH_N 要在 0~{data.n - 1}"

    model = ConvAE(LATENT_DIM)
    model.load_state_dict(torch.load(CKPT, map_location="cpu"))
    model.eval()

    # 跟 train.py 的推論一致：固定朝向、不旋轉
    x = data.render(torch.tensor([n]), rotate=False)
    with torch.no_grad():
        z, log_lam = model(x)
    dev = poisson_deviance(log_lam, x).item()
    nll = poisson_nll(log_lam, x).item()
    lam = torch.exp(log_lam)

    err = np.load(LATENTS)["err"]
    pct = (err < dev).mean() * 100
    n_poi = int(data.n_poi[n])

    print(f"patch {n}：POI {n_poi}  ({data.lat[n]:.5f}, {data.lon[n]:.5f})")
    print(f"latent z = ({z[0, 0]:+.3f}, {z[0, 1]:+.3f})")
    print(f"deviance = {dev:.6f}  "
          f"（全體中位數 {np.median(err):.6f}，此 patch 排在第 {pct:.1f} 百分位）")
    print(f"NLL = {nll:.6f}（省略 log(y!) 常數項，可能為負）")

    # 逐類別：deviance 只算圓內；λ 總和可以直接跟輸入的 count 總和對帳
    m = MASK.to(log_lam.dtype)
    cell = 2 * (torch.xlogy(x, x) - x * log_lam - x + lam) * m
    per_cat = cell[0].sum(dim=(1, 2)) / MASK.sum()
    cnt_x = (x * m)[0].sum(dim=(1, 2))
    cnt_r = (lam * m)[0].sum(dim=(1, 2))
    print("\n逐類別 deviance（圓內平均）與 POI 數對帳：")
    for c, name in enumerate(CAT_ZH):
        print(f"  {name:<6}deviance {per_cat[c]:.6f}   "
              f"輸入 {cnt_x[c]:6.1f} 個 -> 重建 λ 總和 {cnt_r[c]:6.1f}")

    fig, (a, b) = plt.subplots(1, 2, figsize=(13, 6.5))
    draw_true(a, np.load(PATCHES), n, f"AE 前（真實 POI，共 {n_poi} 個）")
    total, inside = draw_recon(b, lam, f"AE 後（deviance {dev:.6f}）")
    a.legend(fontsize=6.5, markerscale=1.4, framealpha=0.9,
             loc="upper left", bbox_to_anchor=(1.01, 1.0))

    print(f"\n圓內重建 λ：每格最大 {total[inside].max():.3f}、"
          f"總和 {total[inside].sum():.1f}（圓內真實共 {cnt_x.sum():.0f} 個 POI）")
    print(f"圓外（loss 沒約束）λ 總和 {total[~inside].sum():.1f}，"
          f"僅供參考，未畫在圖上")

    fig.suptitle(f"patch {n}（latent_dim={LATENT_DIM}，raw count + 圓內 Poisson NLL）",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT, bbox_inches="tight")
    print(f"已存 {OUT}")


main()
