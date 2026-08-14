"""挑一個 patch 丟進訓練好的 v0_l32_poisson_nll AE，比較「進去」與「出來」的 POI 分布。

用 --n 指定 patch 編號（0 起算）。

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

import argparse
import os
import sys

import numpy as np
import torch
import matplotlib as mpl
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(f"{os.path.dirname(__file__)}/../../.."))
from ae import (GRID, CELL, HALF_WIDTH, ConvAE, Patches,  # noqa: E402
                poisson_deviance, poisson_nll)
from config.result_style import REBUILD_GRID  # noqa: E402
from config.dataset import CAT_COLORS, CAT_ZH, PATCHES, result  # noqa: E402

LATENTS = result("v0_l32_poisson_nll", "latents.npz")
CKPT = result("v0_l32_poisson_nll", "ae.pt")
OUT = result("v0_l32_poisson_nll", "rebuild_test.png")


DOT = 18          # 點的基本大小
IN_COLOR = "#2c7fb8"
OUT_COLOR = "#c0392b"

mpl.rcParams["font.family"] = ["Heiti TC"]
mpl.rcParams["axes.unicode_minus"] = False
mpl.rcParams["figure.dpi"] = 130


def style(ax, title, edge):
    ax.add_patch(plt.Circle((0, 0), HALF_WIDTH, fill=False, lw=1.6,
                            color=edge, alpha=0.8))
    ax.set_xlim(-HALF_WIDTH * 1.05, HALF_WIDTH * 1.05)
    ax.set_ylim(-HALF_WIDTH * 1.05, HALF_WIDTH * 1.05)
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=10, color=edge)
    ax.tick_params(labelsize=7)
    ax.grid(alpha=0.15, linewidth=0.5)
    if REBUILD_GRID:
        import numpy as _np
        ticks = _np.arange(-GRID // 2, GRID // 2 + 1) * CELL
        for t in ticks:
            ax.axhline(t, color='#888', lw=0.3, alpha=0.35)
            ax.axvline(t, color='#888', lw=0.3, alpha=0.35)
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


def _spiral_offsets(n, spacing=CELL * 0.18):
    """回傳 n 個從中心往外的螺旋位移 (dx, dy)。"""
    if n == 0:
        return np.empty((0, 2))
    pts = np.zeros((n, 2))
    for i in range(1, n):
        r = spacing * np.sqrt(i)
        theta = 2.4 * i          # 黃金角 ≈ 137.5°
        pts[i] = [r * np.cos(theta), r * np.sin(theta)]
    return pts


def draw_recon(ax, lam, title):
    """AE 後：把圓內每格每類的 λ 四捨五入成整數個 POI，
    從格子中心往外螺旋排列，每個 POI 畫一個點。"""
    inten = lam[0].numpy()          # (10,GRID,GRID)
    total = inten.sum(0)

    for c in range(len(CAT_ZH)):
        xs, ys = [], []
        for gy in range(GRID):
            for gx in range(GRID):
                cnt = int(np.round(inten[c, gy, gx]))
                if cnt <= 0:
                    continue
                cx = (gx + 0.5 - GRID / 2) * CELL
                cy = (gy + 0.5 - GRID / 2) * CELL
                offsets = _spiral_offsets(cnt)
                xs.append(cx + offsets[:, 0])
                ys.append(cy + offsets[:, 1])
        if xs:
            xs = np.concatenate(xs)
            ys = np.concatenate(ys)
            ax.scatter(xs, ys, s=DOT, c=CAT_COLORS[c], linewidths=0,
                       alpha=0.85, label=CAT_ZH[c])
    style(ax, title, OUT_COLOR)
    return total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=0, help="patch 編號（0 起算）")
    n = ap.parse_args().n

    data = Patches(PATCHES)
    assert 0 <= n < data.n, f"--n 要在 0~{data.n - 1}"

    sd = torch.load(CKPT, map_location="cpu")
    LATENT_DIM = sd["encoder.6.weight"].shape[0]
    model = ConvAE(LATENT_DIM)
    model.load_state_dict(sd)
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
    cell = 2 * (torch.xlogy(x, x) - x * log_lam - x + lam)
    per_cat = cell[0].sum(dim=(1, 2)) / (GRID * GRID)
    cnt_x = x[0].sum(dim=(1, 2))
    cnt_r = lam[0].sum(dim=(1, 2))
    print("\n逐類別 deviance（平均）與 POI 數對帳：")
    for c, name in enumerate(CAT_ZH):
        print(f"  {name:<6}deviance {per_cat[c]:.6f}   "
              f"輸入 {cnt_x[c]:6.1f} 個 -> 重建 λ 總和 {cnt_r[c]:6.1f}")

    fig, (a, b) = plt.subplots(1, 2, figsize=(13, 6.5))
    draw_true(a, np.load(PATCHES), n, f"AE 前（真實 POI，共 {n_poi} 個）")
    total = draw_recon(b, lam, f"AE 後（deviance {dev:.6f}）")
    a.legend(fontsize=6.5, markerscale=1.4, framealpha=0.9,
             loc="upper left", bbox_to_anchor=(1.01, 1.0))

    print(f"\n重建 λ：每格最大 {total.max():.3f}、"
          f"總和 {total.sum():.1f}（真實共 {cnt_x.sum():.0f} 個 POI）")

    fig.suptitle(f"patch {n}（latent_dim={LATENT_DIM}，raw count + Poisson NLL）",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT, bbox_inches="tight")
    print(f"已存 {OUT}")


main()
