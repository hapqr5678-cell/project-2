"""挑一個 patch 丟進訓練好的 v0 AE，比較「進去」與「出來」的 POI 分布。

用 --n 指定 patch 編號（0 ~ 23699）。
左圖 = AE 前的真實 POI 點圖（半徑 300m 的圓，顏色 = 類別）。
右圖 = AE 後的重建：重建出來的是每格每類的強度（連續值），
不是離散的點，所以把 (格子, 類別) 依強度排序，取前 N 名畫成點
（N = 這個 patch 真實的 POI 數），點大小/深淺 ∝ 強度，
等於「AE 認為最可能有 POI 的 N 個位置」。
另外印出這個 patch 的 MSE loss、它在全體 patch 裡的百分位，以及逐類別的 MSE。
"""

import argparse
import os
import sys

import numpy as np
import torch
import matplotlib as mpl
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ae import GRID, CELL, RADIUS, ConvAE, Patches, mse_loss  # noqa: E402

PATCHES = "modelOverture/patches.npz"
LATENTS = "modelOverture/v0/result/latents.npz"
CKPT = "modelOverture/v0/result/ae.pt"
OUT = "modelOverture/v0/result/rebuild_test100.png"

LATENT_DIM = 2
DOT = 18          # 點的基本大小

CAT_ZH = [
    "餐飲", "商業服務", "生活服務", "教育", "醫療",
    "購物", "藝文娛樂", "文化歷史", "交通旅遊", "運動休閒",
    "住宿", "社區政府", "地理實體",
]
CAT_COLORS = [
    "#e6194b", "#3cb44b", "#ffe119", "#4363d8", "#f58231",
    "#911eb4", "#46f0f0", "#f032e6", "#bcf60c", "#fabebe",
    "#008080", "#e6beff", "#9a6324",
]
IN_COLOR = "#2c7fb8"
OUT_COLOR = "#c0392b"

mpl.rcParams["font.family"] = "sans-serif"
mpl.rcParams["font.sans-serif"] = ["Microsoft JhengHei", "Heiti TC", "Arial"]
mpl.rcParams["axes.unicode_minus"] = False
mpl.rcParams["figure.dpi"] = 130


def style(ax, title, edge):
    ax.add_patch(plt.Circle((0, 0), RADIUS, fill=False, lw=1.6,
                            color=edge, alpha=0.8))
    ax.set_xlim(-RADIUS * 1.15, RADIUS * 1.15)
    ax.set_ylim(-RADIUS * 1.15, RADIUS * 1.15)
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=10, color=edge)
    ax.set_xlabel("東西向位移 (m)", fontsize=8)
    ax.set_ylabel("南北向位移 (m)", fontsize=8)
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


def draw_recon(ax, recon, n_top, title):
    """AE 後：取強度前 n_top 名的 (格子, 類別) 當成重建出來的點。"""
    inten = torch.expm1(recon.clamp(min=0))[0].numpy()   # (10,40,40) 還原成 count
    k, iy, ix = np.unravel_index(
        np.argsort(inten, axis=None)[::-1][:n_top], inten.shape)
    v = inten[k, iy, ix]
    rel = v / v.max()

    # 格子中心換算回公尺
    x = (ix + 0.5 - GRID / 2) * CELL
    y = (iy + 0.5 - GRID / 2) * CELL
    for c in range(len(CAT_ZH)):
        m = k == c
        if m.any():
            ax.scatter(x[m], y[m], s=DOT * (0.3 + 1.4 * rel[m]),
                       c=CAT_COLORS[c], linewidths=0,
                       alpha=0.25 + 0.6 * rel[m], label=CAT_ZH[c])
    style(ax, title, OUT_COLOR)
    return v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=0, help="patch 編號（0 起算）")
    n = ap.parse_args().n

    data = Patches(PATCHES)
    assert 0 <= n < data.n, f"--n 要在 0~{data.n - 1}"

    model = ConvAE(LATENT_DIM)
    model.load_state_dict(torch.load(CKPT, map_location="cpu"))
    model.eval()

    # 跟 train.py 的推論一致：固定朝向、不旋轉
    x = data.render(torch.tensor([n]), rotate=False)
    with torch.no_grad():
        z, recon = model(x)
    loss = mse_loss(recon, x).item()

    err = np.load(LATENTS)["err"]
    pct = (err < loss).mean() * 100
    n_poi = int(data.n_poi[n])

    print(f"patch {n}：POI {n_poi}  ({data.lat[n]:.5f}, {data.lon[n]:.5f})")
    print(f"latent z = ({z[0, 0]:+.3f}, {z[0, 1]:+.3f})")
    print(f"MSE loss = {loss:.6f}  "
          f"（全體中位數 {np.median(err):.6f}，此 patch 排在第 {pct:.1f} 百分位）")

    per_cat = ((recon - x) ** 2).mean(dim=(2, 3))[0]
    print("\n逐類別 MSE：")
    for c, name in enumerate(CAT_ZH):
        print(f"  {name:<6}{per_cat[c]:.6f}   "
              f"輸入總量 {x[0, c].sum():7.2f} -> 重建 {recon[0, c].sum():7.2f}")

    fig, (a, b) = plt.subplots(1, 2, figsize=(13, 6.5))
    draw_true(a, np.load(PATCHES), n, f"AE 前（真實 POI，共 {n_poi} 個）")
    v = draw_recon(b, recon, n_poi,
                   f"AE 後（重建強度前 {n_poi} 名，MSE {loss:.6f}）")
    a.legend(fontsize=6.5, markerscale=1.4, framealpha=0.9,
             loc="upper left", bbox_to_anchor=(1.01, 1.0))

    print(f"\n重建強度前 {n_poi} 名：最大 {v.max():.3f}、最小 {v.min():.3f} "
          f"（真實每格至少是 1 個 POI）")

    fig.suptitle(f"v0 重建測試：patch {n}（latent_dim={LATENT_DIM}）",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT, bbox_inches="tight")
    print(f"已存 {OUT}")


main()
