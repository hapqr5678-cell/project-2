"""挑一個 patch 丟進訓練好的 v0_l16 AE，比較「進去」與「出來」的 POI 分布。

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

PATCHES = "model/patches.npz"
LATENTS = "model/v0_l16/result/latents.npz"
CKPT = "model/v0_l16/result/ae.pt"
OUT = "model/v0_l16/result/rebuild_test.png"

LATENT_DIM = 16
DOT = 18          # 點的基本大小

CAT_ZH = [
    "餐飲", "零售", "夜生活", "政府", "交通",
    "商業服務", "地標", "藝文娛樂", "醫療", "運動休閒",
]
CAT_COLORS = [
    "#e6194b", "#3cb44b", "#911eb4", "#4363d8", "#f58231",
    "#46f0f0", "#008080", "#f032e6", "#9a6324", "#808000",
]
IN_COLOR = "#2c7fb8"
OUT_COLOR = "#c0392b"

mpl.rcParams["font.family"] = ["Heiti TC"]
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


def draw_recon(ax, recon, title):
    """AE 後：40x40 每一格都畫一點，顏色 = 該格最強的類別，
    大小/透明度 ∝ 該格的總強度（強度 0 的格子自然看不見）。"""
    inten = torch.expm1(recon.clamp(min=0))[0].numpy()   # (10,40,40) 還原成 count
    total = inten.sum(0)
    top_cat = inten.argmax(0)
    rel = total / total.max()

    # 格子中心換算回公尺
    gy, gx = np.mgrid[0:GRID, 0:GRID]
    x = (gx + 0.5 - GRID / 2) * CELL
    y = (gy + 0.5 - GRID / 2) * CELL
    for c in range(len(CAT_ZH)):
        m = top_cat == c
        if m.any():
            ax.scatter(x[m], y[m], s=DOT * (0.15 + 1.5 * rel[m]),
                       c=CAT_COLORS[c], linewidths=0,
                       alpha=np.clip(rel[m], 0, 1) * 0.9, label=CAT_ZH[c])
    style(ax, title, OUT_COLOR)
    return total


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
    print("latent z = " + ", ".join(f"{v:+.3f}" for v in z[0].tolist()))
    print(f"MSE loss = {loss:.6f}  "
          f"（全體中位數 {np.median(err):.6f}，此 patch 排在第 {pct:.1f} 百分位）")

    # MSE 是在 log1p 空間算的；POI 數要 expm1 還原回 count 才有意義
    per_cat = ((recon - x) ** 2).mean(dim=(2, 3))[0]
    cnt_x = torch.expm1(x.clamp(min=0))[0].sum(dim=(1, 2))
    cnt_r = torch.expm1(recon.clamp(min=0))[0].sum(dim=(1, 2))
    print("\n逐類別 MSE（log1p 空間）與 POI 數（expm1 還原）：")
    for c, name in enumerate(CAT_ZH):
        print(f"  {name:<6}MSE {per_cat[c]:.6f}   "
              f"輸入 {cnt_x[c]:6.1f} 個 -> 重建 {cnt_r[c]:6.1f} 個")

    fig, (a, b) = plt.subplots(1, 2, figsize=(13, 6.5))
    draw_true(a, np.load(PATCHES), n, f"AE 前（真實 POI，共 {n_poi} 個）")
    total = draw_recon(b, recon, f"AE 後（全部 40x40 格，MSE {loss:.6f}）")
    a.legend(fontsize=6.5, markerscale=1.4, framealpha=0.9,
             loc="upper left", bbox_to_anchor=(1.01, 1.0))

    print(f"\n重建每格總強度：最大 {total.max():.3f}、"
          f"總和 {total.sum():.1f}（真實共 {n_poi} 個 POI）")

    fig.suptitle(f"v0_l16 重建測試：patch {n}（latent_dim={LATENT_DIM}）",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT, bbox_inches="tight")
    print(f"已存 {OUT}")


main()
