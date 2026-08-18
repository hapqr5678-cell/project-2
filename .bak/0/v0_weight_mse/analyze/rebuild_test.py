"""挑一個 patch 丟進訓練好的 v0_weight_mse AE，比較「進去」與「出來」的 POI 分布。

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
sys.path.insert(0, os.path.abspath(f"{os.path.dirname(__file__)}/../../.."))
from ae import GRID, CELL, HALF_WIDTH, ConvAE, Patches, mse_loss  # noqa: E402
from config.result_style import REBUILD_GRID  # noqa: E402
from config.dataset import CAT_COLORS, CAT_ZH, PATCHES, result  # noqa: E402

LATENTS = result("v0_weight_mse", "latents.npz")
CKPT = result("v0_weight_mse", "ae.pt")
OUT = result("v0_weight_mse", "rebuild_test.png")

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


def draw_recon(ax, recon, title):
    """AE 後：把每格每類的預測強度四捨五入成整數個 POI，
    從格子中心往外螺旋排列，每個 POI 畫一個點，呈現方式與左圖一致。"""
    inten = torch.expm1(recon.clamp(min=0))[0].numpy()   # (10,GRID,GRID) 還原成 count
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
        z, recon = model(x)
    loss = mse_loss(recon, x).item()

    err = np.load(LATENTS)["err"]
    pct = (err < loss).mean() * 100
    n_poi = int(data.n_poi[n])

    print(f"patch {n}：POI {n_poi}  ({data.lat[n]:.5f}, {data.lon[n]:.5f})")
    print(f"latent z = ({z[0, 0]:+.3f}, {z[0, 1]:+.3f})")
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
    total = draw_recon(b, recon, f"AE 後MSE {loss:.6f}）")
    a.legend(fontsize=6.5, markerscale=1.4, framealpha=0.9,
             loc="upper left", bbox_to_anchor=(1.01, 1.0))

    print(f"\n重建每格總強度：最大 {total.max():.3f}、"
          f"總和 {total.sum():.1f}（真實共 {n_poi} 個 POI）")

    fig.suptitle(f"patch {n}（latent_dim={LATENT_DIM}）",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT, bbox_inches="tight")
    print(f"已存 {OUT}")


main()
