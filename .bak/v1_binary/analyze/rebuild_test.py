"""挑一個 patch 丟進訓練好的 v1_binary AE，比較「進去」與「出來」的 POI 分布。

用 --n 指定 patch 編號（0 ~ 23699）。
左圖 = AE 前的真實 POI 點圖（半徑 300m 的圓，顏色 = 類別）。
右圖 = AE 後的重建：decoder 輸出的是 logits，sigmoid 後得到每格每類的
「有 POI 的機率」（0~1）。把機率 > 0.5 的格子畫成點，
點的深淺 ∝ 機率，等於「AE 認為最可能有 POI 的位置」。
另外印出 BCE loss、它在全體 patch 裡的百分位，以及逐類別的 BCE。
"""

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib as mpl
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(f"{os.path.dirname(__file__)}/../../.."))
from ae import GRID, CELL, HALF_WIDTH, ConvAE, Patches, bce_loss  # noqa: E402
from config.result_style import REBUILD_GRID  # noqa: E402
from config.dataset import CAT_COLORS, CAT_ZH, PATCHES, result  # noqa: E402

LATENTS = result("v1_binary", "latents.npz")
CKPT = result("v1_binary", "ae.pt")
OUT = result("v1_binary", "rebuild_test.png")

DOT = 18
IN_COLOR = "#2c7fb8"
OUT_COLOR = "#c0392b"
PROB_THRESH = 0.1   # 低於 0.5 是因為 class imbalance：模型傾向保守預測低機率
                    # 若訓練有加 pos_weight，可再調回 0.5


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
        # 格線畫在格子「邊界」(boundary)，不是格子中心
        # 邊界位置 = (k - GRID/2) * CELL，k = 0..GRID
        boundaries = (np.arange(GRID + 1) - GRID / 2) * CELL
        for t in boundaries:
            ax.axhline(t, color='#888', lw=0.3, alpha=0.35)
            ax.axvline(t, color='#888', lw=0.3, alpha=0.35)
    for s in ax.spines.values():
        s.set_alpha(0.3)


def draw_true(ax, p, i, title):
    """AE 前：真實 POI 一點一個；所有類別都加入 legend。"""
    s, e = p["offsets"][i], p["offsets"][i + 1]
    dx, dy, cat = p["dx"][s:e], p["dy"][s:e], p["cat"][s:e].astype(np.int64)
    for k in range(len(CAT_ZH)):
        m = cat == k
        if m.any():
            ax.scatter(dx[m], dy[m], s=DOT, c=CAT_COLORS[k], linewidths=0,
                       alpha=0.85, label=CAT_ZH[k])
        else:
            # 沒有點也加 legend handle，讓所有類別都顯示
            ax.scatter([], [], s=DOT, c=CAT_COLORS[k], linewidths=0,
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


def draw_recon(ax, logits, title):
    """AE 後：每格顯示機率最大的類別（argmax），顏色代表類別，大小 ∝ 最大機率。

    只有 max prob >= PROB_THRESH 的格子才畫點。
    一格一個點，跟左圖的真實 POI 對比最直覺。
    """
    prob = torch.sigmoid(logits)[0].numpy()   # (N_CAT, GRID, GRID)

    # 每格的 argmax 類別和最大機率
    best_cat = prob.argmax(axis=0)   # (GRID, GRID)
    best_prob = prob.max(axis=0)     # (GRID, GRID)

    cat_xs = [[] for _ in range(len(CAT_ZH))]
    cat_ys = [[] for _ in range(len(CAT_ZH))]
    cat_sz = [[] for _ in range(len(CAT_ZH))]

    for gy in range(GRID):
        for gx in range(GRID):
            p = float(best_prob[gy, gx])
            if p < PROB_THRESH:
                continue
            cx = (gx + 0.5 - GRID / 2) * CELL
            cy = (gy + 0.5 - GRID / 2) * CELL
            c = int(best_cat[gy, gx])
            cat_xs[c].append(cx)
            cat_ys[c].append(cy)
            cat_sz[c].append(p)

    for c in range(len(CAT_ZH)):
        if cat_xs[c]:
            ax.scatter(cat_xs[c], cat_ys[c],
                       s=DOT * np.array(cat_sz[c]) * 2,
                       c=CAT_COLORS[c], linewidths=0,
                       alpha=0.85, label=CAT_ZH[c])
        else:
            ax.scatter([], [], s=DOT, c=CAT_COLORS[c], linewidths=0,
                       alpha=0.85, label=CAT_ZH[c])
    style(ax, title, OUT_COLOR)
    return prob




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

    x = data.render(torch.tensor([n]), rotate=False)
    with torch.no_grad():
        z, logits = model(x)
    loss = bce_loss(logits, x).item()

    err = np.load(LATENTS)["err"]
    pct = (err < loss).mean() * 100
    n_poi = int(data.n_poi[n])

    print(f"patch {n}：POI {n_poi}  ({data.lat[n]:.5f}, {data.lon[n]:.5f})")
    print("latent z = " + ", ".join(f"{v:+.3f}" for v in z[0].tolist()))
    print(f"BCE loss = {loss:.6f}  "
          f"（全體中位數 {np.median(err):.6f}，此 patch 排在第 {pct:.1f} 百分位）")

    per_cat = F.binary_cross_entropy_with_logits(
        logits[0], x[0], reduction='none').mean(dim=(1, 2))
    prob_t = torch.sigmoid(logits[0])
    n_pred = (prob_t > PROB_THRESH).sum(dim=(1, 2))
    n_true = x[0].sum(dim=(1, 2))

    print(f"\n逐類別 BCE（threshold={PROB_THRESH}）：")
    for c, name in enumerate(CAT_ZH):
        print(f"  {name:<6}BCE {per_cat[c]:.6f}   "
              f"真實 {int(n_true[c]):3d} 格 -> 預測 {int(n_pred[c]):3d} 格")

    fig, (a, b) = plt.subplots(1, 2, figsize=(13, 6.5))
    draw_true(a, np.load(PATCHES), n, f"AE 前（真實 POI，共 {n_poi} 個）")
    draw_recon(b, logits, f"AE 後 BCE {loss:.6f}（機率>{PROB_THRESH} 的格子）")
    a.legend(fontsize=6.5, markerscale=1.4, framealpha=0.9,
             loc="upper left", bbox_to_anchor=(1.01, 1.0))

    fig.suptitle(f"patch {n}（latent_dim={LATENT_DIM}，v1_binary）",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT, bbox_inches="tight")
    print(f"已存 {OUT}")


main()
