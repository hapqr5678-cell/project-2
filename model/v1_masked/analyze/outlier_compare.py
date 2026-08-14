"""比較 v1_masked latent space 上的極端離群 patch 與中心（基準）patch。

取 robust 距離最大的 100 個當離群、最小的 100 個當基準，
RANK 指定要看第幾名的離群 patch（1 = 最極端）。

畫四張圖：
  上排：離群 / 基準 patch 的 POI 原始分布地圖（半徑 300m 的圓）
  下排左：單一 patch 的類別組成比較
  下排右：離群 patch vs 基準前 20 個平均的類別組成比較
"""

import os
import sys

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(f"{os.path.dirname(__file__)}/../../.."))
from config.dataset import (CAT_COLORS, CAT_ZH, CATEGORIES,  # noqa: E402
                            PATCHES, HALF_WIDTH, result)

LATENTS = result("v1_masked", "latents.npz")
OUT = result("v1_masked", "outlier_compare.png")

RANK = 1           # 要看第幾名的離群 patch（1~TOP_K，1 = 最極端）
TOP_K = 100        # 各取幾個離群 / 基準 patch
BASE_AVG = 20      # 右下角基準平均用幾個 patch

OUT_COLOR = "#c0392b"
BASE_COLOR = "#2c7fb8"

mpl.rcParams["font.family"] = ["Heiti TC"]
mpl.rcParams["axes.unicode_minus"] = False
mpl.rcParams["figure.dpi"] = 130


def robust_distance(z):
    """對每個軸用 中位數 / MAD 標準化後取歐氏距離，不被離群值本身帶偏。"""
    med = np.median(z, axis=0)
    mad = np.median(np.abs(z - med), axis=0) * 1.4826
    return np.linalg.norm((z - med) / mad, axis=1)


def patch_points(p, i):
    """回傳第 i 個 patch 的 (dx, dy, cat)。"""
    s, e = p["offsets"][i], p["offsets"][i + 1]
    return p["dx"][s:e], p["dy"][s:e], p["cat"][s:e]


def composition(p, i):
    """回傳第 i 個 patch 的 10 類佔比(%)。"""
    _, _, cat = patch_points(p, i)
    c = np.bincount(cat.astype(np.int64), minlength=len(CATEGORIES))
    return c / c.sum() * 100


def draw_map(ax, p, i, title, edge):
    dx, dy, cat = patch_points(p, i)
    ax.add_patch(plt.Circle((0, 0), HALF_WIDTH, fill=False, lw=1.6,
                            color=edge, alpha=0.8))
    for k in range(len(CATEGORIES)):
        m = cat == k
        if m.any():
            ax.scatter(dx[m], dy[m], s=16, c=CAT_COLORS[k], linewidths=0,
                       alpha=0.85, label=CAT_ZH[k])
    ax.set_xlim(-HALF_WIDTH * 1.1, HALF_WIDTH * 1.1)
    ax.set_ylim(-HALF_WIDTH * 1.1, HALF_WIDTH * 1.1)
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=10, color=edge)
    ax.set_xlabel("東西向位移 (m)", fontsize=8)
    ax.set_ylabel("南北向位移 (m)", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.grid(alpha=0.15, linewidth=0.5)
    for s in ax.spines.values():
        s.set_alpha(0.3)


def draw_bars(ax, left, right, left_label, right_label, title, right_err=None):
    y = np.arange(len(CATEGORIES))
    h = 0.38
    ax.barh(y - h / 2, left, h, color=OUT_COLOR, alpha=0.85, label=left_label)
    ax.barh(y + h / 2, right, h, color=BASE_COLOR, alpha=0.85,
            label=right_label,
            xerr=right_err, error_kw=dict(ecolor="#555", elinewidth=0.8,
                                          capsize=2))
    ax.set_yticks(y)
    ax.set_yticklabels(CAT_ZH, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("佔該 patch POI 的比例 (%)", fontsize=8)
    ax.set_title(title, fontsize=10)
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=8, framealpha=0.9)
    ax.grid(axis="x", alpha=0.15, linewidth=0.5)
    for s in ax.spines.values():
        s.set_alpha(0.3)


def main():
    n = RANK
    assert 1 <= n <= TOP_K, f"RANK 要在 1~{TOP_K}"

    d = np.load(LATENTS)
    p = np.load(PATCHES)
    z, n_poi, lat, lon = d["z"], d["n_poi"], d["lat"], d["lon"]
    occ = d["n_occupied"]

    dist = robust_distance(z)
    order = np.argsort(dist)
    base_idx = order[:TOP_K]          # 最靠 latent 中心的 100 個
    out_idx = order[::-1][:TOP_K]     # 最極端的 100 個

    oi = int(out_idx[n - 1])
    bi = int(base_idx[n - 1])

    print(f"離群 #{n}: patch {oi}  robust 距離 {dist[oi]:.2f}  "
          f"POI {n_poi[oi]}（佔 {occ[oi]} 格）  ({lat[oi]:.5f}, {lon[oi]:.5f})")
    print(f"基準 #{n}: patch {bi}  robust 距離 {dist[bi]:.2f}  "
          f"POI {n_poi[bi]}（佔 {occ[bi]} 格）  ({lat[bi]:.5f}, {lon[bi]:.5f})")
    print(f"離群 100 個佔用格數中位數 {np.median(occ[out_idx]):.0f}，"
          f"基準 100 個 {np.median(occ[base_idx]):.0f}，"
          f"全體 {np.median(occ):.0f}")

    comp_o = composition(p, oi)
    comp_b = composition(p, bi)
    avg = np.array([composition(p, int(i)) for i in base_idx[:BASE_AVG]])

    fig, ((a, b), (c, e)) = plt.subplots(2, 2, figsize=(13, 12))

    draw_map(a, p, oi, f"離群 #{n}（patch {oi}，POI {n_poi[oi]}）",
             OUT_COLOR)
    draw_map(b, p, bi, f"基準 #{n}（patch {bi}，POI {n_poi[bi]}）",
             BASE_COLOR)
    a.legend(fontsize=6.5, markerscale=1.2, framealpha=0.9,
             loc="upper left", bbox_to_anchor=(1.01, 1.0))

    draw_bars(c, comp_o, comp_b, f"離群 #{n}", f"基準 #{n}",
              "單一 patch 類別組成比較")
    draw_bars(e, comp_o, avg.mean(0), f"離群 #{n}",
              f"基準前 {BASE_AVG} 個平均",
              f"離群 vs 基準 {BASE_AVG} 個平均的類別組成",
              right_err=avg.std(0))

    fig.suptitle(f"v1_masked latent space：離群 #{n} vs 基準 patch"
                 f"（robust 距離 top/bottom {TOP_K}）", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT, bbox_inches="tight")
    print(f"\n已存 {OUT}")


main()
