"""畫出 v2_perceiver 的 latent space，並檢查它到底編碼了什麼。

跟 v2_ae 對照著看：v2_ae 是「先聚合好再吃」的 MLP 基準組，這一版
encoder 直接吃原始 POI token 集合、自己用 cross-attention 聚合。
如果這裡的 kNN R² 明顯比 v2_ae 低，代表「模型自己學聚合」確實比
「先聚合好再吃」更能把 latent 從密度手上搶回來；如果差不多，
代表瓶頸不在 encoder 讀資料的方式。

「偏心度」這欄特別值得看：v2 系列的輸入根本沒有 x,y 座標（Perceiver
的 token 也只有類別，沒有位置），模型從沒看過哪個 POI 在圓的哪一側。
如果 latent 對偏心度的 kNN R² 仍然明顯大於 0，那不是模型記得座標，
而是「類別組成剛好跟偏心度相關」的巧合，這件事本身也值得在報告裡說清楚。
"""

import os
import sys
import argparse

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.abspath(f"{os.path.dirname(__file__)}/../../.."))
from config.dataset import CAT_ZH, PATCHES, HALF_WIDTH, result  # noqa: E402

VERSION = "v2_perceiver"
LATENTS = result(VERSION, "latents.npz")
OUT = result(VERSION, "latent_plot.png")

ZOOM_PCT = (1, 99)      # 放大視角的座標範圍百分位
OUTLIER_PCT = 99.5      # robust 距離超過這個百分位的點算離群
KNN = 50
DOT = 3.0

mpl.rcParams["font.family"] = ["Heiti TC"]
mpl.rcParams["axes.unicode_minus"] = False
mpl.rcParams["figure.dpi"] = 130


def eccentricity(p):
    """|POI 質心| / R：0 = 圍繞中心，大 = POI 全擠在圓的一側。"""
    offs, dx, dy = p["offsets"], p["dx"], p["dy"]
    v = np.zeros(len(offs) - 1)
    for i in range(len(v)):
        s, e = offs[i], offs[i + 1]
        v[i] = np.hypot(dx[s:e].mean(), dy[s:e].mean()) / HALF_WIDTH
    return v


def knn_r2(z, y):
    """用 latent 的 k 個鄰居預測 y，回傳 R²（排除自己）。"""
    zs = (z - z.mean(0)) / z.std(0)
    _, idx = cKDTree(zs).query(zs, k=KNN + 1)
    pred = y[idx[:, 1:]].mean(1)
    return 1 - np.var(y - pred) / np.var(y)


def robust_distance(z):
    """對每個軸用 中位數 / MAD 標準化後取歐氏距離，不被離群值本身帶偏。"""
    med = np.median(z, axis=0)
    mad = np.median(np.abs(z - med), axis=0) * 1.4826
    return np.linalg.norm((z - med) / mad, axis=1)


def scatter(ax, z, c, title, label, cmap="viridis"):
    sc = ax.scatter(z[:, 0], z[:, 1], c=c, s=DOT, cmap=cmap, linewidths=0,
                    alpha=0.6, rasterized=True,
                    vmin=np.percentile(c, 1), vmax=np.percentile(c, 99))
    cb = ax.figure.colorbar(sc, ax=ax, fraction=0.046, pad=0.02)
    cb.set_label(label, fontsize=8)
    cb.ax.tick_params(labelsize=7)
    style(ax, title)


def style(ax, title, xlabel="z1", ylabel="z2"):
    ax.set_title(title, fontsize=10)
    ax.set_xlabel(xlabel, fontsize=8)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.tick_params(labelsize=7)
    ax.grid(alpha=0.15, linewidth=0.5)
    for s in ax.spines.values():
        s.set_alpha(0.3)


def zoom(ax, z):
    lo = np.percentile(z, ZOOM_PCT[0], axis=0)
    hi = np.percentile(z, ZOOM_PCT[1], axis=0)
    pad = (hi - lo) * 0.05
    ax.set_xlim(lo[0] - pad[0], hi[0] + pad[0])
    ax.set_ylim(lo[1] - pad[1], hi[1] + pad[1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--c", type=int, default=0, help="類別 ID (預設 0)")
    c_id = ap.parse_args().c

    p = np.load(PATCHES)
    owner = np.repeat(np.arange(len(p["n_poi"])), p["n_poi"])
    e_i = np.bincount(owner[p["cat"] == c_id], minlength=len(p["n_poi"]))
    e_t = p["n_poi"]
    with np.errstate(divide='ignore', invalid='ignore'):
        local_ratio = e_i / e_t
    global_ratio = e_i.sum() / e_t.sum()
    lq = np.nan_to_num(local_ratio / global_ratio)

    print(f"LQ_i ({CAT_ZH[c_id]}): max={lq.max():.2f}, mean={lq.mean():.2f}")

    d = np.load(LATENTS)
    z, n_poi, err = d["z"], d["n_poi"], d["err"]
    lat, lon = d["lat"], d["lon"]
    ecc = eccentricity(np.load(PATCHES))

    print(f"latent 能還原多少（kNN R², k={KNN}）")
    for name, y in [("log(POI 數) ← 密度", np.log(n_poi.astype(float))),
                    ("偏心度", ecc),
                    ("Poisson deviance", err.astype(float))]:
        print(f"  {name:<22}R² = {knn_r2(z, y):+.3f}")

    dist = robust_distance(z)
    out = dist > np.percentile(dist, OUTLIER_PCT)

    zp, axes_labels = z, ("z1", "z2")

    print()
    print(f"離群 {out.sum()} 個：POI 數中位數 {np.median(n_poi[out]):.0f} "
          f"(全體 {np.median(n_poi):.0f})，"
          f"偏心度 {np.median(ecc[out]):.3f} (全體 {np.median(ecc):.3f})")

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    (a, b, c), (e, f, g) = axes

    scatter(a, zp, np.log10(n_poi), "全域：色=POI 數", "log10(POI 數)")
    zoom(b, zp)
    scatter(b, zp, lq, f"放大 {ZOOM_PCT[0]}~{ZOOM_PCT[1]}%：LQ ({CAT_ZH[c_id]})", "LQ", cmap="magma")
    zoom(c, zp)
    scatter(c, zp, ecc, "放大：色=偏心度", "|質心|/R", cmap="magma")

    zoom(e, zp)
    scatter(e, zp, err, "放大：色=重建誤差", "Poisson deviance")

    f.scatter(zp[~out, 0], zp[~out, 1], s=DOT, c="#b8bcc4",
              linewidths=0, alpha=0.5, rasterized=True)
    f.scatter(zp[out, 0], zp[out, 1], s=DOT * 4, c="#c0392b",
              linewidths=0, alpha=0.8, rasterized=True,
              label=f"離群 (>{OUTLIER_PCT}%，n={out.sum()})")
    f.legend(fontsize=8, markerscale=2, framealpha=0.9)
    style(f, "全域：robust 距離離群點", *axes_labels)

    g.scatter(lon[~out], lat[~out], s=1.5, c="#b8bcc4",
              linewidths=0, alpha=0.4, rasterized=True)
    g.scatter(lon[out], lat[out], s=8, c="#c0392b",
              linewidths=0, alpha=0.85, rasterized=True, label="離群 patch")
    g.legend(fontsize=8, markerscale=2, framealpha=0.9)
    g.set_aspect(1 / np.cos(np.deg2rad(float(lat.mean()))))
    g.set_title("離群 patch 的地理位置", fontsize=10)
    g.set_xlabel("經度", fontsize=8)
    g.set_ylabel("緯度", fontsize=8)
    g.tick_params(labelsize=7)
    g.grid(alpha=0.15, linewidth=0.5)

    fig.suptitle(f"{VERSION} latent space（{len(z)} 個 patch，latent_dim=2，"
                 f"原始 POI token + Poisson NLL，Perceiver Cross-Attention encoder）", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT, bbox_inches="tight")
    print(f"\n已存 {OUT}")


main()
