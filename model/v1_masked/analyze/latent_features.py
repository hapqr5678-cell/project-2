"""用手算特徵表替 v1_masked 的 latent 上色，看 latent 到底編碼了哪個維度。

特徵表跟 v0 共用（data/patch/build_features.py 產的那份），四個代表性特徵：
  log1p(n_total)  量
  entropy         組成的多樣性
  max_p           組成的單一化程度
  mean_r          空間分布（POI 平均離中心多遠）
"""

import os
import sys

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.abspath(f"{os.path.dirname(__file__)}/../../.."))
from config.dataset import FEATURES, result  # noqa: E402

LATENTS = result("v1_masked", "latents.npz")
OUT = result("v1_masked", "latent_features.png")

KNN = 50
DOT = 3.0

COLS = [
    ("log_n_total", "log1p(POI 總數)", "量", "viridis"),
    ("entropy", "類別 Shannon 熵 (bits)", "組成：多樣性", "cividis"),
    ("max_p", "最大類別佔比", "組成：單一化", "magma"),
    ("mean_r", "平均距中心距離 (m)", "空間", "plasma"),
]

mpl.rcParams["font.family"] = ["Heiti TC"]
mpl.rcParams["axes.unicode_minus"] = False
mpl.rcParams["figure.dpi"] = 130


def knn_r2(z, y):
    """用 latent 的 k 個鄰居預測 y，回傳 R²（排除自己）。"""
    zs = (z - z.mean(0)) / z.std(0)
    _, idx = cKDTree(zs).query(zs, k=KNN + 1)
    pred = y[idx[:, 1:]].mean(1)
    return 1 - np.var(y - pred) / np.var(y)


def style(ax, title):
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("z1", fontsize=8)
    ax.set_ylabel("z2", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.grid(alpha=0.15, linewidth=0.5)
    for s in ax.spines.values():
        s.set_alpha(0.3)


def scatter(ax, z, c, title, label, cmap):
    sc = ax.scatter(z[:, 0], z[:, 1], c=c, s=DOT, cmap=cmap, linewidths=0,
                    alpha=0.6, rasterized=True,
                    vmin=np.percentile(c, 1), vmax=np.percentile(c, 99))
    cb = ax.figure.colorbar(sc, ax=ax, fraction=0.046, pad=0.02)
    cb.set_label(label, fontsize=8)
    cb.ax.tick_params(labelsize=7)
    style(ax, title)


def main():
    z = np.load(LATENTS)["z"]
    f = np.load(FEATURES)

    print(f"latent 能還原多少（kNN R², k={KNN}）")
    for key, label, group, _ in COLS:
        print(f"  {group:<12}{key:<14}R² = {knn_r2(z, f[key].astype(float)):+.3f}")

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    for ax, (key, label, group, cmap) in zip(axes.ravel(), COLS):
        scatter(ax, z, f[key].astype(float), f"色={group}", label, cmap)

    fig.suptitle(f"v1_masked latent space 依手算特徵上色（{len(z)} 個 patch）",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT, bbox_inches="tight")
    print(f"\n已存 {OUT}")


main()
