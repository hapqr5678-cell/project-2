"""比較 v2_ae_tanh_fsce（latent 被 tanh 夾在 (-1,1)²）跟 v2_ae_fsce（latent
不受界限）兩個都加了 FSCE loss 的版本，latent space 到底有沒有分群。

先用 log(POI 數) 驗證過兩版都有結構（kNN R² 都不低），但這只回答了「latent
有沒有編碼密度」，回答不了「tanh 版那條沿 z1 的乾淨梯度，是密度、還是剛好
是某個類別的組成比例」——這裡多加兩個跟總量無關的「組成」訊號來拆解這件事：

  主導類別：每個 patch 10 類 POI 裡數量最多的那一類，離散上色。如果 latent
  的形狀其實是按主導類別分開，這裡會看到色塊而不是雜色。
  組成熵：把 count 正規化成比例後算 Shannon entropy，熵低 = 集中在少數
  類別、熵高 = 均勻分散在多類。這個量刻意跟總量（POI 數）正交，只看「形狀」
  不看「量」，用來檢查 latent 是不是把「量」跟「質」都塞進同一條軸，
  還是分開編碼在不同軸上。

兩個版本各自 zoom 到自己的 1~99 百分位範圍（不共用座標軸），因為這裡要比較
的是「形狀」，不是「絕對尺度」——tanh 版本來就被夾在 (-1,1)²，無 tanh 版的
座標範圍大了一個數量級，共用軸只會讓無 tanh 版縮成一個點。
"""

import os
import sys

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Patch as LegendPatch
from scipy.spatial import cKDTree

ROOT = os.path.abspath(f"{os.path.dirname(__file__)}/..")
sys.path.insert(0, ROOT)
from config.dataset import CAT_COLORS, CAT_ZH, N_CAT, PATCHES, result  # noqa: E402

VERSIONS = ["v2_ae_tanh_fsce", "v2_ae_fsce"]
LABELS = {"v2_ae_tanh_fsce": "有 tanh", "v2_ae_fsce": "無 tanh"}
OUT = os.path.join(ROOT, "lab", "tanh_vs_notanh_ae_fsce.png")

ZOOM_PCT = (1, 99)
KNN = 50
DOT = 4.0

mpl.rcParams["font.family"] = ["Heiti TC"]
mpl.rcParams["axes.unicode_minus"] = False
mpl.rcParams["figure.dpi"] = 130


def load_composition():
    """從 PATCHES 直接算每個 patch 的類別 count（跟各版本 ae.py 的
    Patches.agg() 是同一份邏輯，這裡用 numpy 重算一次，避免拉進 torch
    依賴）。回傳 dominant：每個 patch 數量最多的類別 id (N,)；entropy：
    正規化成比例後的 Shannon entropy (N,)，熵低代表集中在少數類別、
    熵高代表均勻分散在多類，刻意跟總量正交，只看組成的「形狀」。
    """
    p = np.load(PATCHES)
    cat, offsets = p["cat"], p["offsets"]
    n = len(offsets) - 1
    owner = np.repeat(np.arange(n), np.diff(offsets))
    flat = owner * N_CAT + cat
    counts = np.bincount(flat, minlength=n * N_CAT).reshape(n, N_CAT).astype(np.float64)
    dominant = counts.argmax(axis=1)
    props = counts / counts.sum(axis=1, keepdims=True)
    entropy = -(props * np.log(props + 1e-12)).sum(axis=1)
    return dominant, entropy


def knn_r2(z, y):
    """z 是 (N,2) 的 latent 座標，y 是 (N,) 要還原的目標值，用 z 標準化後的
    k 個最近鄰居的 y 平均值當預測，回傳 R²（越高代表 latent 距離跟 y 越一致）。
    """
    zs = (z - z.mean(0)) / z.std(0)
    _, idx = cKDTree(zs).query(zs, k=KNN + 1)
    pred = y[idx[:, 1:]].mean(1)
    return 1 - np.var(y - pred) / np.var(y)


def knn_categorical_accuracy(z, labels):
    """z 是 (N,2) 的 latent 座標，labels 是 (N,) 的類別 id（這裡傳主導類別）。
    用 z 標準化後的 k 個最近鄰居做多數決當預測，回傳準確率（越高代表 latent
    距離跟主導類別越一致；隨機基準是 1/類別數）。
    """
    zs = (z - z.mean(0)) / z.std(0)
    _, idx = cKDTree(zs).query(zs, k=KNN + 1)
    neighbor_labels = labels[idx[:, 1:]]
    n_labels = labels.max() + 1
    pred = np.array([np.bincount(row, minlength=n_labels).argmax()
                      for row in neighbor_labels])
    return (pred == labels).mean()


def zoom(ax, z):
    lo = np.percentile(z, ZOOM_PCT[0], axis=0)
    hi = np.percentile(z, ZOOM_PCT[1], axis=0)
    pad = (hi - lo) * 0.05
    ax.set_xlim(lo[0] - pad[0], hi[0] + pad[0])
    ax.set_ylim(lo[1] - pad[1], hi[1] + pad[1])


def style(ax, title):
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("z1", fontsize=8)
    ax.set_ylabel("z2", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.grid(alpha=0.15, linewidth=0.5)
    for s in ax.spines.values():
        s.set_alpha(0.3)


def scatter(ax, z, c, title, label):
    sc = ax.scatter(z[:, 0], z[:, 1], c=c, s=DOT, cmap="viridis", linewidths=0,
                    alpha=0.65, rasterized=True,
                    vmin=np.percentile(c, 1), vmax=np.percentile(c, 99))
    cb = ax.figure.colorbar(sc, ax=ax, fraction=0.046, pad=0.02)
    cb.set_label(label, fontsize=8)
    cb.ax.tick_params(labelsize=7)
    style(ax, title)
    zoom(ax, z)


def scatter_categorical(ax, z, labels, title):
    colors = [CAT_COLORS[i] for i in labels]
    ax.scatter(z[:, 0], z[:, 1], c=colors, s=DOT, linewidths=0,
              alpha=0.7, rasterized=True)
    style(ax, title)
    zoom(ax, z)


def main():
    dominant, entropy = load_composition()
    n_cat_present = len(np.unique(dominant))
    print(f"主導類別涵蓋 {n_cat_present}/{N_CAT} 類，隨機基準準確率 "
          f"≈{1 / N_CAT:.3f}（若類別不平均，實際基準會更高）\n")

    fig, axes = plt.subplots(3, 2, figsize=(11, 14.5))

    for col, v in enumerate(VERSIONS):
        d = np.load(result(v, "latents.npz"))
        z, n_poi = d["z"], d["n_poi"]
        log_poi = np.log(n_poi.astype(float))
        assert len(z) == len(dominant), "latents.npz 的 patch 數跟目前 PATCHES 對不上，先重新訓練"

        r2_poi = knn_r2(z, log_poi)
        r2_ent = knn_r2(z, entropy)
        acc_dom = knn_categorical_accuracy(z, dominant)
        print(f"{v:20s}（{LABELS[v]}）  kNN R² log(POI 數)={r2_poi:+.3f}  "
              f"組成熵={r2_ent:+.3f}  主導類別多數決準確率={acc_dom:.3f}")

        scatter(axes[0, col], z, log_poi,
                f"{LABELS[v]}：色=log(POI 數)  R²={r2_poi:.3f}", "log(POI 數)")
        scatter_categorical(axes[1, col], z, dominant,
                            f"{LABELS[v]}：色=主導類別  準確率={acc_dom:.3f}")
        scatter(axes[2, col], z, entropy,
                f"{LABELS[v]}：色=組成熵  R²={r2_ent:.3f}", "Shannon entropy")

    handles = [LegendPatch(color=CAT_COLORS[i], label=CAT_ZH[i]) for i in range(N_CAT)]
    fig.legend(handles=handles, loc="lower center", ncol=5, fontsize=8,
              frameon=False, bbox_to_anchor=(0.5, -0.01))

    fig.suptitle("v2_ae_tanh_fsce vs v2_ae_fsce：tanh 邊界對 FSCE latent「密度 vs 組成」的影響",
                 fontsize=13)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(OUT, bbox_inches="tight")
    print(f"\n已存 {OUT}")


main()
