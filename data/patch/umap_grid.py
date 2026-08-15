"""不經任何模型，把縮小後的網格直接 flatten 丟進 UMAP，看 2 維上有沒有群聚。

為什麼要有這支：
  這是設計 AE 之前的可行性檢查。UMAP 不需要訓練、專門找流形結構，
  如果連它都壓不出群聚，AE 的 2 維 latent 大概也不會有，
  「城市 POI 飽和 -> latent 出現離群值」的題目假設就站不住。
  所以這裡的產物是決策依據，不是模型的一部分——刻意不碰 torch、不讀任何 ae.py。

為什麼是 2x2 panel：
  格值編碼有兩種選擇（保留密度的 log1p(count) vs. 只看有無的 binary），
  距離度量也有兩種（euclidean vs. cosine）。360 維裡約 99% 是 0，
  這種稀疏高維下 euclidean 很容易被「POI 總數」主導、cosine 則只看組成比例，
  兩個一起看才知道群聚是形狀造成的還是密度造成的。

為什麼主圖不上色：
  這階段只看分布形狀，先不要被標籤（POI 數、類別）先入為主。
  上色版另存 umap_grid_poi.png，是自己判斷用的診斷圖。

已知風險：
  HALF_WIDTH=50（取樣方框 100m）但 GRID*CELL=90m，網格覆蓋不到最外圈 5m，
  那些點會被 clamp 到邊界格，導致邊界格虛高。執行時會印出被 clamp 的點數佔比，
  佔比高的話要回頭調 config 的 GRID 或 HALF_WIDTH，不在這支腳本裡改。

binning 邏輯是照抄 model/v0/ae.py 的 Patches.render()（改成 numpy 版、不旋轉）。
沒有 import 過來，因為 repo 的慣例是各版本目錄自我獨立、複製而非共用。
"""

import os
import sys
import time

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import umap

sys.path.insert(0, os.path.abspath(f"{os.path.dirname(__file__)}/../.."))
from config.dataset import (CELL, GRID, HALF_WIDTH, N_CAT,  # noqa: E402
                            PATCHES, result)

mpl.rcParams["font.family"] = ["Heiti TC"]
mpl.rcParams["axes.unicode_minus"] = False

N_NEIGHBORS = 15
MIN_DIST = 0.1
SEED = 42
POINT_SIZE = 6

OUT_PLAIN = result("v0", "umap_grid.png")
OUT_POI = result("v0", "umap_grid_poi.png")

# (欄標題, UMAP 的 metric)
METRICS = [("euclidean", "euclidean"), ("cosine", "cosine")]


def render_counts():
    """把稀疏點列表展開成 (n_patch, N_CAT, GRID, GRID) 的 count。

    推論階段一律不旋轉，讓每個 patch 的向量唯一。
    """
    d = np.load(PATCHES)
    dx, dy, cat = d["dx"], d["dy"], d["cat"].astype(np.int64)
    offsets, n_poi = d["offsets"], d["n_poi"]
    n = len(n_poi)

    # 每個點屬於第幾個 patch
    owner = np.repeat(np.arange(n), np.diff(offsets))

    raw_ix = np.floor(dx / CELL + GRID / 2).astype(np.int64)
    raw_iy = np.floor(dy / CELL + GRID / 2).astype(np.int64)
    clamped = ((raw_ix < 0) | (raw_ix >= GRID) |
               (raw_iy < 0) | (raw_iy >= GRID)).sum()
    ix = np.clip(raw_ix, 0, GRID - 1)
    iy = np.clip(raw_iy, 0, GRID - 1)

    flat = ((owner * N_CAT + cat) * GRID + iy) * GRID + ix
    counts = np.bincount(flat, minlength=n * N_CAT * GRID * GRID)
    counts = counts.reshape(n, N_CAT, GRID, GRID).astype(np.float32)

    total = len(dx)
    print(f"patch {n}，總點數 {total}，"
          f"取樣方框 {int(2 * HALF_WIDTH)}m，網格只覆蓋 {GRID * CELL}m")
    print(f"落在網格外被 clamp 到邊界格的點：{clamped} / {total} "
          f"({100 * clamped / total:.2f}%)")
    assert counts.sum() == total, "binning 掉點了"

    return counts, n_poi


def main():
    counts, n_poi = render_counts()
    n = len(n_poi)

    encodings = [
        ("log1p(count)", np.log1p(counts).reshape(n, -1)),
        ("binary 0/1", (counts > 0).astype(np.float32).reshape(n, -1)),
    ]
    print(f"flatten 後維度 {encodings[0][1].shape[1]}，"
          f"零值比例 {100 * (counts == 0).mean():.2f}%")

    # 先全部算好 embedding，兩張圖共用
    embeddings = {}
    for enc_name, x in encodings:
        for metric_name, metric in METRICS:
            t0 = time.time()
            reducer = umap.UMAP(
                n_components=2, n_neighbors=N_NEIGHBORS, min_dist=MIN_DIST,
                metric=metric, random_state=SEED,
            )
            z = reducer.fit_transform(x)
            embeddings[(enc_name, metric_name)] = z
            print(f"{enc_name:14s} {metric_name:10s} "
                  f"{time.time() - t0:5.1f}s  "
                  f"x [{z[:, 0].min():7.2f}, {z[:, 0].max():7.2f}]  "
                  f"y [{z[:, 1].min():7.2f}, {z[:, 1].max():7.2f}]")

    _plot(encodings, embeddings, color=None, path=OUT_PLAIN,
          suptitle=f"UMAP：{GRID}x{GRID} 網格直接 flatten（未經模型），n={n}")
    _plot(encodings, embeddings, color=np.log(n_poi), path=OUT_POI,
          suptitle=f"同上，以 log(POI 數) 上色——顏色若呈乾淨漸層代表 UMAP 抓的是密度，n={n}")


def _plot(encodings, embeddings, color, path, suptitle):
    fig, axes = plt.subplots(len(encodings), len(METRICS),
                             figsize=(11, 10), constrained_layout=True)
    sc = None
    for r, (enc_name, _) in enumerate(encodings):
        for c, (metric_name, _) in enumerate(METRICS):
            ax = axes[r, c]
            z = embeddings[(enc_name, metric_name)]
            if color is None:
                ax.scatter(z[:, 0], z[:, 1], s=POINT_SIZE, c="0.35",
                           linewidths=0)
            else:
                sc = ax.scatter(z[:, 0], z[:, 1], s=POINT_SIZE, c=color,
                                cmap="viridis", linewidths=0)
            ax.set_title(f"{enc_name} / {metric_name}")
            ax.set_xticks([])
            ax.set_yticks([])
    if sc is not None:
        fig.colorbar(sc, ax=axes, label="log(POI 數)", shrink=0.6)
    fig.suptitle(suptitle)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"已寫出 {path}")


main()
