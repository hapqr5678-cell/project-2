"""對 patch 的類別 count 向量做 HDBSCAN 分群，是 dbscan_grid.py 的變體。

跟 dbscan_grid.py 讀一樣的特徵（直接對 data/patch/patches.npz 現算
bincount(cat)，不經過模型、不用 log1p、不依賴 build_features.py）。

換成 HDBSCAN 是因為 dbscan_grid.py 印出的診斷已經證實 patch 密度（市
中心 vs 郊區）跨好幾個量級，DBSCAN 的單一全域 EPS 沒辦法同時適配。
HDBSCAN 用 MIN_CLUSTER_SIZE 取代 EPS，對每個密度尺度各自找穩定的群，
理論上密集區不會被切碎、稀疏區也不會整片變噪聲。

MIN_CLUSTER_SIZE / MIN_SAMPLES 不是手動猜的，是對 MCS_GRID x MS_GRID
做網格搜尋，每組算排除噪聲後的輪廓係數，挑最高的一組。輪廓係數只在
非噪聲點上算、群數 >= 2 才有定義，所以噪聲比例 > NOISE_CAP 的組合直接
跳過——不然全部丟噪聲留兩三個緊密的小群這種退化解，輪廓係數會虛高但
沒有分群的意義（HDBSCAN 允許把大部分點標成噪聲，跟 k-means 那種輪廓
係數搜尋不同，這裡必須額外擋這條退化路徑）。

即使擋了 NOISE_CAP，輪廓係數本身還是偏好「切掉幾個極端離群點自成一
群、其餘全部留在一個大群」這種解——兩群彼此距離拉滿，係數自然很高，
但這不是「找到有意義的多群結構」，是找到離群值。實測最佳解常常長這樣
（例如 1204+7 兩群、22 噪聲）。所以看到「最佳」不能只看係數，一定要看
下面印出的群大小分布；如果第二大群小到個位數，代表這組參數只是把
DBSCAN/HDBSCAN 拿來當離群值偵測器用，跟「分群」是两回事，此時該看的
是次高分、群數更多、群大小更平衡的那幾組。
"""

import os
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import HDBSCAN
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score

sys.path.insert(0, os.path.abspath(f"{os.path.dirname(__file__)}/.."))
from config.dataset import N_CAT, PATCHES  # noqa: E402

mpl.rcParams["font.family"] = ["Heiti TC"]
mpl.rcParams["axes.unicode_minus"] = False

MCS_GRID = list(range(3, 31))
MS_GRID = [None, 3, 5, 8, 10, 15]
NOISE_CAP = 0.7   # 噪聲比例超過這個門檻的組合視為退化解，不列入候選

OUT_PNG = os.path.join(os.path.dirname(__file__), "hdbscan_grid.png")


def search(x):
    """對 MCS_GRID x MS_GRID 網格搜尋，回傳依輪廓係數排序的結果表。"""
    rows = []
    for mcs in MCS_GRID:
        for ms in MS_GRID:
            labels = HDBSCAN(min_cluster_size=mcs, min_samples=ms,
                              metric="euclidean", copy=True).fit_predict(x)
            uniq, cnt = np.unique(labels, return_counts=True)
            n_clusters = (uniq != -1).sum()
            n_noise = cnt[uniq == -1][0] if -1 in uniq else 0
            noise_frac = n_noise / len(labels)
            if n_clusters < 2 or noise_frac > NOISE_CAP:
                continue
            keep = labels != -1
            s = silhouette_score(x[keep], labels[keep])
            rows.append((s, mcs, ms, n_clusters, noise_frac))
    rows.sort(key=lambda r: -r[0])
    return rows


def main():
    d = np.load(PATCHES)
    cat, offsets = d["cat"].astype(np.int64), d["offsets"]
    n = len(offsets) - 1
    n_total = np.diff(offsets)
    owner = np.repeat(np.arange(n), n_total)

    x = np.bincount(owner * N_CAT + cat, minlength=n * N_CAT)
    x = x.reshape(n, N_CAT).astype(np.float64)
    lat, lon = d["center_lat"], d["center_lon"]

    rows = search(x)
    if not rows:
        print(f"搜尋完 {len(MCS_GRID) * len(MS_GRID)} 組參數，"
              f"沒有一組同時滿足群數>=2 且噪聲<={NOISE_CAP:.0%}，放寬 NOISE_CAP 再試")
        return

    print(f"{'輪廓係數':>8}{'mcs':>6}{'ms':>6}{'群數':>6}{'噪聲%':>8}")
    for s, mcs, ms, n_clusters, noise_frac in rows[:10]:
        print(f"{s:>8.3f}{mcs:>6}{str(ms):>6}{n_clusters:>6}{noise_frac:>8.1%}")

    best_s, mcs, ms, n_clusters, noise_frac = rows[0]
    print(f"\n最佳：min_cluster_size={mcs} min_samples={ms}"
          f"（輪廓係數 {best_s:.3f}，{n_clusters} 群，噪聲 {noise_frac:.1%}）")

    labels = HDBSCAN(min_cluster_size=mcs, min_samples=ms,
                      metric="euclidean", copy=True).fit_predict(x)
    uniq, cnt = np.unique(labels, return_counts=True)
    order = uniq[np.argsort(-cnt)]
    sizes = ", ".join(str(cnt[uniq == c][0]) for c in order if c != -1)
    print(f"群大小（由大到小）：{sizes}  <- 看這行，太懸殊代表在找離群值不是分群")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    z = TSNE(n_components=2, init="pca", random_state=0).fit_transform(x)
    ax = axes[0]
    m = labels == -1
    ax.scatter(z[m, 0], z[m, 1], s=3, c="lightgray")
    for c in order[order != -1]:
        m = labels == c
        ax.scatter(z[m, 0], z[m, 1], s=4)
    ax.set_title("t-SNE(count 向量) 上的 HDBSCAN 分群")

    ax = axes[1]
    m = labels == -1
    ax.scatter(lon[m], lat[m], s=3, c="lightgray")
    for c in order[order != -1]:
        m = labels == c
        ax.scatter(lon[m], lat[m], s=4)
    ax.set_title("分群結果的地理分布")
    ax.set_aspect("equal")

    fig.savefig(OUT_PNG, dpi=150, bbox_inches="tight")
    print(f"\n已存 {OUT_PNG}")


main()
