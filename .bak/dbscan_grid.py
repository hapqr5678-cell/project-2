"""對 patch 的類別 count 向量做 DBSCAN 分群，跟 AE 的 latent 分群互相對照。

特徵直接對 data/patch/patches.npz 的稀疏點列表算 bincount(cat)，也就是
每個 patch 的 count-based encoding [餐飲:1, 零售:4, 夜生活:0, ...]，不經
過模型、不用 log1p、不依賴 build_features.py。理由是 DBSCAN 用歐氏距離，
而 CLAUDE.md 定義的輸入本來就是原始 count，直接對這個空間分群才是跟
latent 分群同一個問題的另一種解法。

DBSCAN 對尺度敏感：count 越大的 patch（餐飲熱區）彼此的歐氏距離天生比
稀疏 patch 大，同一個 EPS 在密集區太鬆、在稀疏區太緊。這裡先不標準化，
如果分群結果被 n_total 主導（大 patch 全部同一群、小 patch 全部噪聲），
再考慮改用 StandardScaler 或改良過的 metric（如 cosine）。

已知限制：patch 密度（市中心 vs 郊區）本來就跨好幾個量級，單一全域
EPS 沒辦法同時適配，密集區容易被切成一堆小群、稀疏區大量標成噪聲，
這是 DBSCAN 對變密度資料的結構性限制，不是參數沒調好。EPS 也強烈依賴
`config/dataset.py` 目前的 HALF_WIDTH/CELL（決定 patch 裡有多少點、
count 的量級），改了幾何參數、重跑過 build_patches.py 之後，一定要
看下面印出的 k-距離分布重新調 EPS，不能沿用舊值。如果要更乾淨的分群，
之後可以考慮 HDBSCAN。

DBSCAN 不是在原始 N_CAT 維 count 空間上跑，是先用 t-SNE 降到 2 維、
每軸各自 min-max 正規化到 [-1, 1]，再對這個 2D 座標分群。這是刻意選的
（使用者要求），但要注意跟前一版（直接在原始空間分群）比，這裡的
EPS、噪聲比例、輪廓係數全部不能跟原始空間的結果比較數值大小——尺度
完全不同。已知風險：t-SNE 的優化目標只保證局部近鄰關係，不保證全域
距離/密度有意義，換 random_state 或 perplexity，這裡選出的 EPS 可能
就不適用；而且 t-SNE 常把群壓成彎曲、非凸的形狀，輪廓係數（假設群是
凸的、用歐氏距離判斷）在這種空間上可能給出誤導的低分甚至負分，分數
低不代表分群品質差，要搭配左圖肉眼看形狀。
"""

import os
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler

sys.path.insert(0, os.path.abspath(f"{os.path.dirname(__file__)}/.."))
from config.dataset import N_CAT, PATCHES  # noqa: E402

mpl.rcParams["font.family"] = ["Heiti TC"]
mpl.rcParams["axes.unicode_minus"] = False

EPS = 0.1
MIN_SAMPLES = 5

OUT_PNG = os.path.join(os.path.dirname(__file__), "dbscan_grid.png")


def main():
    d = np.load(PATCHES)
    cat, offsets = d["cat"].astype(np.int64), d["offsets"]
    n = len(offsets) - 1
    n_total = np.diff(offsets)
    owner = np.repeat(np.arange(n), n_total)

    x = np.bincount(owner * N_CAT + cat, minlength=n * N_CAT)
    x = x.reshape(n, N_CAT).astype(np.float64)
    lat, lon = d["center_lat"], d["center_lon"]

    z = TSNE(n_components=2, init="pca", random_state=0).fit_transform(x)
    z = MinMaxScaler(feature_range=(-1, 1)).fit_transform(z)

    nn = NearestNeighbors(n_neighbors=MIN_SAMPLES).fit(z)
    kd = np.sort(nn.kneighbors(z)[0][:, -1])
    print(f"k={MIN_SAMPLES} 最近鄰距離（t-SNE 正規化後）：", end="")
    for p in [50, 70, 90, 95]:
        print(f" p{p}={np.percentile(kd, p):.3f}", end="")
    print(f"  (目前 EPS={EPS})\n")

    db = DBSCAN(eps=EPS, min_samples=MIN_SAMPLES, metric="euclidean").fit(z)
    labels = db.labels_
    n = len(labels)
    uniq, cnt = np.unique(labels, return_counts=True)
    n_clusters = (uniq != -1).sum()
    n_noise = cnt[uniq == -1][0] if -1 in uniq else 0

    print(f"EPS={EPS} MIN_SAMPLES={MIN_SAMPLES}")
    print(f"{n} 個 patch，{n_clusters} 群，噪聲 {n_noise}（{n_noise / n:.1%}）")

    # 輪廓係數只在非噪聲點上算（噪聲不是群），至少要有兩群才算得出來
    keep = labels != -1
    if n_clusters >= 2:
        s = silhouette_score(z[keep], labels[keep])
        print(f"輪廓係數（t-SNE 空間，排除噪聲，{keep.sum()} 點）：{s:.3f}")
    else:
        print("輪廓係數：無法計算（非噪聲的群數 < 2）")

    order = uniq[np.argsort(-cnt)]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    ax = axes[0]
    m = labels == -1
    ax.scatter(z[m, 0], z[m, 1], s=3, c="lightgray")
    for c in order[order != -1]:
        m = labels == c
        ax.scatter(z[m, 0], z[m, 1], s=4)
    ax.set_title("t-SNE(count 向量) 正規化後的 DBSCAN 分群")

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
