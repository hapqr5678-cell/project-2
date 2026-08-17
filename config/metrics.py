"""latent 品質的共用量化指標。

給 model/<version>/result/latents.npz 裡的 z 打分，讓不同版本（v2_ddae、
v2_ddae_fsce、v2_ddae_tanh_fsce…）可以用同一把尺比較，而不是只能看散點圖。

指標分四類：
  重建   val Poisson deviance —— 模型有沒有把資料學好。
  保鄰   trustworthiness / continuity / kNN recall —— 高維鄰居關係有沒有被
         latent 保住，這正是 FSCE 想優化的東西。
  分群   高維 KMeans 標籤在 latent 上的 ARI 與 silhouette —— latent 有沒有把
         高維的群分開。
  展開   佔用熵與 PCA 各向異性 —— latent 有沒有塌成一條線或擠成一團，
         也就是這張圖「還剩多少可用的表達空間」。

高維空間一律用「整包 count 的 log1p + cosine」，跟 build_fsce_graph() 建圖
時的定義一致；latent 空間一律用 euclidean。
"""

import numpy as np
from scipy.stats import spearmanr
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.metrics import pairwise_distances


def patch_counts(patches_npz):
    """把 patches.npz 的稀疏點列表聚合成每個 patch 一條 count 向量。

    patches_npz：data/patch/patches.npz 的路徑。

    回傳 (N,N_CAT) 的 float64 陣列，第 i 列是第 i 個 patch 各類別的 POI 數量。
    """
    with np.load(patches_npz) as d:
        cat = d["cat"].astype(np.int64)
        offsets = d["offsets"]
        n = len(d["n_poi"])
    n_cat = int(cat.max()) + 1 if len(cat) else 1
    owner = np.repeat(np.arange(n), np.diff(offsets))
    flat = owner * n_cat + cat
    return np.bincount(flat, minlength=n * n_cat).reshape(n, n_cat).astype(float)


def val_split(n, val_frac=0.1, seed=0):
    """重現 train.py 的 train/val 切分（torch.randperm，同 seed 同結果）。

    n：patch 總數。val_frac：驗證集比例。seed：train.py 裡的 SEED。

    回傳 (val_idx, train_idx)，兩個都是 int64 的 numpy 索引陣列。
    """
    import torch
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=g).numpy()
    n_val = int(n * val_frac)
    return perm[:n_val], perm[n_val:]


def _rank_matrix(dist):
    """把距離矩陣轉成名次矩陣。

    dist：(N,N) 的距離矩陣。

    回傳 (N,N) 的 int 陣列，rank[i,j] 是 j 在「以 i 為中心由近到遠」排序中的
    名次；自己是 0，最近的鄰居是 1。
    """
    order = np.argsort(dist, axis=1, kind="stable")
    rank = np.empty_like(order)
    rows = np.arange(dist.shape[0])[:, None]
    rank[rows, order] = np.arange(dist.shape[1])[None, :]
    return rank


def _neighbors(dist, k):
    """取每個點最近的 k 個鄰居（排除自己）。

    dist：(N,N) 的距離矩陣。k：鄰居數。

    回傳 (N,k) 的索引陣列。
    """
    return np.argsort(dist, axis=1, kind="stable")[:, 1:k + 1]


def _trustworthiness(d_from, d_to, k):
    """通用的 trustworthiness：以 d_from 的鄰居為準，罰它們在 d_to 裡排多遠。

    d_from / d_to：(N,N) 距離矩陣。k：鄰居數。

    回傳 float ∈[0,1]，1 代表 d_from 的 k 鄰居在 d_to 裡也都是 k 鄰居。
    把 (高維, 低維) 傳進去得到 continuity，(低維, 高維) 得到 trustworthiness。
    """
    n = d_from.shape[0]
    rank_to = _rank_matrix(d_to)
    nn_from, nn_to = _neighbors(d_from, k), _neighbors(d_to, k)
    penalty = 0.0
    for i in range(n):
        intruders = np.setdiff1d(nn_from[i], nn_to[i], assume_unique=True)
        penalty += np.sum(rank_to[i, intruders] - k)
    return 1.0 - 2.0 / (n * k * (2 * n - 3 * k - 1)) * penalty


def neighborhood_scores(d_hi, d_lo, k=15):
    """保鄰性三兄弟。

    d_hi / d_lo：(N,N) 的高維 / latent 距離矩陣。k：鄰居數，預設對齊建圖的
    N_NEIGHBORS=15。

    回傳 dict：
      trust      trustworthiness ∈[0,1]，latent 上的鄰居在高維也是鄰居的程度
                 （罰「假鄰居」，也就是圖上看起來像一群但其實不像的點）。
      cont       continuity ∈[0,1]，高維鄰居在 latent 上也保持鄰近的程度
                 （罰「被拆散的真鄰居」）。
      knn_recall 高維 k 鄰居與 latent k 鄰居的平均交集比例 ∈[0,1]。
    """
    nn_hi, nn_lo = _neighbors(d_hi, k), _neighbors(d_lo, k)
    overlap = np.mean([len(np.intersect1d(a, b)) / k for a, b in zip(nn_hi, nn_lo)])
    return {
        "trust": _trustworthiness(d_lo, d_hi, k),
        "cont": _trustworthiness(d_hi, d_lo, k),
        "knn_recall": float(overlap),
    }


def distance_spearman(d_hi, d_lo):
    """高維與 latent 的兩兩距離的 Spearman 等級相關。

    d_hi / d_lo：(N,N) 距離矩陣。

    回傳 float ∈[-1,1]，衡量全域尺度（不只鄰居）有沒有被保住；保鄰指標好但
    這個低，代表 latent 只對了局部、整張圖的遠近關係是亂的。
    """
    iu = np.triu_indices(d_hi.shape[0], k=1)
    return float(spearmanr(d_hi[iu], d_lo[iu]).statistic)


def cluster_scores(x_hi, z, n_clusters=8, seed=0):
    """用高維分群當 pseudo-label，檢查 latent 有沒有把同一群留在一起。

    x_hi：(N,D) 高維特徵（這裡傳 log1p 後的 count）。z：(N,2) latent。
    n_clusters：KMeans 的群數。seed：KMeans 的 random_state。

    回傳 dict：
      ari         latent 上重跑 KMeans 得到的分群與高維分群的 Adjusted Rand
                  Index ∈[-1,1]，1 代表兩邊切出同一組群。
      silhouette  高維標籤畫在 latent 上的 silhouette ∈[-1,1]，衡量這些群在
                  latent 平面上分不分得開。
    """
    lab_hi = KMeans(n_clusters, n_init=10, random_state=seed).fit_predict(x_hi)
    lab_lo = KMeans(n_clusters, n_init=10, random_state=seed).fit_predict(z)
    return {
        "ari": float(adjusted_rand_score(lab_hi, lab_lo)),
        "silhouette": float(silhouette_score(z, lab_hi)),
    }


def spread_scores(z, bins=32):
    """latent 平面被用掉多少——「還有多少表達空間」的代理指標。

    z：(N,2) latent。bins：把 latent 的 bounding box 切成 bins×bins 個格子。

    回傳 dict：
      occupancy  佔用熵除以 log(有點的格子數上限)，∈[0,1]。點平均鋪滿愈接近 1，
                 全擠在少數幾格愈接近 0。
      aniso      PCA 兩個主軸標準差的比值 ∈(0,1]，1 是圓形分布，接近 0 代表
                 latent 幾乎塌成一條線、第二維其實沒在用。
    """
    h, _, _ = np.histogram2d(z[:, 0], z[:, 1], bins=bins)
    p = h[h > 0] / h.sum()
    ent = -np.sum(p * np.log(p))
    max_ent = np.log(min(len(p), bins * bins))
    sd = np.sqrt(np.linalg.eigvalsh(np.cov(z.T)))
    return {
        "occupancy": float(ent / max_ent) if max_ent > 0 else 0.0,
        "aniso": float(sd[0] / sd[1]) if sd[1] > 0 else 0.0,
    }


def evaluate(z, err, x_hi, val_idx, k=15, n_clusters=8, seed=0):
    """跑完整套指標。

    z：(N,2) latent。err：(N,) 每個 patch 的 Poisson deviance（latents.npz 的
    err 欄）。x_hi：(N,D) 高維 count。val_idx：驗證集索引，用來只取沒被訓練
    到的那些 patch 算 deviance。k / n_clusters / seed 同上面各函式。

    回傳 dict，把 val_dev（驗證集平均 deviance，越小越好）跟 neighborhood /
    distance / cluster / spread 的所有欄位攤平在同一層。
    """
    x_log = np.log1p(x_hi)
    d_hi = pairwise_distances(x_log, metric="cosine")
    d_lo = pairwise_distances(z, metric="euclidean")
    out = {"val_dev": float(np.mean(err[val_idx]))}
    out.update(neighborhood_scores(d_hi, d_lo, k))
    out["dist_rho"] = distance_spearman(d_hi, d_lo)
    out.update(cluster_scores(x_log, z, n_clusters, seed))
    out.update(spread_scores(z))
    return out
