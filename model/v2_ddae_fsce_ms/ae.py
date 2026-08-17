"""v2_ddae_fsce_ms：v2_ddae_fsce_copy 加上「多尺度環狀 context」的變體。
模型容量、破壞方式、FSCE loss、Poisson NLL 全部跟 _copy 一樣，唯一的差別是
decoder 多吃一組觀測到的周邊環特徵。

為什麼要有這一版：patch 中位數只有 15 個 POI，10 維 count 的組成訊號有一半
以上是多項式抽樣噪聲（實測 log1p+cosine 的 kNN 圖，把 POI 隨機切兩半各自
建圖，15-NN 一致率只有 0.025，隨機基準 0.012）。把周邊 100/200/400/800m 的
環狀 count 加進來之後，同一個 split-half 協定的一致率跳到 0.245（已排除地理
距離 < 1500m 的平凡鄰居）。但拆解後發現這個增益幾乎全部來自「密度輪廓」而
不是類別組成，訊噪比比組成高一個數量級——所以環特徵不能串進 encoder 輸入，
那會讓它主導距離、把 latent 變成一張密度圖。

因此環特徵走 decoder 的條件變數這條路：

    z       = encoder(x_patch)          # 只吃 patch 自己的 N_CAT 維 count
    log_lam = decoder([z, s_ring])      # s_ring 不經過 bottleneck

decoder 拿得到「這個街區長什麼樣」，latent 就不必浪費維度去記它，被迫只表達
「在這個街區脈絡下，這個 patch 有什麼特別」。

FSCE 的高維鄰接關係仍然用「整包 count 向量的 log1p、cosine 距離」kNN 建的
fuzzy simplicial set，不使用環特徵——encoder 看不到環，用環建圖等於逼 encoder
去擬合它拿不到的資訊。

破壞方式有兩種，用 NOISE_MODE 切換（意義同 v2_dae）：
  "thinning"  binomial thinning：每個 POI 以 1-NOISE_P 的機率被保留。
  "mask"      整個類別歸零：每一類以 NOISE_P 的機率整維被抹成 0。
兩種都會除以 1-NOISE_P 做尺度補償，讓訓練/推論的輸入尺度一致。
"""

import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pyproj import Transformer
from scipy.spatial import cKDTree
from sklearn.neighbors import NearestNeighbors
from umap.umap_ import find_ab_params, fuzzy_simplicial_set

sys.path.insert(0, os.path.abspath(f"{os.path.dirname(__file__)}/../.."))
from config.dataset import (CAT_COL, CATEGORIES, CRS, CSV,  # noqa: E402,F401
                            HALF_WIDTH, N_CAT)

HIDDEN = 64
N_HIDDEN_LAYERS = 4   # v2_dae_fsce 是 2 層，這一版加倍


class Patches:
    """稀疏點列表；agg() 把整個 patch 聚合成一個 (N_CAT,) 的 count 向量。"""

    def __init__(self, path):
        d = np.load(path)
        self.cat = torch.from_numpy(d["cat"].astype(np.int64))
        self.offsets = torch.from_numpy(d["offsets"])
        self.n_poi = d["n_poi"]
        self.lat = d["center_lat"]
        self.lon = d["center_lon"]
        # 平面座標（公尺，CRS 見 config），build_rings() 要用
        self.cx = d["center_x"]
        self.cy = d["center_y"]
        self.n = len(self.n_poi)

    def agg(self, idx):
        """idx 是 patch 編號的 tensor，回傳 (B,N_CAT) 的整包類別 count 向量。"""
        b = len(idx)
        starts, ends = self.offsets[idx], self.offsets[idx + 1]
        lens = ends - starts
        pos = torch.repeat_interleave(starts, lens) + \
            torch.arange(int(lens.sum())) - torch.repeat_interleave(
                torch.cumsum(lens, 0) - lens, lens)
        owner = torch.repeat_interleave(torch.arange(b), lens)
        cat = self.cat[pos]

        flat = owner * N_CAT + cat
        counts = torch.bincount(flat, minlength=b * N_CAT)
        return counts.view(b, N_CAT).float()


def build_rings(center_x, center_y, radii, inner=HALF_WIDTH, mode="total"):
    """算每個 patch 中心周圍「同心方環」裡的 POI 統計，當 decoder 的條件變數。

    center_x / center_y：(N,) 的 patch 中心平面座標（公尺，CRS 見 config），
        直接傳 patches.npz 的 center_x / center_y。
    radii：由小到大的外邊界半徑序列（公尺），每個值定義一個環。
    inner：最內圈的內邊界（公尺），預設 HALF_WIDTH，也就是 patch 自己的窗。
    mode："total" 每個環只回傳一個總 POI 數（密度輪廓），維度 len(radii)；
        "count" 每個環回傳完整的 N_CAT 維類別 count，維度 len(radii)*N_CAT。

    回傳 (N,D) 的 float32 陣列，值都取過 log1p。第 k 段是第 k 個環
    （radii[k-1] 到 radii[k]，k=0 時是 inner 到 radii[0]）。

    mode 預設 "total" 的理由：實測環特徵帶來的 split-half kNN 一致率增益幾乎
    全部來自密度輪廓（每環總數 0.295，跨尺度的類別組成只有 0.059），而
    "count" 的 len(radii)*N_CAT 維向量對這份資料的 patch 數來說幾乎是唯一
    指紋，decoder 會拿它當索引背訓練集——實測 4 環 40 維時 val deviance 在
    epoch 120 觸底之後一路惡化到 baseline 的三倍。

    環一律扣掉 inner 以內的 POI，所以回傳的特徵完全不含 patch 自身的統計。
    這是刻意的：decoder 若拿得到 patch 自己的 count，就能繞過 latent 直接
    重建，latent 會整個塌掉。

    用環狀差分而不是累積窗，因為累積窗的各段高度共線（800m 的窗包含 100m 的
    窗），實測差分版的 split-half kNN 一致率明顯較高（0.245 vs 0.160）。
    也刻意不做面積正規化：除以面積後各尺度變得高度相關，一致率會崩到 0.046。

    窗形用 Chebyshev（p=inf）方窗，跟 data/patch/build_patches.py 一致。
    """
    if mode not in ("total", "count"):
        raise ValueError(f"未知的 mode：{mode}")
    radii = tuple(float(r) for r in radii)
    if any(b <= a for a, b in zip((inner,) + radii, radii)):
        raise ValueError(f"radii 必須嚴格遞增且都大於 inner={inner}：{radii}")

    df = pd.read_csv(CSV)
    cat = df[CAT_COL].map({name: i for i, name in enumerate(CATEGORIES)})
    keep = cat.notna().to_numpy()          # 類別對不上 CATEGORIES 的整列丟掉
    cat = cat.to_numpy()[keep].astype(np.int64)
    transformer = Transformer.from_crs("EPSG:4326", CRS, always_xy=True)
    x, y = transformer.transform(df["lon"].to_numpy()[keep],
                                 df["lat"].to_numpy()[keep])
    tree = cKDTree(np.column_stack([x, y]).astype(np.float64))

    centers = np.column_stack([center_x, center_y]).astype(np.float64)
    cum = []
    for r in (inner,) + radii:
        counts = np.zeros((len(centers), N_CAT))
        for i, idx in enumerate(tree.query_ball_point(centers, r, p=np.inf)):
            counts[i] = np.bincount(cat[idx], minlength=N_CAT)
        cum.append(counts)

    rings = [cum[k + 1] - cum[k] for k in range(len(radii))]
    if mode == "total":
        rings = [r.sum(axis=1, keepdims=True) for r in rings]
    return np.log1p(np.concatenate(rings, axis=1)).astype(np.float32)


def corrupt(x, p, mode="thinning", generator=None):
    """把乾淨的 count 向量破壞成 DAE 的輸入。

    x：(B,N_CAT) 的乾淨 count 向量（float，值是非負整數）。
    p：破壞強度 ∈[0,1)。thinning 是每個 POI 被丟掉的機率，mask 是每一類
       整維被抹成 0 的機率。p=0 時直接回傳 x。
    mode："thinning" 或 "mask"，意義見模組 docstring。
    generator：torch.Generator，給定就用它抽亂數，讓每個 epoch 的破壞可重現。

    回傳跟 x 同 shape 的 (B,N_CAT) tensor，已經除以 1-p 做過尺度補償，
    期望值等於 x，所以可以直接餵進跟推論時同一個 encoder。
    """
    if p <= 0:
        return x
    keep = 1.0 - p
    if mode == "thinning":
        # binomial 沒有吃 generator 的版本，用 x 個 Bernoulli 的和等價實作：
        # 每個 patch 的每一類最多 max_c 個 POI，各自擲一次銅板再依真實 count 遮掉
        max_c = int(x.max().item())
        if max_c == 0:
            return x
        coin = torch.rand(x.shape + (max_c,), generator=generator,
                          device=x.device) < keep
        alive = torch.arange(max_c, device=x.device) < x.unsqueeze(-1)
        noisy = (coin & alive).sum(dim=-1).float()
    elif mode == "mask":
        m = (torch.rand(x.shape, generator=generator, device=x.device) < keep)
        noisy = x * m.float()
    else:
        raise ValueError(f"未知的 mode：{mode}")
    return noisy / keep


def _mlp_block(d_in, d_out, n_layers):
    """疊 n_layers 個 Linear(HIDDEN,HIDDEN)+GELU，前面加一層 d_in->HIDDEN，
    回傳 list of nn.Module（不含最後把 HIDDEN 投影到 d_out 的那一層）。
    """
    layers = [nn.Linear(d_in, HIDDEN), nn.GELU()]
    for _ in range(n_layers - 1):
        layers += [nn.Linear(HIDDEN, HIDDEN), nn.GELU()]
    return layers


class MLPAE(nn.Module):
    """encoder/decoder 各 N_HIDDEN_LAYERS 層隱藏層，latent 前不接激活。
    denoising 完全發生在資料端（見 corrupt()），模型本身不需要知道。

    跟 v2_ddae_fsce_copy 的 MLPAE 唯一差異是 ctx_dim：decoder 的輸入除了 z
    之外還接一段 ctx_dim 維的觀測條件變數（build_rings() 算出的周邊環）。
    encoder 完全不變、看不到 ctx。ctx_dim=0 時行為與 _copy 完全相同。
    """

    def __init__(self, latent_dim=2, ctx_dim=0):
        super().__init__()
        self.ctx_dim = ctx_dim
        self.encoder = nn.Sequential(
            *_mlp_block(N_CAT, HIDDEN, N_HIDDEN_LAYERS),
            nn.Linear(HIDDEN, latent_dim),   # latent 前不接激活，離群值才不會被壓回來
        )
        self.decoder = nn.Sequential(
            *_mlp_block(latent_dim + ctx_dim, HIDDEN, N_HIDDEN_LAYERS),
            nn.Linear(HIDDEN, N_CAT),   # 輸出是 log λ
        )

    def encode(self, x):
        """x 是 (B,N_CAT) 的 count 向量，回傳 (B,latent_dim) 的 z——只跑 encoder。"""
        return self.encoder(x)

    def decode(self, z, s=None):
        """z 是 (B,latent_dim) 的 latent，s 是 (B,ctx_dim) 的條件變數
        （ctx_dim=0 時傳 None）。回傳 (B,N_CAT) 的 log λ。
        """
        if self.ctx_dim == 0:
            return self.decoder(z)
        if s is None:
            raise ValueError(f"ctx_dim={self.ctx_dim}，decode() 必須傳 s")
        return self.decoder(torch.cat([z, s], dim=1))

    def forward(self, x, s=None):
        """x 是 (B,N_CAT) 的 count 向量（訓練時是加噪版、推論時是乾淨版），
        s 是 (B,ctx_dim) 的條件變數。
        回傳 (z, log_lam)：z 是 (B,latent_dim) 的 latent，log_lam 是 (B,N_CAT)。
        """
        z = self.encoder(x)
        return z, self.decode(z, s)


def poisson_nll(log_lam, x):
    """Poisson NLL，省略跟模型無關的 log(y!) 常數項，回傳每個 patch 一個數字。"""
    cell = torch.exp(log_lam) - x * log_lam
    return cell.mean(dim=1)


def poisson_deviance(log_lam, x):
    """Poisson deviance：2·(y·log(y/λ) - (y - λ))，越小越好、有下界，可跨 patch 比較。"""
    lam = torch.exp(log_lam)
    cell = 2 * (torch.xlogy(x, x) - x * log_lam - x + lam)
    return cell.mean(dim=1)


def build_fsce_graph(x, n_neighbors=15, metric="cosine"):
    """x 是高維空間的特徵矩陣 (N,D)（這裡傳乾淨 count 的 log1p），在上面建一次
    UMAP 的 fuzzy simplicial set。回傳 edge_i、edge_j：邊兩端的 patch 編號 (E,)
    LongTensor；edge_w：這條邊在高維空間的模糊隸屬度 (E,)∈(0,1] FloatTensor，
    當作 FSCE loss 裡的正樣本權重；a、b：UMAP 低維核函數 1/(1+a·d^(2b)) 的形狀
    參數，由 find_ab_params(spread=1.0, min_dist=0.1) 算出。
    """
    knn = NearestNeighbors(n_neighbors=n_neighbors, metric=metric).fit(x)
    knn_dists, knn_idx = knn.kneighbors(x)
    graph, _, _ = fuzzy_simplicial_set(
        x, n_neighbors=n_neighbors, random_state=0, metric=metric,
        knn_indices=knn_idx, knn_dists=knn_dists,
    )
    graph = graph.tocoo()
    edge_i = torch.from_numpy(graph.row).long()
    edge_j = torch.from_numpy(graph.col).long()
    edge_w = torch.from_numpy(graph.data).float()
    a, b = find_ab_params(spread=1.0, min_dist=0.1)
    return edge_i, edge_j, edge_w, a, b


def fsce_loss(z_i, z_j, w, a, b, eps=1e-4):
    """FSCE（fuzzy set cross entropy）：z_i、z_j 是一批 pair 的 latent 座標
    (P,latent_dim)，w 是這對點在高維空間的模糊隸屬度 (P,)——正樣本填
    build_fsce_graph() 算出的 edge_w，負樣本填 0。a、b 是 UMAP 核函數參數。
    回傳每個 pair 一個數字：q（低維距離換算出的相似度）跟 w 差越多，這個數字
    越大，逼著 encoder 把 w 大的點拉近、w=0 的點推遠。

    d2 用 eps 墊底：b<1 時 d2^b 在 d2=0 處的梯度是無限大，兩個不同 patch 剛好
    被 encoder 映到同一點（或負樣本剛好抽到 i==j）並不無法排除，沒墊底會讓
    backward 炸成 NaN。
    """
    d2 = (z_i - z_j).pow(2).sum(dim=1).clamp(min=eps)
    q = (1.0 + a * d2.pow(b)).reciprocal().clamp(eps, 1 - eps)
    return -(w * q.log() + (1 - w) * (1 - q).log())
