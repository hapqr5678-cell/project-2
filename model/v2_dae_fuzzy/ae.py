"""v2_ae：v2 系列的基準組。輸入不再是空間矩陣，而是整個 patch 聚合成一個
(N_CAT,) 的類別 count 向量，MLP AutoEncoder + Poisson NLL。

v0/v1 系列把 patch 內的 POI 依 (x,y) binning 成 GRID×GRID 的矩陣，
「格子」本身帶有空間資訊，latent 有機會偷學到形狀/朝向。v2 系列刻意拿掉
這個概念：輸入只是半徑 HALF_WIDTH 圓內「每一類有幾個」的一維向量，
例如 [餐飲:3, 零售:2, 其他:0, ...]——沒有 x,y、沒有 cell、沒有卷積。
latent 因此只能從純粹的類別組成比例與總量去判斷飽和度，跟這個題目
「城市 POI 飽和度」的假設更貼近：我們關心的是「這裡的類別組合合不合理」，
不是「這裡的形狀合不合理」。

decoder 輸出的是 log λ（跟 v0_poisson_nll 一樣的理由：λ 必須為正、
對 log λ 的梯度是 λ-y，數值溫和），loss 是每個 patch N_CAT 維的
Poisson NLL，不再有「圓內/圓外」的區別——本來就沒有格子可以區分。

v2_ae / v2_perceiver / v2_vae 三個版本的 decoder、loss 完全共用同一份定義；
差別只在 encoder 怎麼把「一包 POI」壓成 2 維 latent：
  v2_ae        直接吃已經算好的 (N_CAT,) 聚合向量，MLP
  v2_perceiver 吃還沒聚合的原始 POI 類別 token 集合，靠 cross-attention 自己聚合
  v2_vae       跟 v2_ae 一樣吃聚合向量，但 encoder 出口機率化(mu/logvar)
"""

from dataclasses import dataclass
import math
import os
import sys

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.abspath(f"{os.path.dirname(__file__)}/../.."))
from config.dataset import N_CAT, HALF_WIDTH  # noqa: E402,F401

HIDDEN = 64
MIN_K_DIST_SCALE = 1e-3
SMOOTH_K_TOLERANCE = 1e-5
PROB_EPS = 1e-4
LATENT_DIST_EPS = 1e-8


class Patches:
    """稀疏點列表；agg() 把整個 patch 聚合成一個 (N_CAT,) 的 count 向量。"""

    def __init__(self, path):
        d = np.load(path)
        self.cat = torch.from_numpy(d["cat"].astype(np.int64))
        self.offsets = torch.from_numpy(d["offsets"])
        self.n_poi = d["n_poi"]
        self.lat = d["center_lat"]
        self.lon = d["center_lon"]
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


class MLPAE(nn.Module):
    """in = out 的純 MLP AutoEncoder：(N_CAT,) 向量進，(N_CAT,) 的 log λ 出。"""

    def __init__(self, latent_dim=2):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(N_CAT, HIDDEN),
            nn.GELU(),
            nn.Linear(HIDDEN, HIDDEN),
            nn.GELU(),
            nn.Linear(HIDDEN, latent_dim),   # latent 前不接激活，離群值才不會被壓回來
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, HIDDEN),
            nn.GELU(),
            nn.Linear(HIDDEN, HIDDEN),
            nn.GELU(),
            nn.Linear(HIDDEN, N_CAT),   # 輸出是 log λ
        )

    def forward(self, x):
        z = self.encoder(x)
        return z, self.decoder(z)


def poisson_nll(log_lam, x):
    """Poisson NLL，省略跟模型無關的 log(y!) 常數項，回傳每個 patch 一個數字。"""
    cell = torch.exp(log_lam) - x * log_lam
    return cell.mean(dim=1)


def poisson_deviance(log_lam, x):
    """Poisson deviance：2·(y·log(y/λ) - (y - λ))，越小越好、有下界，可跨 patch 比較。"""
    lam = torch.exp(log_lam)
    cell = 2 * (torch.xlogy(x, x) - x * log_lam - x + lam)
    return cell.mean(dim=1)


@dataclass(frozen=True)
class FuzzyGraph:
    """Sparse undirected fuzzy graph stored as one copy of each edge."""

    head: torch.Tensor
    tail: torch.Tensor
    weight: torch.Tensor
    n_nodes: int

    @property
    def n_edges(self):
        return len(self.weight)


def _smooth_knn_dist(distances, n_neighbors, n_iter=64):
    """Solve UMAP's local smooth-kNN scale for each row.

    ``distances`` must be sorted and contain self at column zero. The
    implementation follows UMAP's local_connectivity=1 behavior.
    """
    target = math.log2(n_neighbors)
    mean_distance = distances.mean().item()
    sigmas = torch.empty(len(distances), dtype=torch.float32)
    rhos = torch.empty(len(distances), dtype=torch.float32)

    for i, row in enumerate(distances):
        positive = row[row > 0]
        rho = positive[0].item() if len(positive) else 0.0
        rhos[i] = rho

        lo = 0.0
        hi = math.inf
        mid = 1.0
        shifted = row[1:] - rho

        for _ in range(n_iter):
            membership = torch.exp(-torch.clamp_min(shifted, 0.0) / mid)
            value = membership.sum().item()
            if abs(value - target) < SMOOTH_K_TOLERANCE:
                break
            if value > target:
                hi = mid
                mid = (lo + hi) / 2.0
            else:
                lo = mid
                mid = mid * 2.0 if math.isinf(hi) else (lo + hi) / 2.0

        local_floor = MIN_K_DIST_SCALE * (
            row.mean().item() if rho > 0 else mean_distance
        )
        sigmas[i] = max(mid, local_floor)

    return sigmas, rhos


def build_fuzzy_graph(features, n_neighbors=15):
    """Build an exact UMAP-style fuzzy kNN graph from all training features.

    ``n_neighbors`` follows UMAP's convention and includes self, so the graph
    has at most ``n_neighbors - 1`` outgoing non-self neighbors per node before
    fuzzy union symmetrization.
    """
    x = features.detach().float().cpu()
    if x.ndim != 2:
        raise ValueError("features must have shape (n_samples, n_features)")
    if not torch.isfinite(x).all():
        raise ValueError("features contain NaN or infinity")

    n_nodes = len(x)
    if not 2 <= n_neighbors < n_nodes:
        raise ValueError("n_neighbors must be at least 2 and smaller than n_samples")

    pairwise = torch.cdist(x, x, p=2)
    pairwise.fill_diagonal_(-1.0)
    knn_dist, knn_idx = torch.topk(
        pairwise, n_neighbors, dim=1, largest=False, sorted=True
    )

    expected_self = torch.arange(n_nodes)
    if not torch.equal(knn_idx[:, 0], expected_self):
        raise RuntimeError("failed to place self at the start of every kNN row")
    knn_dist[:, 0] = 0.0

    sigmas, rhos = _smooth_knn_dist(knn_dist, n_neighbors)
    shifted = knn_dist - rhos[:, None]
    directed_weight = torch.exp(
        -torch.clamp_min(shifted, 0.0) / sigmas[:, None]
    )
    directed_weight[:, 0] = 0.0

    rows = torch.arange(n_nodes)[:, None].expand_as(knn_idx)
    directed = torch.zeros((n_nodes, n_nodes), dtype=torch.float32)
    directed[rows, knn_idx] = directed_weight

    product = directed * directed.T
    symmetric = directed + directed.T - product
    upper = torch.triu(symmetric, diagonal=1)
    head, tail = (upper > 0).nonzero(as_tuple=True)
    weight = symmetric[head, tail]

    if len(weight) == 0 or not torch.isfinite(weight).all():
        raise RuntimeError("fuzzy graph has no finite positive edges")
    return FuzzyGraph(head=head, tail=tail, weight=weight, n_nodes=n_nodes)


def sample_edge_batch(graph, n_positive, negative_sample_rate, generator):
    """Sample membership-weighted positive edges and uniform negative pairs."""
    if n_positive <= 0:
        raise ValueError("n_positive must be positive")
    if negative_sample_rate <= 0:
        raise ValueError("negative_sample_rate must be positive")

    edge_idx = torch.multinomial(
        graph.weight,
        n_positive,
        replacement=True,
        generator=generator,
    )
    positive_head = graph.head[edge_idx]
    positive_tail = graph.tail[edge_idx]
    # The graph stores each undirected edge only once. Random orientation keeps
    # the negative-sampling anchor distribution from depending on node index.
    swap = torch.rand(n_positive, generator=generator) < 0.5
    oriented_head = torch.where(swap, positive_tail, positive_head)
    oriented_tail = torch.where(swap, positive_head, positive_tail)
    positive_head, positive_tail = oriented_head, oriented_tail

    negative_head = positive_head.repeat_interleave(negative_sample_rate)
    negative_tail = torch.randint(
        graph.n_nodes,
        (len(negative_head),),
        generator=generator,
    )
    same = negative_head == negative_tail
    while same.any():
        negative_tail[same] = torch.randint(
            graph.n_nodes,
            (int(same.sum()),),
            generator=generator,
        )
        same = negative_head == negative_tail

    return positive_head, positive_tail, negative_head, negative_tail


def _latent_membership(z_head, z_tail, a, b):
    squared_distance = (z_head - z_tail).square().sum(dim=1)
    # b < 1 makes d2**b have an infinite derivative at d2=0. Identical count
    # vectors necessarily produce identical latents, so keep the power away
    # from exactly zero to prevent autograd from forming 0 * infinity = NaN.
    safe_squared_distance = squared_distance + LATENT_DIST_EPS
    return 1.0 / (1.0 + a * safe_squared_distance.pow(b))


def fuzzy_set_cross_entropy(
    positive_head,
    positive_tail,
    negative_head,
    negative_tail,
    a,
    b,
):
    """Return sampled fuzzy CE plus separately reportable pull/push terms."""
    positive_q = _latent_membership(positive_head, positive_tail, a, b)
    negative_q = _latent_membership(negative_head, negative_tail, a, b)

    # Add epsilon inside the logarithm instead of clamping q. A max-clamp would
    # zero the repulsive gradient precisely when two negative samples are too
    # close, which is where the push term is needed most.
    normalizer = 1.0 + PROB_EPS
    attraction = -torch.log((positive_q + PROB_EPS) / normalizer)
    repulsion = -torch.log((1.0 - negative_q + PROB_EPS) / normalizer)
    cross_entropy = torch.cat((attraction, repulsion)).mean()
    return cross_entropy, attraction.mean(), repulsion.mean()
