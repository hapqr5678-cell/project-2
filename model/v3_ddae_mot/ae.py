"""
破壞方式有兩種，用 NOISE_MODE 切換（意義同 v2_dae）：
  "thinning"  binomial thinning：每個 POI 以 1-NOISE_P 的機率被保留。
  "mask"      整個類別歸零：每一類以 NOISE_P 的機率整維被抹成 0。
兩種都會除以 1-NOISE_P 做尺度補償，讓訓練/推論的輸入尺度一致。
"""

import os
import sys

import numpy as np
import torch
import torch.nn as nn
from sklearn.neighbors import NearestNeighbors
from umap.umap_ import find_ab_params, fuzzy_simplicial_set

sys.path.insert(0, os.path.abspath(f"{os.path.dirname(__file__)}/../.."))
from config.dataset import N_CAT, HALF_WIDTH  # noqa: E402,F401

HIDDEN = 64
N_EXPERTS = 3
ROUTER_TEMPERATURE = 1.0


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


class MoEEncoder(nn.Module):
    """共享特徵抽取器加上多個 latent expert 的 dense MoE encoder。

    前三層 MLP 是所有 expert 共用的 trunk；每個 expert 各自保留原 encoder
    的第四層與 latent 投影。router 依每個 patch 的共享表示產生 softmax
    權重，最後的 z 是所有 expert latent 的凸組合。
    """

    def __init__(self, latent_dim=2, n_experts=N_EXPERTS,
                 temperature=ROUTER_TEMPERATURE):
        super().__init__()
        if n_experts < 2:
            raise ValueError("n_experts 必須至少為 2")
        if temperature <= 0:
            raise ValueError("router temperature 必須大於 0")
        self.n_experts = n_experts
        self.temperature = temperature
        self.trunk = nn.Sequential(
            nn.Linear(N_CAT, HIDDEN),
            nn.LayerNorm(HIDDEN),
            nn.GELU(),
            nn.Linear(HIDDEN, HIDDEN),
            nn.LayerNorm(HIDDEN),
            nn.GELU(),
            nn.Linear(HIDDEN, HIDDEN),
            nn.LayerNorm(HIDDEN),
            nn.GELU(),
        )
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(HIDDEN, HIDDEN),
                nn.LayerNorm(HIDDEN),
                nn.GELU(),
                nn.Linear(HIDDEN, latent_dim),
            )
            for _ in range(n_experts)
        ])
        self.router = nn.Linear(HIDDEN, n_experts)

    def forward(self, x):
        """回傳融合 latent (B,L) 與每個 patch 的 gate 權重 (B,E)。"""
        h = self.trunk(x)
        gates = torch.softmax(self.router(h) / self.temperature, dim=-1)
        expert_z = torch.stack([expert(h) for expert in self.experts], dim=1)
        z = (gates.unsqueeze(-1) * expert_z).sum(dim=1)
        return z, gates


def moe_balance_loss(*gate_batches):
    """讓 batch 平均 expert 使用率接近均勻，避免 router collapse。"""
    if not gate_batches:
        raise ValueError("至少需要一批 gate 權重")
    gates = torch.cat(gate_batches, dim=0)
    mean_usage = gates.mean(dim=0)
    target = torch.full_like(mean_usage, 1.0 / gates.shape[1])
    return (mean_usage - target).pow(2).mean()


class MLPAE(nn.Module):
    """四層 DDAE，其中 encoder 後段改成 dense Mixture of Experts。

    ``forward`` 與 ``encode`` 保留原本介面，讓既有分析程式可以繼續使用；
    訓練若需要 MoE gate，使用 ``forward_with_gates`` 或
    ``encode_with_gates``。
    """

    def __init__(self, latent_dim=2, n_experts=N_EXPERTS,
                 router_temperature=ROUTER_TEMPERATURE):
        super().__init__()
        self.encoder = MoEEncoder(latent_dim, n_experts, router_temperature)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, HIDDEN),
            nn.LayerNorm(HIDDEN),
            nn.GELU(),
            nn.Linear(HIDDEN, HIDDEN),
            nn.LayerNorm(HIDDEN),
            nn.GELU(),
            nn.Linear(HIDDEN, HIDDEN),
            nn.LayerNorm(HIDDEN),
            nn.GELU(),
            nn.Linear(HIDDEN, HIDDEN),
            nn.LayerNorm(HIDDEN),
            nn.GELU(),
            nn.Linear(HIDDEN, N_CAT),   # 輸出是 log λ，不接 norm
        )

    def encode(self, x):
        """x 是 (B,N_CAT) 的 count 向量，回傳 (B,latent_dim) 的 z——只跑 encoder。"""
        z, _ = self.encoder(x)
        return z

    def encode_with_gates(self, x):
        """回傳 z 與 router 對各 expert 的權重。"""
        return self.encoder(x)

    def forward(self, x):
        """x 是 (B,N_CAT) 的 count 向量（訓練時是加噪版、推論時是乾淨版），
        回傳 (z, log_lam)：z 是 (B,latent_dim) 的 latent，log_lam 是 (B,N_CAT)。
        """
        z, _ = self.encoder(x)
        return z, self.decoder(z)

    def forward_with_gates(self, x):
        """回傳 (z, log_lam, gates)，供 MoE 訓練與分析使用。"""
        z, gates = self.encoder(x)
        return z, self.decoder(z), gates


def poisson_nll(log_lam, x):
    """Poisson NLL，省略跟模型無關的 log(y!) 常數項，回傳每個 patch 一個數字。"""
    cell = torch.exp(log_lam) - x * log_lam
    return cell.mean(dim=1)


def poisson_deviance(log_lam, x):
    """Poisson deviance：2·(y·log(y/λ) - (y - λ))，越小越好、有下界，可跨 patch 比較。"""
    lam = torch.exp(log_lam)
    cell = 2 * (torch.xlogy(x, x) - x * log_lam - x + lam)
    return cell.mean(dim=1)


def build_fsce_graph(x, n_neighbors=15, metric="euclidean"):
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
