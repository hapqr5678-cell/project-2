"""跟 v2_ddae_fsce 的唯一差異：encoder/decoder 的每一層都加了 residual
shortcut（見 ResidualLinear）。維度相同的層（HIDDEN→HIDDEN）shortcut 是
identity；維度改變的層（N_CAT→HIDDEN、HIDDEN→latent_dim 等）shortcut 是
另一個不帶 bias 的 Linear，把輸入投影到輸出維度後再相加（ResNet 的
projection shortcut）。encoder 跟 decoder 的 residual 各自關在自己的
nn.Sequential 內部，兩者之間唯一的張量交換只有 forward() 裡的
z = encoder(x)，所以不會有 x 或 encoder 中間層繞過 z 直接漏給 decoder
的情況。

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


class ResidualLinear(nn.Module):
    """pre-norm 的 residual block：輸出 = shortcut(x) + act(linear(norm(x)))。
    LayerNorm 只作用在 residual branch 的入口，主幹（shortcut）維持一條沒有
    任何 normalize 的通路，梯度可以直接流回前面的層。in_dim/out_dim 相同時
    shortcut 是 identity；不同時 shortcut 是另一個不帶 bias 的 Linear，把 x
    投影到 out_dim 維後再相加（ResNet 的 projection shortcut）。

    in_dim/out_dim：輸入/輸出維度。activate：是否對 linear 的輸出套 GELU
    （latent、log_lam 這種輸出層要傳 False，維持沒有激活函數）。prenorm：
    是否在 residual branch 入口放 LayerNorm(in_dim)。encoder 第一層（輸入是
    raw count，normalize 掉會抹掉整包 patch 的總量資訊）跟 decoder 第一層
    （輸入是 2 維 latent，LayerNorm 會把它壓成 ±1）都要傳 False。
    """

    def __init__(self, in_dim, out_dim, activate=True, prenorm=True):
        super().__init__()
        self.norm = nn.LayerNorm(in_dim) if prenorm else nn.Identity()
        self.linear = nn.Linear(in_dim, out_dim)
        if in_dim == out_dim:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = nn.Linear(in_dim, out_dim, bias=False)
            nn.init.zeros_(self.shortcut.weight)
        self.act = nn.GELU() if activate else nn.Identity()

    def forward(self, x):
        """x 是 (B,in_dim)，回傳 (B,out_dim)。"""
        return self.shortcut(x) + self.act(self.linear(self.norm(x)))


class MLPAE(nn.Module):
    """跟 v2_ddae_fsce 的 MLPAE 差異只有每層都套了 ResidualLinear（見模組
    docstring）；層數、HIDDEN 維度、latent 前不接激活都沒變。
    denoising 完全發生在資料端（見 corrupt()），模型本身不需要知道。
    """

    def __init__(self, latent_dim=2):
        super().__init__()
        self.encoder = nn.Sequential(
            ResidualLinear(N_CAT, HIDDEN, prenorm=False),   # 輸入是 raw count，normalize 會抹掉 patch 的總量
            ResidualLinear(HIDDEN, HIDDEN),
            ResidualLinear(HIDDEN, HIDDEN),
            ResidualLinear(HIDDEN, HIDDEN),
            ResidualLinear(HIDDEN, latent_dim, activate=False),   # latent 前不接激活，離群值才不會被壓回來
        )
        self.decoder = nn.Sequential(
            ResidualLinear(latent_dim, HIDDEN, prenorm=False),   # 輸入是 2 維 latent，LayerNorm 會把它壓成 ±1
            ResidualLinear(HIDDEN, HIDDEN),
            ResidualLinear(HIDDEN, HIDDEN),
            ResidualLinear(HIDDEN, HIDDEN),
            ResidualLinear(HIDDEN, N_CAT, activate=False),   # 輸出是 log λ
        )

    def encode(self, x):
        """x 是 (B,N_CAT) 的 count 向量，回傳 (B,latent_dim) 的 z——只跑 encoder。"""
        return self.encoder(x)

    def forward(self, x):
        """x 是 (B,N_CAT) 的 count 向量（訓練時是加噪版、推論時是乾淨版），
        回傳 (z, log_lam)：z 是 (B,latent_dim) 的 latent，log_lam 是 (B,N_CAT)。
        """
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
