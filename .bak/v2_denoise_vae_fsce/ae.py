"""v2_denoise_vae_fsce：等於把 v2_denoise_vae（VAE + DAE 的加噪）再疊上
v2_vae_fsce 的 FSCE（fuzzy set cross entropy，UMAP 訓練用的那個 loss）。
模型架構跟 v2_denoise_vae 完全一樣，corrupt() 也原樣照抄，額外的東西只有
FSCE 那一組：build_fsce_graph() 建高維鄰接圖、fsce_loss() 算低維的
cross entropy，以及 VAE.encode() 讓 FSCE 可以只算到 mu。

FSCE 作用在 mu 上（不是取樣後的 z）——mu 是 eval 時唯一代表位置的座標，
取樣噪聲交給 logvar 負責，跟座標範圍/拓撲關係無關。

三種正則化各自管不同的事，這一版是要看它們疊起來會不會打架：
  * KL      管 latent 的形狀：把 q(z|x) 往連續、以原點為中心的先驗拉。
  * 加噪    管 latent 的內容：強迫模型從殘缺觀測猜完整分布，latent 不能
            只當個「密度計」，必須抓住跨類別的組成結構。
  * FSCE    管 latent 的拓撲：把高維空間的鄰居在低維也拉在一起、非鄰居
            推開，直接把 UMAP 的目標寫進 loss。
已知的張力是 KL 把 mu 往原點收、FSCE 的 negative sampling 把 mu 往外推，
兩者方向相反；train.py 的 FSCE warm-up 就是為了不讓它們一開始就對撞。

FSCE 的高維鄰接關係是用「整包 count 向量的 log1p、cosine 距離」kNN 建的
fuzzy simplicial set（跟 data/patch/umap_grid.py 的可行性檢查用同一種 metric），
只建一次、整個訓練共用同一張圖，而且是用**乾淨** count 建的——鄰接關係是
資料的性質，不是這次抽到的噪聲的性質。

破壞方式有兩種，用 NOISE_MODE 切換（意義同 v2_dae）：
  "thinning"  binomial thinning：每個 POI 以 1-NOISE_P 的機率被保留。
  "mask"      整個類別歸零：每一類以 NOISE_P 的機率整維被抹成 0。
兩種都會把破壞後的向量除以 1-NOISE_P 再送進 encoder（inverted scaling），
讓訓練/推論的輸入尺度一致，噪聲只留下變異、不留下偏移。

推論階段（model.eval()）不加噪、reparameterize() 直接回傳 mu，latent 是
決定性的、每個 patch 唯一，跟其他版本可以直接比較。
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
    """把乾淨的 count 向量破壞成 denoising 的輸入。

    x：(B,N_CAT) 的乾淨 count 向量（float，值是非負整數）。
    p：破壞強度 ∈[0,1)。thinning 是每個 POI 被丟掉的機率，mask 是每一類
       整維被抹成 0 的機率。p=0 時直接回傳 x（等價於退化成 v2_vae_fsce）。
    mode："thinning" 或 "mask"，意義見模組 docstring。
    generator：torch.Generator，給定就用它抽亂數，讓每個 epoch 的破壞可重現。

    回傳跟 x 同 shape 的 (B,N_CAT) tensor，已經除以 1-p 做過尺度補償，
    期望值等於 x（thinning 嚴格成立，mask 對每一維的期望也成立），
    所以可以直接餵進跟推論時同一個 encoder。
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


class VAE(nn.Module):
    """跟 v2_denoise_vae / v2_vae_fsce 同架構、同權重形狀（checkpoint 可互換）：
    denoising 完全發生在資料端（見 corrupt()），FSCE 只多用一個 encode()，
    模型本身兩件事都不需要知道。
    """

    def __init__(self, latent_dim=2):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(N_CAT, HIDDEN),
            nn.GELU(),
            nn.Linear(HIDDEN, HIDDEN),
            nn.GELU(),
        )
        self.fc_mu = nn.Linear(HIDDEN, latent_dim)
        self.fc_logvar = nn.Linear(HIDDEN, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, HIDDEN),
            nn.GELU(),
            nn.Linear(HIDDEN, HIDDEN),
            nn.GELU(),
            nn.Linear(HIDDEN, N_CAT),   # 輸出是 log λ
        )

    def encode(self, x):
        """x 是 (B,N_CAT) 的 count 向量，回傳 (B,latent_dim) 的 mu——跟 forward()
        算出來的 mu 是同一條路徑，只是不接 logvar/取樣/decoder。
        """
        h = self.trunk(x)
        return self.fc_mu(h)

    def reparameterize(self, mu, logvar):
        """mu、logvar 都是 (B,latent_dim)；訓練時回傳取樣的 z，
        eval 時直接回傳 mu，讓推論的 latent 唯一。
        """
        if not self.training:
            return mu
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    def forward(self, x):
        """x 是 (B,N_CAT) 的 count 向量（訓練時是加噪版、推論時是乾淨版），
        回傳 (mu, logvar, z, log_lam)：前三個是 (B,latent_dim)，
        log_lam 是 (B,N_CAT) 的 log λ。
        """
        h = self.trunk(x)
        mu = self.fc_mu(h)
        # logvar 沒有任何界限的話，訓練初期數值不穩時 exp(0.5*logvar) 可能
        # 溢位、整個 forward 變成 inf/nan（跟 FSCE 無關，是 fc_logvar 本身
        # 沒有界限的既有風險），clamp 純粹是數值穩定用，不影響 mu 這個
        # FSCE 真正在乎的座標。
        logvar = self.fc_logvar(h).clamp(-10.0, 10.0)
        z = self.reparameterize(mu, logvar)
        return mu, logvar, z, self.decoder(z)


def poisson_nll(log_lam, x):
    """Poisson NLL，省略跟模型無關的 log(y!) 常數項，回傳每個 patch 一個數字。"""
    cell = torch.exp(log_lam) - x * log_lam
    return cell.mean(dim=1)


def poisson_deviance(log_lam, x):
    """Poisson deviance：2·(y·log(y/λ) - (y - λ))，越小越好、有下界，可跨 patch 比較。"""
    lam = torch.exp(log_lam)
    cell = 2 * (torch.xlogy(x, x) - x * log_lam - x + lam)
    return cell.mean(dim=1)


def kl_divergence(mu, logvar):
    """KL(N(mu, diag(exp(logvar))) || N(0,I))，回傳每個 patch 一個數字。"""
    return -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=1)


def build_fsce_graph(x, n_neighbors=15, metric="cosine"):
    """x 是高維空間的特徵矩陣 (N,D)（這裡傳 log1p(乾淨 count)），在上面建一次
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
    """FSCE（fuzzy set cross entropy）：z_i、z_j 是一批 pair 的 mu 座標
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
