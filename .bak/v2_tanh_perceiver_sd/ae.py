"""v2_tanh_perceiver_agg_pure：跟 v2_tanh_perceiver_agg 同樣吃聚合好的
(N_CAT,) count 向量，但 latent array 本身直接就是 AE 的 latent space
（維度 = latent_dim=2），不是另外養一個高維（NUM_LATENTS x D_MODEL）
的 latent array、跑完 attention 後再用一層 Linear 投影/退化到 2 維。

也就是：cross-attention 的 query 端（latents）、self-attention 全程都
在 latent_dim（2）這個維度裡跑；只有 byte array（類別 token，key/value
那一側）維持 D_MODEL（64）的高維表示，因為它要能承載 N_CAT 個類別各自
的語意 + count 資訊，跟 latent array 的維度是兩件事，不需要一樣大。
MultiheadAttention 用 kdim/vdim 讓 query 端（latent_dim）跟 key/value 端
（D_MODEL）維度不同但仍可互相 attend。

head 只剩 mean pool（NUM_LATENTS 個 latent 平均成一個 latent_dim 向量，
對照 Perceiver 論文最後的 Average）+ BatchNorm1d(affine=False)（防塌縮，
理由跟 v2_tanh_perceiver_agg 一樣：poisson NLL 初期梯度方向一致會把
Tanh 前的輸出焊死成同一點）+ Tanh。中間沒有任何會改變維度的 Linear——
latent array 每一格的數值就是最終拿去畫圖、算離群值的 z 分量本身。

已知風險：latent_dim=2 太窄，attention 的 K/Q/V 只能用 1 個 head、每個
head 的表示能力非常有限，訓練可能比 v2_tanh_perceiver_agg_pure 舊版
（高維 latent array + Linear 投影）更難收斂/deviance 更差，這裡先如實
測試、不預先加補償機制。
"""

import os
import sys

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.abspath(f"{os.path.dirname(__file__)}/../.."))
from config.dataset import N_CAT, HALF_WIDTH  # noqa: E402,F401

HIDDEN = 64
D_MODEL = 64          # byte array（類別 token）的維度
LATENT_HEADS = 1      # latent_dim=2 太小，attention 只能開 1 個 head
NUM_LATENTS = 16
NUM_BLOCKS = 2
FF_HIDDEN = 64        # latent 側 FF 的隱藏維度，跟 latent_dim 脫鉤，避免 2*4=8 太窄


class Patches:
    """稀疏點列表；agg() 把整個 patch 聚合成一個 (N_CAT,) 的 count 向量，
    這裡直接拿來當 encoder 輸入（同時也是 loss target）。"""

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


class CrossAttnBlock(nn.Module):
    """latents（query，latent_dim 維）讀一次「N_CAT 個類別 token」
    （key/value，D_MODEL 維）。用 kdim/vdim 讓兩側維度不同但仍可 attend。"""

    def __init__(self, latent_dim, kv_dim, n_heads):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            latent_dim, n_heads, kdim=kv_dim, vdim=kv_dim, batch_first=True)
        self.ln_q = nn.LayerNorm(latent_dim)
        self.ln_kv = nn.LayerNorm(kv_dim)
        self.ln_ff = nn.LayerNorm(latent_dim)
        self.ff = nn.Sequential(
            nn.Linear(latent_dim, FF_HIDDEN), nn.GELU(), nn.Linear(FF_HIDDEN, latent_dim))

    def forward(self, latents, tokens):
        q = self.ln_q(latents)
        kv = self.ln_kv(tokens)
        attn_out, _ = self.attn(q, kv, kv)
        latents = latents + attn_out
        latents = latents + self.ff(self.ln_ff(latents))
        return latents


class PerceiverAggEncoder(nn.Module):
    def __init__(self, latent_dim=2):
        super().__init__()
        self.cat_embed = nn.Embedding(N_CAT, D_MODEL)
        self.count_proj = nn.Linear(1, D_MODEL)   # count 是每個 token 明確帶的特徵
        # latent array 本身就是 AE 的 latent space：每一格是 latent_dim 維，
        # 不是另外養一個高維陣列再投影下來。
        self.latents = nn.Parameter(torch.randn(NUM_LATENTS, latent_dim) * 0.02)
        self.cross_blocks = nn.ModuleList(
            [CrossAttnBlock(latent_dim, D_MODEL, LATENT_HEADS) for _ in range(NUM_BLOCKS)])
        self.self_blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(latent_dim, LATENT_HEADS, dim_feedforward=FF_HIDDEN,
                                        batch_first=True, norm_first=True)
            for _ in range(NUM_BLOCKS)])
        # affine=False：只借 BatchNorm「減掉跨樣本共同方向」這個結構性質防塌縮，
        # 不給它可學習的 shift/scale，也不接任何會改變維度的 Linear。
        self.norm = nn.BatchNorm1d(latent_dim, affine=False)
        self.tanh = nn.Tanh()

    def forward(self, x):
        """x: (B,N_CAT) count 向量。"""
        b = x.shape[0]
        cat_ids = torch.arange(N_CAT, device=x.device)
        log_x = torch.log1p(x)
        tokens = self.cat_embed(cat_ids).unsqueeze(0).expand(b, -1, -1) \
            + self.count_proj(log_x.unsqueeze(-1))
        latents = self.latents.unsqueeze(0).expand(b, -1, -1)
        for cross, self_attn in zip(self.cross_blocks, self.self_blocks):
            latents = cross(latents, tokens)
            latents = self_attn(latents)
        z = latents.mean(dim=1)   # (NUM_LATENTS, latent_dim) -> (latent_dim,)，沒有 Linear
        return self.tanh(self.norm(z))


class PerceiverAggAE(nn.Module):
    """encoder 是吃聚合向量的 Perceiver cross-attention，z 就是 latent array
    mean pool 後的原始數值；decoder 跟 v2 系列共用同一種 MLP。"""

    def __init__(self, latent_dim=2):
        super().__init__()
        self.encoder = PerceiverAggEncoder(latent_dim)
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
