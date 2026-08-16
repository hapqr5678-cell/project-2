"""v2_perceiver：encoder 換成 Perceiver 式 Cross-Attention，decoder/loss 跟 v2_ae 共用。

v2_ae 的 encoder 拿到的是「已經算好的」(N_CAT,) 聚合 count 向量——binning
這件事在 encoder 看到資料之前就做完了。這一版把聚合這件事交給模型自己：
encoder 直接吃整包 patch「還沒聚合的」原始 POI 列表，每個 POI 是一個
token（token 的內容只有它的類別，沒有座標——這個題目已經決定不用空間
資訊，兩個版本要能公平比較，Perceiver 就不能透過 token 多拿到位置這種
額外資訊），用一組固定數量的可學習 latent array 對這個集合做
cross-attention，讀完再互相 self-attention 整合，重複 NUM_BLOCKS 輪，
最後攤平投影到 2 維 latent。

跟 v2_ae 唯一的變因就是 encoder；decoder 沿用一模一樣的 MLP，
loss 也是同一組 poisson_nll / poisson_deviance（reconstruction target
一律是 agg() 算出的聚合向量），這樣兩邊的 latent 才能直接對照，看
「模型自己學聚合」有沒有比「先聚合好再吃」保留更多有用的資訊。

已知風險：
  * softmax attention 本質上是加權平均，天生不會「數數」——如果 100 個
    餐飲 token 跟 3 個餐飲 token 讓 latent array 的 attention 分布長得
    一樣，模型會分不出兩者的量級。這裡用一個側通道補救：把
    log1p(該 patch 的 token 數) 額外接到最後的投影層之前，讓模型至少
    能拿到「這包東西多大」的線索，細節見 PerceiverEncoder.forward()。
    這個設計呼應 v0_sizefactor 把 log(size factor) 額外接給 decoder
    的做法——都是「count 資訊天生不在架構的歸納偏誤裡，就外掛一個
    明確的量級輸入」。
  * batch 內 token 數不一（每個 patch 的 POI 數不同），用 padding +
    key_padding_mask 對齊；MIN_POI=10 保證每個 patch 至少有 1 個 token。
  * 同類別的多個 POI 在 embedding 空間裡完全相同（只有類別、沒有座標
    區分它們），所以「加 k 個某類 POI」不管加在哪裡結果都一樣——這是
    刻意的，跟 v2_ae 的聚合向量语义一致，不是 bug。
"""

import os
import sys

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.abspath(f"{os.path.dirname(__file__)}/../.."))
from config.dataset import N_CAT, HALF_WIDTH  # noqa: E402,F401

HIDDEN = 64
D_MODEL = 64
N_HEADS = 4
NUM_LATENTS = 16
NUM_BLOCKS = 2


class Patches:
    """稀疏點列表；agg() 聚合成 (N_CAT,) 向量當 loss target，
    tokens() 保留原始 POI 列表（只留類別）當 encoder 輸入。"""

    def __init__(self, path):
        d = np.load(path)
        self.cat = torch.from_numpy(d["cat"].astype(np.int64))
        self.offsets = torch.from_numpy(d["offsets"])
        self.n_poi = d["n_poi"]
        self.lat = d["center_lat"]
        self.lon = d["center_lon"]
        self.n = len(self.n_poi)

    def _flatten(self, idx):
        """回傳 (owner, within, cat)：owner=batch 內第幾個 patch，
        within=在該 patch 裡的第幾個點，cat=類別。"""
        b = len(idx)
        starts, ends = self.offsets[idx], self.offsets[idx + 1]
        lens = ends - starts
        pos = torch.repeat_interleave(starts, lens) + \
            torch.arange(int(lens.sum())) - torch.repeat_interleave(
                torch.cumsum(lens, 0) - lens, lens)
        owner = torch.repeat_interleave(torch.arange(b), lens)
        within = pos - starts[owner]
        return owner, within, self.cat[pos], b, lens

    def agg(self, idx):
        """idx 是 patch 編號的 tensor，回傳 (B,N_CAT) 的整包類別 count 向量。"""
        owner, _, cat, b, _ = self._flatten(idx)
        flat = owner * N_CAT + cat
        counts = torch.bincount(flat, minlength=b * N_CAT)
        return counts.view(b, N_CAT).float()

    def tokens(self, idx):
        """回傳 (B,T) 的類別 id（pad 用 0）與 pad_mask（True=padding）。"""
        owner, within, cat, b, lens = self._flatten(idx)
        T = int(lens.max().item())
        tok = torch.zeros(b, T, dtype=torch.long)
        pad_mask = torch.ones(b, T, dtype=torch.bool)
        tok[owner, within] = cat
        pad_mask[owner, within] = False
        return tok, pad_mask


class CrossAttnBlock(nn.Module):
    """讓 latent array 讀一次 token 集合：latents 當 query，tokens 當 key/value。"""

    def __init__(self, d_model, n_heads):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.ln_q = nn.LayerNorm(d_model)
        self.ln_kv = nn.LayerNorm(d_model)
        self.ln_ff = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 4), nn.GELU(), nn.Linear(d_model * 4, d_model))

    def forward(self, latents, tokens, pad_mask):
        q = self.ln_q(latents)
        kv = self.ln_kv(tokens)
        attn_out, _ = self.attn(q, kv, kv, key_padding_mask=pad_mask)
        latents = latents + attn_out
        latents = latents + self.ff(self.ln_ff(latents))
        return latents


class PerceiverEncoder(nn.Module):
    def __init__(self, latent_dim=2):
        super().__init__()
        self.cat_embed = nn.Embedding(N_CAT, D_MODEL)
        self.latents = nn.Parameter(torch.randn(NUM_LATENTS, D_MODEL) * 0.02)
        self.cross_blocks = nn.ModuleList(
            [CrossAttnBlock(D_MODEL, N_HEADS) for _ in range(NUM_BLOCKS)])
        self.self_blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(D_MODEL, N_HEADS, dim_feedforward=D_MODEL * 4,
                                        batch_first=True, norm_first=True)
            for _ in range(NUM_BLOCKS)])
        self.head = nn.Sequential(
            nn.Linear(NUM_LATENTS * D_MODEL + 1, 256),   # +1 = log1p(token 數) 側通道
            nn.GELU(),
            nn.Linear(256, latent_dim),   # latent 前不接激活，離群值才不會被壓回來
        )

    def forward(self, tok, pad_mask):
        tokens = self.cat_embed(tok)
        b = tok.shape[0]
        latents = self.latents.unsqueeze(0).expand(b, -1, -1)
        for cross, self_attn in zip(self.cross_blocks, self.self_blocks):
            latents = cross(latents, tokens, pad_mask)
            latents = self_attn(latents)
        n_valid = (~pad_mask).sum(dim=1, keepdim=True).float()
        h = torch.cat([latents.reshape(b, -1), torch.log1p(n_valid)], dim=-1)
        return self.head(h)


class PerceiverAE(nn.Module):
    """encoder 是 Perceiver Cross-Attention，decoder 跟 v2_ae 共用同一種 MLP。"""

    def __init__(self, latent_dim=2):
        super().__init__()
        self.encoder = PerceiverEncoder(latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, HIDDEN),
            nn.GELU(),
            nn.Linear(HIDDEN, HIDDEN),
            nn.GELU(),
            nn.Linear(HIDDEN, N_CAT),   # 輸出是 log λ
        )

    def forward(self, tok, pad_mask):
        z = self.encoder(tok, pad_mask)
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
