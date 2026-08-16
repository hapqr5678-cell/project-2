"""v2_tanh_perceiver：跟 v2_perceiver 完全同架構，唯一差異是 encoder 的
head 最後多接一個 Tanh()，把 latent 在訓練時就強制夾在 (-1,1) 之間。
跟 v2_tanh_ae 同一組實驗，用來看「訓練時就限制範圍」對 cross-attention
架構的影響是不是也一樣（讓 kNN R²/deviance 變差、離群訊號被壓縮）。
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
        owner, _, cat, b, _ = self._flatten(idx)
        flat = owner * N_CAT + cat
        counts = torch.bincount(flat, minlength=b * N_CAT)
        return counts.view(b, N_CAT).float()

    def tokens(self, idx):
        owner, within, cat, b, lens = self._flatten(idx)
        T = int(lens.max().item())
        tok = torch.zeros(b, T, dtype=torch.long)
        pad_mask = torch.ones(b, T, dtype=torch.bool)
        tok[owner, within] = cat
        pad_mask[owner, within] = False
        return tok, pad_mask


class CrossAttnBlock(nn.Module):
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
            nn.Linear(NUM_LATENTS * D_MODEL + 1, 256),
            nn.GELU(),
            nn.Linear(256, latent_dim),
            nn.Tanh(),   # <- 跟 v2_perceiver 唯一的差異：訓練時就把 latent 夾在 (-1,1)
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
    def __init__(self, latent_dim=2):
        super().__init__()
        self.encoder = PerceiverEncoder(latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, HIDDEN),
            nn.GELU(),
            nn.Linear(HIDDEN, HIDDEN),
            nn.GELU(),
            nn.Linear(HIDDEN, N_CAT),
        )

    def forward(self, tok, pad_mask):
        z = self.encoder(tok, pad_mask)
        return z, self.decoder(z)


def poisson_nll(log_lam, x):
    cell = torch.exp(log_lam) - x * log_lam
    return cell.mean(dim=1)


def poisson_deviance(log_lam, x):
    lam = torch.exp(log_lam)
    cell = 2 * (torch.xlogy(x, x) - x * log_lam - x + lam)
    return cell.mean(dim=1)
