"""v0：最單純的 in = out 卷積 AutoEncoder，loss = MSE。

這一版刻意不做任何統計上的設計，就是教科書式的 autoencoder：
  輸入 = 輸出 = (10, 40, 40)，10 channel = 10 類的 log1p(count)
  loss = 逐格 MSE

沒有圓形遮罩、沒有 Poisson/multinomial likelihood，
目的是先跑通整個 pipeline（render -> encode -> latent 2 維 -> decode），
並當成後面版本的對照組：看「單純重建」的 latent 會被什麼主導。
"""

import os
import sys

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.abspath(f"{os.path.dirname(__file__)}/../.."))
from config.dataset import CELL, GRID, N_CAT, HALF_WIDTH  # noqa: E402,F401

IN_CH = N_CAT


class Patches:
    """稀疏點列表；render() 才展開成稠密矩陣。"""

    def __init__(self, path):
        d = np.load(path)
        self.dx = torch.from_numpy(d["dx"])
        self.dy = torch.from_numpy(d["dy"])
        self.cat = torch.from_numpy(d["cat"].astype(np.int64))
        self.offsets = torch.from_numpy(d["offsets"])
        self.n_poi = d["n_poi"]
        self.lat = d["center_lat"]
        self.lon = d["center_lon"]
        self.n = len(self.n_poi)

    def render(self, idx, rotate):
        """idx 是 patch 編號的 tensor，回傳 (B,10,40,40) 的輸入矩陣。"""
        b = len(idx)
        starts, ends = self.offsets[idx], self.offsets[idx + 1]
        lens = ends - starts
        # 把每個 patch 的點攤平成一條，並記住它屬於 batch 裡的第幾個
        pos = torch.repeat_interleave(starts, lens) + \
            torch.arange(int(lens.sum())) - torch.repeat_interleave(
                torch.cumsum(lens, 0) - lens, lens)
        owner = torch.repeat_interleave(torch.arange(b), lens)

        dx, dy, cat = self.dx[pos], self.dy[pos], self.cat[pos]
        if rotate:
            theta = torch.rand(b) * (2 * torch.pi)
            cos, sin = torch.cos(theta)[owner], torch.sin(theta)[owner]
            dx, dy = dx * cos - dy * sin, dx * sin + dy * cos
            flip = (torch.rand(b) < 0.5)[owner]
            dx = torch.where(flip, -dx, dx)

        ix = (dx / CELL + GRID / 2).floor().long().clamp(0, GRID - 1)
        iy = (dy / CELL + GRID / 2).floor().long().clamp(0, GRID - 1)

        flat = ((owner * N_CAT + cat) * GRID + iy) * GRID + ix
        counts = torch.bincount(flat, minlength=b * N_CAT * GRID * GRID)
        counts = counts.view(b, N_CAT, GRID, GRID).float()

        return torch.log1p(counts)


def block(cin, cout, transpose=False):
    conv = (nn.ConvTranspose2d(cin, cout, 4, 2, 1) if transpose
            else nn.Conv2d(cin, cout, 3, 2, 1))
    return nn.Sequential(conv, nn.GroupNorm(8, cout), nn.GELU())


class ConvAE(nn.Module):
    def __init__(self, latent_dim=2):
        super().__init__()
        s1 = (GRID + 2*1 - 3) // 2 + 1
        s2 = (s1 + 2*1 - 3) // 2 + 1
        s3 = (s2 + 2*1 - 3) // 2 + 1
        self.enc_size = s3
        self.encoder = nn.Sequential(
            block(IN_CH, 32),   # 40 -> 20
            block(32, 64),      # 20 -> 10
            block(64, 128),     # 10 -> 5
            nn.Flatten(),
            nn.Linear(128 * s3 * s3, 256),
            nn.GELU(),
            nn.Linear(256, latent_dim),   # latent 前不接激活，離群值才不會被壓回來
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.GELU(),
            nn.Linear(256, 128 * s3 * s3),
            nn.Unflatten(1, (128, s3, s3)),
            block(128, 64, transpose=True),   # 5 -> 10
            block(64, 32, transpose=True),    # 10 -> 20
            nn.ConvTranspose2d(32, IN_CH, 4, 2, 1),   # 20 -> 40，輸出跟輸入同形狀
        )

    def forward(self, x):
        z = self.encoder(x)
        out = self.decoder(z)
        if out.shape[-1] != x.shape[-1] or out.shape[-2] != x.shape[-2]:
            import torch.nn.functional as F
            out = F.interpolate(out, size=(x.shape[-2], x.shape[-1]), mode='bilinear', align_corners=False)
        return z, out


def mse_loss(recon, x):
    """逐格 MSE，回傳每個 patch 一個數字。"""
    return ((recon - x) ** 2).mean(dim=(1, 2, 3))
