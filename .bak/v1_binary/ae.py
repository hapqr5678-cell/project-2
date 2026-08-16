"""v1_binary：把 v0 的 log1p(count) 換成 binary presence（0/1），loss 改用 BCE。

跟 v0 的對比：

  v0          (10, 40, 40)   10 channel = 10 類的 log1p(count)，loss = MSE
  v1_binary   (10, 40, 40)   10 channel = 10 類的 binary presence，loss = BCE

差別只有格值的語意：v0 保留密度（1 個 vs. 20 個 POI 長得不一樣），
v1_binary 只保留「有沒有」。

loss 用 BCE 而不是 MSE 的原因：
  target 是 0/1，MSE 的 decoder 輸出容易飄到 0.5 附近（梯度平衡點），
  BCE（等價於 logistic regression loss）才是 0/1 target 的正確 loss。
  decoder 最後一層輸出 logits（不加 sigmoid），由 BCEWithLogitsLoss 處理。

架構（CNN 層、latent 維度）跟 v0_l32 完全一樣，可以直接比較 latent 品質。
"""

import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.abspath(f"{os.path.dirname(__file__)}/../.."))
from config.dataset import CELL, GRID, N_CAT, HALF_WIDTH  # noqa: E402,F401

IN_CH = N_CAT


class Patches:
    """稀疏點列表；render() 才展開成稠密 binary 矩陣。"""

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
        """idx 是 patch 編號的 tensor，回傳 (B,10,40,40) 的 binary float 矩陣。

        值為 1.0 表示該類別在該格有至少一個 POI，0.0 表示沒有。
        跟 v0 相比只差最後從 log1p(counts) 改成 (counts > 0).float()。
        """
        b = len(idx)
        starts, ends = self.offsets[idx], self.offsets[idx + 1]
        lens = ends - starts
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

        return (counts > 0).float()   # ← 唯一差異：binary


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
            nn.Linear(256, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.GELU(),
            nn.Linear(256, 128 * s3 * s3),
            nn.Unflatten(1, (128, s3, s3)),
            block(128, 64, transpose=True),   # 5 -> 10
            block(64, 32, transpose=True),    # 10 -> 20
            nn.ConvTranspose2d(32, IN_CH, 4, 2, 1),   # 20 -> 40，輸出 logits
        )

    def forward(self, x):
        """回傳 (latent, logits)；logits 直接給 bce_loss，不加 sigmoid。"""
        z = self.encoder(x)
        logits = self.decoder(z)
        if logits.shape[-2:] != x.shape[-2:]:
            logits = F.interpolate(logits, size=x.shape[-2:],
                                   mode='bilinear', align_corners=False)
        return z, logits


# pos_weight：有 POI 的格子佔 ~3.3%，空格佔 ~96.7%，比例約 29:1。
# pos_weight 把「有 POI 但預測為 0」的 loss 放大，讓模型不偷懶全預測 0。
POS_WEIGHT = 30.0


def bce_loss(logits, x, pos_weight=POS_WEIGHT):
    """逐格 weighted BCE，回傳每個 patch 一個數字。

    pos_weight > 1 讓「有 POI 但預測為 0」的懲罰比「空格預測為 1」更重，
    解決 96.7% 空格的 class imbalance 問題。
    設成 None 則退回普通（unweighted）BCE。
    """
    pw = (torch.tensor(pos_weight, device=logits.device)
          if pos_weight is not None else None)
    return F.binary_cross_entropy_with_logits(
        logits, x, pos_weight=pw, reduction='none').mean(dim=(1, 2, 3))
