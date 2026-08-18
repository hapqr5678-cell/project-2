"""v0_weight_mse：跟 v0 同一個 in = out ConvAE，但 loss 把非零位置的權重提高。

架構、輸入、訓練流程都跟 v0 一模一樣（輸入 = 輸出 = (10, 40, 40)），
唯一的差別在 loss：

v0 的逐格 MSE 是 1600 x 10 = 16000 個位置一視同仁，但實測平均只有 0.6%
的位置非零（每個 patch 約 5.4% 的格子有 POI，再攤到 10 個類別 channel）。
也就是說「全部輸出 0」就能拿到很低的 loss，模型幾乎沒有動機去描述 POI 本身，
latent 很容易只剩下密度。

這一版給非零位置乘上 POS_WEIGHT 再取加權平均。POS_WEIGHT=1 時完全等同 v0；
非零位置佔總權重的比例是 0.0062w / (0.0062w + 0.9938)：
  w=1 -> 0.6%    w=10 -> 5.8%    w=50 -> 24%    w=100 -> 38%    w=161 -> 50%

跟 v1_masked 的差別：masked 版是把空白格整個丟掉（權重 0），模型不必學
「這裡應該是空的」；這一版空白格仍然算，只是相對變輕，所以「空白區域」
還是重建目標的一部分。

--- POS_WEIGHT 掃描結果（30 epoch、SEED=0，全部同一組切分）---
  w    val loss   R²密度   R²偏心   漏出倍數
  1     0.00329    0.934    0.186     1.0    (= v0)
  2     0.00608    0.934    0.122     1.9
  3     0.00855    0.930    0.277     2.7    <- 現在用這個
  5     0.01279    0.948    0.114     4.1
 10     0.02099    0.957    0.126     7.1
 20     0.03199    0.958    0.136    11.3
 50     0.04895    0.971    0.123    19.9
（漏出倍數 = 重建矩陣總強度 / 真實總強度，1.0 表示空白區有被壓乾淨）

結論跟原本的假設相反：加權**不會**把 latent 從密度手上搶回來，R²密度
反而隨 w 單調上升（0.934 -> 0.971），因為 loss 越集中在非零位置，
「這裡有幾個 POI」就越主導重建。同時漏出倍數幾乎跟 w 成正比。
w=3 是唯一一個兩邊都略優於 v0 的點（密度 R² 最低、偏心 R² 最高），
但只有單一 seed，這個差距不一定穩，別當成顯著結論。

已知風險：w 拉大時 loss 幾乎只剩非零位置，行為逼近 v1_masked，
重建矩陣到處都有殘值（w=50 時總強度是真實的 20 倍）。
用 analyze/mse.py 對照 zeros / mean baseline 時要記得那三個數字也是加權後的。
"""

import os
import sys

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.abspath(f"{os.path.dirname(__file__)}/../.."))
from config.dataset import CELL, GRID, N_CAT, HALF_WIDTH  # noqa: E402,F401

IN_CH = N_CAT
POS_WEIGHT = 3.0   # 非零位置的權重，1.0 = 退回 v0 的純 MSE


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
    """加權逐格 MSE，回傳每個 patch 一個數字。

    非零位置權重 POS_WEIGHT、零位置權重 1，最後除以權重總和（不是格數），
    所以疏密不同的 patch 之間仍可直接比較，POS_WEIGHT=1 時數值等同 v0。
    權重看的是輸入 x 而不是 recon，避免模型靠「把輸出調成 0」來閃掉權重。
    """
    w = torch.where(x > 0, POS_WEIGHT, 1.0)
    se = ((recon - x) ** 2 * w).sum(dim=(1, 2, 3))
    return se / w.sum(dim=(1, 2, 3))
