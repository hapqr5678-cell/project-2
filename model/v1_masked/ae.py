"""v1_masked：跟 v1_embeding 完全一樣，只差 loss 不算空白格。

  v0            (10, 40, 40)   10 channel = 10 類的 log1p(count)，1600 格全算 loss
  v1_embeding   ( 8, 40, 40)   每格一個類別 embedding，1600 格全算 loss
  v1_masked     ( 8, 40, 40)   同上，但 loss 只算有 POI 的那幾格

v1_embeding 的 loss 有 96.7% 在講「這裡是空的」，類別訊號被稀釋掉；
這一版把空白格排掉，只問「有東西的格子，類別對不對」。

先講清楚可預期的風險：embedding 是可學習的，一旦所有類別轉成同一個向量 u，
被算 loss 的格子上 target 就變成常數 u，decoder 整張輸出 u 即可讓 loss 精確等於 0。
v1_embeding 至少還逼模型交代「哪些格是空的」，這一版連那個約束都拿掉了，
所以塌縮的動機比 v1_embeding 更強而不是更弱。train.py 印的兩兩 cosine
與這裡的 loss 絕對值（是否掉到接近 0）就是驗證這件事的兩個數字。

一格可能落到多個 POI，這一版的規則是「隨機挑一個代表」，而且是在載入 patch
時用固定 seed 抽一次就定案（見 Patches），所以每個 patch 的輸入矩陣是唯一的、
latent 也是決定性的。代價是完全不保留密度：一格有 1 個還是 20 個 POI 長得一樣。
也因為抽樣結果綁死在固定的 binning 上，這一版沒有隨機旋轉。

embedding 是可學習的 nn.Embedding，跟 AE 一起訓練。注意 in = out 的設定下
「輸入本身」也是模型的一部分，模型可以靠把 embedding 縮小來作弊降 loss，
所以查表後一律做 L2 normalize，把每個類別鎖在單位球面上：模型只能轉方向、
不能縮長度。（單位長度擋得住尺度塌縮，但擋不住所有類別轉到同一個方向，
train.py 最後會印出兩兩 cosine 讓這件事看得見。）
"""

import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.abspath(f"{os.path.dirname(__file__)}/../.."))
from config.dataset import CELL, GRID, N_CAT, HALF_WIDTH  # noqa: E402,F401

EMB_DIM = 8
PICK_SEED = 0     # 重疊格挑代表用的 seed，換了整份輸入就變了


class Patches:
    """稀疏點列表；每格先隨機留一個代表，render() 才展開成類別編號矩陣。"""

    def __init__(self, path):
        d = np.load(path)
        dx, dy = d["dx"], d["dy"]
        cat = d["cat"].astype(np.int64)
        offsets = d["offsets"]

        self.n_poi = d["n_poi"]
        self.lat = d["center_lat"]
        self.lon = d["center_lon"]
        self.n = len(self.n_poi)

        # 每個點落在哪個 patch 的哪一格
        owner = np.repeat(np.arange(self.n), np.diff(offsets))
        ix = np.clip(np.floor(dx / CELL + GRID / 2), 0, GRID - 1).astype(np.int64)
        iy = np.clip(np.floor(dy / CELL + GRID / 2), 0, GRID - 1).astype(np.int64)
        cell = (iy * GRID + ix)
        key = owner * (GRID * GRID) + cell

        # 每個 (patch, 格) 只留一個點：給每個點一個隨機優先度，同格取最大的
        pri = np.random.default_rng(PICK_SEED).random(len(key))
        order = np.lexsort((pri, key))
        k = key[order]
        keep = order[np.append(k[1:] != k[:-1], True)]   # 每組的最後一個 = 優先度最大

        self.cell = torch.from_numpy(cell[keep])
        self.cat = torch.from_numpy(cat[keep])
        # keep 依 key 排序，本來就照 patch 分好組了，直接數數量就是新的 offsets
        n_occ = np.bincount(owner[keep], minlength=self.n)
        self.n_occupied = n_occ
        self.offsets = torch.from_numpy(
            np.concatenate([[0], np.cumsum(n_occ)]).astype(np.int64))

    def render(self, idx):
        """idx 是 patch 編號的 tensor，回傳 (B,40,40) 的類別編號矩陣。

        值 0 代表空白格，類別 c 存成 c+1（embedding 查表時第 0 列是零向量）。
        """
        b = len(idx)
        starts, ends = self.offsets[idx], self.offsets[idx + 1]
        lens = ends - starts
        # 把每個 patch 留下的點攤平成一條，並記住它屬於 batch 裡的第幾個
        pos = torch.repeat_interleave(starts, lens) + \
            torch.arange(int(lens.sum())) - torch.repeat_interleave(
                torch.cumsum(lens, 0) - lens, lens)
        owner = torch.repeat_interleave(torch.arange(b), lens)

        g = torch.zeros(b, GRID * GRID, dtype=torch.long)
        g[owner, self.cell[pos]] = self.cat[pos] + 1
        return g.view(b, GRID, GRID)


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
        self.emb = nn.Embedding(N_CAT, EMB_DIM)
        self.encoder = nn.Sequential(
            block(EMB_DIM, 32),   # 40 -> 20
            block(32, 64),        # 20 -> 10
            block(64, 128),       # 10 -> 5
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
            nn.ConvTranspose2d(32, EMB_DIM, 4, 2, 1),   # 20 -> 40，跟輸入同形狀
        )

    def table(self):
        """(N_CAT+1, EMB_DIM) 的查表：第 0 列是空白格的零向量，其餘單位長度。"""
        w = F.normalize(self.emb.weight, dim=1)
        return torch.cat([w.new_zeros(1, EMB_DIM), w])

    def embed(self, g):
        """(B,40,40) 類別編號 -> (B,EMB_DIM,40,40) 的輸入矩陣。"""
        return self.table()[g].permute(0, 3, 1, 2)

    def forward(self, g):
        """回傳 (輸入矩陣, latent, 重建)；輸入也要拿出來才能算 loss。"""
        x = self.embed(g)
        z = self.encoder(x)
        return x, z, self.decoder(z)


def mse_loss(recon, x, g):
    """只算有 POI 的格子的 MSE，回傳每個 patch 一個數字。

    v1_embeding 是 1600 格全算，其中約 96.7% 是「這裡是空的」；
    這一版把空白格整個排掉，loss 只問「有東西的那幾格，類別對不對」。
    分母是該 patch 的佔用格數 x EMB_DIM，所以不同疏密的 patch 可以直接比。
    """
    m = (g > 0).unsqueeze(1)                     # (B,1,40,40)
    se = ((recon - x) ** 2).sum(dim=(1, 2, 3))
    return se / (m.sum(dim=(1, 2, 3)) * EMB_DIM)
