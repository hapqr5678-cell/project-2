"""v3_multinomial vs v2 全系列（ae/vae/perceiver）：multinomial 分解到底把什麼從 latent 移出去了。

四個版本必須跑在同一份 patches.npz 上，這支腳本才有意義
（換 DATASET 之後全部都要重跑 train.py，否則會拿舊資料集的 ae.pt 比）。

看三件事，重要性由高到低：

1. latent 對 log(POI 數) 的 kNN R² —— 這是這一版存在的理由。
   v2 的 latent 有一半在編碼密度，v3 若成功，這個數字要明顯掉下來。

2. deviance —— **不能直接比高下**。v3 的 decoder 拿到 log n 當 offset，
   總量是白給的，deviance 本來就佔便宜；反過來 v3 的 encoder 看不到
   總量，又吃虧。列出來只是確認 v3 沒有崩掉，不是成績單。

3. 噪聲下限 —— 拿模型自己的 λ 重抽 Poisson 樣本再算 deviance。
   真實 deviance 若已經低於這個下限，代表模型的重建誤差主要來自
   不可約的抽樣噪聲，繼續壓 deviance 就是在擬合噪聲。
   patch 的 POI 數越少這條線越高，是判斷「還原不出來是不是資料的錯」
   的唯一客觀依據。
"""

import importlib
import os
import sys

import numpy as np
import torch
from scipy.spatial import cKDTree

ROOT = os.path.abspath(f"{os.path.dirname(__file__)}/../../..")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "model", "v3_multinomial"))
from ae import MultinomialAE, Patches, poisson_deviance  # noqa: E402
from config.dataset import CAT_ZH, N_CAT, PATCHES, result  # noqa: E402

KNN = 50
N_RESAMPLE = 20     # 估噪聲下限時重抽幾次
SEED = 0


def knn_r2(z, y):
    """用 latent 的 k 個鄰居預測 y，回傳 R²（排除自己）。"""
    zs = (z - z.mean(0)) / z.std(0)
    _, idx = cKDTree(zs).query(zs, k=KNN + 1)
    pred = y[idx[:, 1:]].mean(1)
    return 1 - np.var(y - pred) / np.var(y)


def noise_floor(lam, log_lam, rng):
    """把模型的 λ 當真值重抽 Poisson，回傳合成資料的 mean deviance。"""
    ds = []
    for _ in range(N_RESAMPLE):
        xs = torch.from_numpy(rng.poisson(lam).astype(np.float32))
        ds.append(poisson_deviance(log_lam, xs).numpy().mean())
    return float(np.mean(ds)), float(np.std(ds))


def _load_module(version, filename="ae.py"):
    """v2_ae/v2_vae/v2_perceiver 的類別跟 v3 同名不同義，獨立 import 避免撞名。"""
    path = os.path.join(ROOT, "model", version, filename)
    spec = importlib.util.spec_from_file_location(f"{version}_ae", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_v2_ae(X, tok, pad_mask):
    mod = _load_module("v2_ae")
    m = mod.MLPAE(2)
    m.load_state_dict(torch.load(result("v2_ae", "ae.pt"), map_location="cpu"))
    m.eval()
    with torch.no_grad():
        z, log_lam = m(X)
    return z.numpy(), log_lam


def load_v2_vae(X, tok, pad_mask):
    mod = _load_module("v2_vae")
    m = mod.VAE(2)
    m.load_state_dict(torch.load(result("v2_vae", "ae.pt"), map_location="cpu"))
    m.eval()
    with torch.no_grad():
        mu, logvar, z, log_lam = m(X)
    return mu.numpy(), log_lam


def load_v2_perceiver(X, tok, pad_mask):
    mod = _load_module("v2_perceiver")
    m = mod.PerceiverAE(2)
    m.load_state_dict(torch.load(result("v2_perceiver", "ae.pt"), map_location="cpu"))
    m.eval()
    with torch.no_grad():
        z, log_lam = m(tok, pad_mask)
    return z.numpy(), log_lam


LOADERS = {
    "v2_ae（raw count，latent 得自己扛總量）": load_v2_ae,
    "v2_vae（同 v2_ae + KL 正規化）": load_v2_vae,
    "v2_perceiver（cross-attention 自己聚合）": load_v2_perceiver,
}


def report(name, X, z, log_lam, log_n, rng):
    lam = torch.exp(log_lam).numpy()
    dev = poisson_deviance(log_lam, X).numpy()
    floor, floor_sd = noise_floor(lam, log_lam, rng)

    print(f"\n=== {name} ===")
    print(f"  mean deviance      {dev.mean():.4f}")
    print(f"  噪聲下限           {floor:.4f} ± {floor_sd:.4f}"
          f"   ({'已低於下限，開始擬合噪聲' if dev.mean() < floor else '仍在下限之上'})")
    print(f"  latent -> log(POI 數) kNN R² = {knn_r2(z, log_n):+.3f}   <- 越低越好")
    for d in range(z.shape[1]):
        print(f"    z[{d}] 對 log(POI 數) corr = "
              f"{np.corrcoef(z[:, d], log_n)[0, 1]:+.3f}")
    return lam, dev


def main():
    rng = np.random.default_rng(SEED)
    data = Patches(PATCHES)
    Xn = data.agg(torch.arange(data.n)).numpy()
    X = torch.from_numpy(Xn).float()
    log_n = np.log(Xn.sum(1))

    # v2_perceiver 需要未聚合的 token 列表；用它自己的 Patches.tokens() 較保險，
    # 但欄位跟 v3 的 Patches 相容，直接沿用同一個 data 物件即可。
    tok, pad_mask = None, None
    perceiver_mod = _load_module("v2_perceiver")
    tok_data = perceiver_mod.Patches(PATCHES)
    tok, pad_mask = tok_data.tokens(torch.arange(tok_data.n))

    print(f"{data.n} 個 patch，N_CAT={N_CAT}，"
          f"POI/patch 中位數 {np.median(Xn.sum(1)):.0f}")

    m3 = MultinomialAE(2)
    m3.load_state_dict(torch.load(result("v3_multinomial", "ae.pt"),
                                  map_location="cpu"))
    m3.eval()
    with torch.no_grad():
        z3, ll3 = m3(X)

    results = {}
    for name, loader in LOADERS.items():
        z, ll = loader(X, tok, pad_mask)
        lam, dev = report(name, X, z, ll, log_n, rng)
        results[name] = (lam, dev)

    lam3, dev3 = report("v3_multinomial（組成比例 + log n offset）",
                         X, z3.numpy(), ll3, log_n, rng)

    print("\n=== 逐類別重建 R²（真實 count vs 重建 λ，v3 vs 各 v2 變體）===")
    header = f"  {'類別':<10}{'真mean':>8}" + \
        "".join(f"{n.split('（')[0]+' R²':>14}" for n in LOADERS) + f"{'v3 R²':>14}"
    print(header)
    for c in range(N_CAT):
        x_ = Xn[:, c]
        ss = ((x_ - x_.mean()) ** 2).sum()
        row = f"  {CAT_ZH[c]:<10}{x_.mean():8.2f}"
        for name in LOADERS:
            lam, _ = results[name]
            r2 = 1 - ((x_ - lam[:, c]) ** 2).sum() / ss
            row += f"{r2:14.3f}"
        r2_3 = 1 - ((x_ - lam3[:, c]) ** 2).sum() / ss
        row += f"{r2_3:14.3f}"
        print(row)


main()
