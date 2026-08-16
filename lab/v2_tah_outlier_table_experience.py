"""跟 v2_tah_all_outlier_experience.py 同一組實驗（全體 patch、同一個 CATS/
OFFSETS 設定），但不畫圖，改整理成表格：對每個類別加入 k 個 POI 後，
latent 平均移動了多少（不穩定性）、重建損失（poisson deviance）比加之前
平均提升了多少。

三個 v2_tanh 版本各自一張表，印在 console 也存成 csv。
"""

import importlib.util
import os
import sys

import numpy as np
import pandas as pd
import torch

ROOT = os.path.abspath(f"{os.path.dirname(__file__)}/..")
sys.path.insert(0, ROOT)
from config.dataset import CAT_ZH, N_CAT, PATCHES, result  # noqa: E402

VERSIONS = ["v2_ae_tanh", "v2_vae_tanh", "v2_tanh_perceiver"]
CATS = ["餐飲", "零售", "商業服務", "藝文娛樂"]   # 要跑哪幾個類別，改這個清單即可
OFFSETS = [1, 2, 3]      # 每個類別各自加入的點數
BATCH = 512
OUT_DIR = os.path.join(ROOT, "lab")


def load_ae(version):
    """把某個版本的 ae.py 用獨立模組名載入，避免三個版本的 `ae` 互相覆蓋 sys.modules。"""
    path = os.path.join(ROOT, "model", version, "ae.py")
    spec = importlib.util.spec_from_file_location(f"ae_{version}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def model_class(mod, version):
    if "perceiver" in version:
        return mod.PerceiverAE
    if "vae" in version:
        return mod.VAE
    return mod.MLPAE


def all_base_counts(p):
    """(N_patch, N_CAT) 的原始類別 count 向量。"""
    n = len(p["n_poi"])
    cat = p["cat"].astype(np.int64)
    owner = np.repeat(np.arange(n), np.diff(p["offsets"]))
    counts = np.zeros((n, N_CAT), dtype=np.float32)
    np.add.at(counts, (owner, cat), 1.0)
    return counts


def all_base_cat_lists(p):
    """每個 patch 的原始類別 id 陣列，perceiver 用（token 列表）。"""
    cat = p["cat"].astype(np.int64)
    offsets = p["offsets"]
    return [cat[offsets[i]:offsets[i + 1]] for i in range(len(p["n_poi"]))]


def batch_encode(model, mod, version, base_counts_all, base_cat_lists, cat_id, k):
    """回傳加了 k 個 cat_id 後全體 patch 的 (z (N,2), deviance (N,))。"""
    zs, devs = [], []
    is_perceiver = "perceiver" in version
    with torch.no_grad():
        if is_perceiver:
            cat_lists_k = [np.concatenate([b, np.full(k, cat_id, dtype=np.int64)])
                          for b in base_cat_lists]
            for i in range(0, len(cat_lists_k), BATCH):
                chunk = cat_lists_k[i:i + BATCH]
                b, T = len(chunk), max(len(c) for c in chunk)
                tok = torch.zeros(b, T, dtype=torch.long)
                pad_mask = torch.ones(b, T, dtype=torch.bool)
                for j, c in enumerate(chunk):
                    tok[j, :len(c)] = torch.from_numpy(c)
                    pad_mask[j, :len(c)] = False
                z, log_lam = model(tok, pad_mask)
                counts = np.stack([np.bincount(c, minlength=N_CAT) for c in chunk]
                                  ).astype(np.float32)
                x = torch.from_numpy(counts)
                devs.append(mod.poisson_deviance(log_lam, x).numpy())
                zs.append(z.numpy())
        else:
            counts_k = base_counts_all.copy()
            counts_k[:, cat_id] += k
            for i in range(0, len(counts_k), BATCH):
                x = torch.from_numpy(counts_k[i:i + BATCH])
                out = model(x)
                z = out[2] if "vae" in version else out[0]
                log_lam = out[-1]
                devs.append(mod.poisson_deviance(log_lam, x).numpy())
                zs.append(z.numpy())
    return np.concatenate(zs, axis=0), np.concatenate(devs, axis=0)


def run_version(version, base_counts_all, base_cat_lists, cat_ids):
    mod = load_ae(version)
    d = np.load(result(version, "latents.npz"))
    z_all, base_dev = d["z"], d["err"]

    model = model_class(mod, version)(2)
    model.load_state_dict(torch.load(result(version, "ae.pt"), map_location="cpu"))
    model.eval()

    rows = []
    for c in cat_ids:
        for k in OFFSETS:
            z_new, dev_new = batch_encode(model, mod, version, base_counts_all,
                                          base_cat_lists, c, k)
            dist = np.linalg.norm(z_new - z_all, axis=1)
            dev_inc = dev_new - base_dev
            rows.append(dict(類別=CAT_ZH[c], 加入點數=k,
                             移動距離_不穩定性=dist.mean(),
                             重建損失提升=dev_inc.mean()))
    return pd.DataFrame(rows)


def main():
    cat_ids = [CAT_ZH.index(c) for c in CATS]
    p = np.load(PATCHES)
    base_counts_all = all_base_counts(p)
    base_cat_lists = all_base_cat_lists(p)

    for v in VERSIONS:
        df = run_version(v, base_counts_all, base_cat_lists, cat_ids)
        print(f"\n=== {v}（{len(base_cat_lists)} 個 patch 的平均）===")
        print(df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
        out = os.path.join(OUT_DIR, f"v2_tah_outlier_table_experience_{v}.csv")
        df.to_csv(out, index=False, float_format="%.4f")
        print(f"已存 {out}")


main()
