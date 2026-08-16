"""不挑單一個案，改看「全體 patch」：對 v2_ae_tanh / v2_vae_tanh /
v2_tanh_perceiver 三個版本，每個都對全部 patch 同時做同一件事——某個
類別加入 1 / 10 / 30 個——然後把加完之後的新位置畫成 heatmap 風格的
散點圖：背景是全體 patch 原本的 latent 位置（半透明灰點），前景是加了
POI 之後的新位置，顏色深淺代表這個 patch 移動了多遠（離它自己原本位置
的距離），移動越遠顏色數值越高。

一個版本一張圖，N_CAT 個類別 x 3 個 offset = 30 個子圖。
"""

import importlib.util
import os
import sys

import numpy as np
import torch
import matplotlib as mpl
import matplotlib.pyplot as plt

ROOT = os.path.abspath(f"{os.path.dirname(__file__)}/..")
sys.path.insert(0, ROOT)
from config.dataset import CAT_ZH, N_CAT, PATCHES, result  # noqa: E402

VERSIONS = ["v2_ae_tanh", "v2_vae_tanh", "v2_tanh_perceiver"]
CATS = ["餐飲", "零售", "商業服務", "藝文娛樂"]   # 要跑哪幾個類別，改這個清單即可
OFFSETS = [1, 2, 3]      # 每個類別各自加入的點數
ZOOM_PCT = (1, 99)         # latent 圖初始視野
DOT_BG = 3.0
DOT_FG = 5.0
BATCH = 512
OUT_DIR = os.path.join(ROOT, "lab")

mpl.rcParams["font.family"] = ["Heiti TC"]
mpl.rcParams["axes.unicode_minus"] = False
mpl.rcParams["figure.dpi"] = 130


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


def batch_encode_counts(model, version, counts):
    """ae / vae：批次跑聚合 count 向量，回傳 (N_patch, 2) 的 z。"""
    zs = []
    with torch.no_grad():
        for i in range(0, len(counts), BATCH):
            x = torch.from_numpy(counts[i:i + BATCH])
            out = model(x)
            z = out[2] if "vae" in version else out[0]
            zs.append(z.numpy())
    return np.concatenate(zs, axis=0)


def batch_encode_tokens(model, cat_lists):
    """perceiver：批次跑變長 token 列表（batch 內互相 pad），回傳 (N_patch, 2) 的 z。"""
    zs = []
    with torch.no_grad():
        for i in range(0, len(cat_lists), BATCH):
            chunk = cat_lists[i:i + BATCH]
            b, T = len(chunk), max(len(c) for c in chunk)
            tok = torch.zeros(b, T, dtype=torch.long)
            pad_mask = torch.ones(b, T, dtype=torch.bool)
            for j, c in enumerate(chunk):
                tok[j, :len(c)] = torch.from_numpy(c)
                pad_mask[j, :len(c)] = False
            z, _ = model(tok, pad_mask)
            zs.append(z.numpy())
    return np.concatenate(zs, axis=0)


def run_version(version, base_counts_all, base_cat_lists, cat_ids):
    mod = load_ae(version)
    d = np.load(result(version, "latents.npz"))
    z_all = d["z"]

    model = model_class(mod, version)(2)
    model.load_state_dict(torch.load(result(version, "ae.pt"), map_location="cpu"))
    model.eval()

    is_perceiver = "perceiver" in version
    z_new = {}   # (cat_id, k) -> (N_patch, 2)
    dist = {}    # (cat_id, k) -> (N_patch,)
    for c in cat_ids:
        for k in OFFSETS:
            if is_perceiver:
                cat_lists_k = [np.concatenate([base, np.full(k, c, dtype=np.int64)])
                              for base in base_cat_lists]
                zk = batch_encode_tokens(model, cat_lists_k)
            else:
                counts_k = base_counts_all.copy()
                counts_k[:, c] += k
                zk = batch_encode_counts(model, version, counts_k)
            z_new[c, k] = zk
            dist[c, k] = np.linalg.norm(zk - z_all, axis=1)

    return dict(version=version, z_all=z_all, z_new=z_new, dist=dist)


def zoom(ax, z_all):
    lo = np.percentile(z_all, ZOOM_PCT[0], axis=0)
    hi = np.percentile(z_all, ZOOM_PCT[1], axis=0)
    pad = (hi - lo) * 0.05
    ax.set_xlim(lo[0] - pad[0], hi[0] + pad[0])
    ax.set_ylim(lo[1] - pad[1], hi[1] + pad[1])


def style(ax, title):
    ax.set_title(title, fontsize=9)
    ax.tick_params(labelsize=6)
    ax.grid(alpha=0.15, linewidth=0.5)
    for sp in ax.spines.values():
        sp.set_alpha(0.3)


def plot_version(res, cat_ids):
    """一個版本一張圖：len(cat_ids) 列 x len(OFFSETS) 欄，每格是加了 k 個某類別後
    全體 patch 的新位置，顏色代表移動距離（heatmap）。"""
    z_all, z_new, dist = res["z_all"], res["z_new"], res["dist"]
    vmax = max(d.max() for d in dist.values())

    fig, axes = plt.subplots(len(cat_ids), len(OFFSETS),
                             figsize=(4.0 * len(OFFSETS), 3.4 * len(cat_ids)))
    axes = np.array(axes).reshape(len(cat_ids), len(OFFSETS))
    for row, c in enumerate(cat_ids):
        for j, k in enumerate(OFFSETS):
            ax = axes[row, j]
            ax.scatter(z_all[:, 0], z_all[:, 1], s=DOT_BG, c="#b8bcc4",
                       linewidths=0, alpha=0.35, rasterized=True)
            zk = z_new[c, k]
            ax.scatter(zk[:, 0], zk[:, 1], s=DOT_FG, c=dist[c, k],
                      cmap="inferno", vmin=0, vmax=vmax,
                      linewidths=0, alpha=0.9, rasterized=True)
            zoom(ax, z_all)
            style(ax, f"{CAT_ZH[c]}  +{k}")
            if j == 0:
                ax.set_ylabel("z2", fontsize=7)
            if row == len(cat_ids) - 1:
                ax.set_xlabel("z1", fontsize=7)

    fig.suptitle(f"{res['version']}｜ offset: {OFFSETS} ", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    out = os.path.join(OUT_DIR, f"v2_tah_all_outlier_experience_{res['version']}.png")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"已存 {out}")


def main():
    cat_ids = [CAT_ZH.index(c) for c in CATS]
    p = np.load(PATCHES)
    base_counts_all = all_base_counts(p)
    base_cat_lists = all_base_cat_lists(p)

    for v in VERSIONS:
        res = run_version(v, base_counts_all, base_cat_lists, cat_ids)
        print(f"{res['version']}：{len(base_cat_lists)} 個 patch")
        for c in cat_ids:
            row = "  ".join(f"+{k}:{res['dist'][c, k].mean():.2f}" for k in OFFSETS)
            print(f"  {CAT_ZH[c]:<6} 平均移動距離  {row}")
        plot_version(res, cat_ids)


main()
