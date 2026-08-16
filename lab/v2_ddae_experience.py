"""v2_ddae_fsce 的 outlier 平移實驗。

流程：
1. 載入 v2_ddae_fsce 的 latents.npz，用 robust 距離丟掉最外圍 5% 的離群 patch。
2. 對保留下來的 latent 做 min-max 正規化到 [0,1]，用來視覺化。
3. 對保留的全體 patch，分別在每個類別加入 1/2/3 個 POI，重新 encode，
   看加完之後的 latent 在 min-max 空間裡移動了多遠。
4. 畫圖：每個類別 × 每個 offset 一格子圖，前景顏色 = 移動距離。

一張大圖 + 一張表（CSV + console）。
"""

import importlib.util
import os
import sys

import numpy as np
import pandas as pd
import torch
import matplotlib as mpl
import matplotlib.pyplot as plt

ROOT = os.path.abspath(f"{os.path.dirname(__file__)}/..")
sys.path.insert(0, ROOT)
from config.dataset import CAT_ZH, N_CAT, PATCHES, result  # noqa: E402

VERSION = "v2_ddae_fsce"
CATS = ["餐飲", "零售", "商業服務", "藝文娛樂"]
OFFSETS = [1, 2, 3]
OUTLIER_DROP_PCT = 5          # 丟掉 robust 距離最大的這個百分比
BATCH = 512
DOT_BG = 3.0
DOT_FG = 5.0
OUT_DIR = os.path.join(ROOT, "lab")

mpl.rcParams["font.family"] = ["Heiti TC"]
mpl.rcParams["axes.unicode_minus"] = False
mpl.rcParams["figure.dpi"] = 130


# ── helpers ──────────────────────────────────────────────────────────────

def load_ae(version):
    """把 ae.py 用獨立模組名載入。"""
    path = os.path.join(ROOT, "model", version, "ae.py")
    spec = importlib.util.spec_from_file_location(f"ae_{version}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def robust_distance(z, ref):
    """用全體 latent 的 中位數 / MAD 標準化後，算到中心的距離。"""
    med = np.median(ref, axis=0)
    mad = np.median(np.abs(ref - med), axis=0) * 1.4826
    mad = np.where(mad > 0, mad, 1.0)
    return np.linalg.norm((z - med) / mad, axis=-1)


def all_base_counts(p):
    """(N_patch, N_CAT) 的原始類別 count 向量。"""
    n = len(p["n_poi"])
    cat = p["cat"].astype(np.int64)
    owner = np.repeat(np.arange(n), np.diff(p["offsets"]))
    counts = np.zeros((n, N_CAT), dtype=np.float32)
    np.add.at(counts, (owner, cat), 1.0)
    return counts


def minmax(z, z_min, z_max):
    """min-max 正規化到 [0,1]。"""
    rng = z_max - z_min
    rng = np.where(rng > 0, rng, 1.0)
    return (z - z_min) / rng


# ── encode ───────────────────────────────────────────────────────────────

def batch_encode(model, counts):
    """批次跑聚合 count 向量，回傳 (N, 2) 的 z。"""
    zs = []
    with torch.no_grad():
        for i in range(0, len(counts), BATCH):
            x = torch.from_numpy(counts[i:i + BATCH])
            z, _ = model(x)        # MLPAE: (z, log_lam)
            zs.append(z.numpy())
    return np.concatenate(zs, axis=0)


def batch_deviance(model, mod, counts):
    """批次算 Poisson deviance，回傳 (N,)。"""
    devs = []
    with torch.no_grad():
        for i in range(0, len(counts), BATCH):
            x = torch.from_numpy(counts[i:i + BATCH])
            _, log_lam = model(x)
            devs.append(mod.poisson_deviance(log_lam, x).numpy())
    return np.concatenate(devs, axis=0)


# ── main logic ───────────────────────────────────────────────────────────

def run(base_counts_all, keep_mask, cat_ids):
    """回傳 dict：z_keep_norm, z_new_norm, dist_norm, table_rows。"""
    mod = load_ae(VERSION)
    d = np.load(result(VERSION, "latents.npz"))
    z_all = d["z"]                   # (N_all, 2)
    err_all = d["err"]

    # ── 丟 5% 離群值 ──
    z_keep = z_all[keep_mask]        # (N_keep, 2)
    err_keep = err_all[keep_mask]
    counts_keep = base_counts_all[keep_mask]

    # min-max 基準用 keep 的範圍
    z_min = z_keep.min(axis=0)
    z_max = z_keep.max(axis=0)
    z_keep_norm = minmax(z_keep, z_min, z_max)

    # 載入模型
    model = mod.MLPAE(2)
    model.load_state_dict(torch.load(result(VERSION, "ae.pt"), map_location="cpu"))
    model.eval()

    z_new_norm = {}      # (cat_id, k) -> (N_keep, 2) 已正規化
    dist_norm = {}       # (cat_id, k) -> (N_keep,)
    rows = []

    for c in cat_ids:
        for k in OFFSETS:
            counts_k = counts_keep.copy()
            counts_k[:, c] += k
            zk = batch_encode(model, counts_k)
            dk = batch_deviance(model, mod, counts_k)

            zk_n = minmax(zk, z_min, z_max)
            z_new_norm[c, k] = zk_n
            d_move = np.linalg.norm(zk_n - z_keep_norm, axis=1)
            dist_norm[c, k] = d_move

            dev_inc = dk - err_keep
            rows.append(dict(
                類別=CAT_ZH[c], 加入點數=k,
                移動距離_norm=d_move.mean(),
                重建損失提升=dev_inc.mean(),
            ))

    return dict(
        z_keep_norm=z_keep_norm,
        z_new_norm=z_new_norm,
        dist_norm=dist_norm,
        table=pd.DataFrame(rows),
    )


# ── plot ─────────────────────────────────────────────────────────────────

def zoom_01(ax, margin=0.04):
    """min-max 正規化後視野固定 [0,1]。"""
    ax.set_xlim(-margin, 1 + margin)
    ax.set_ylim(-margin, 1 + margin)


def style(ax, title):
    ax.set_title(title, fontsize=9)
    ax.tick_params(labelsize=6)
    ax.grid(alpha=0.15, linewidth=0.5)
    for sp in ax.spines.values():
        sp.set_alpha(0.3)


def plot(res, cat_ids):
    z_keep_norm = res["z_keep_norm"]
    z_new_norm = res["z_new_norm"]
    dist_norm = res["dist_norm"]
    vmax = max(d.max() for d in dist_norm.values())

    n_cats = len(cat_ids)
    n_offs = len(OFFSETS)
    fig, axes = plt.subplots(n_cats, n_offs,
                             figsize=(4.0 * n_offs, 3.4 * n_cats))
    axes = np.array(axes).reshape(n_cats, n_offs)

    for row, c in enumerate(cat_ids):
        for j, k in enumerate(OFFSETS):
            ax = axes[row, j]
            # 背景：保留 patch 的原始 latent（min-max 正規化後）
            ax.scatter(z_keep_norm[:, 0], z_keep_norm[:, 1],
                       s=DOT_BG, c="#b8bcc4", linewidths=0,
                       alpha=0.35, rasterized=True)
            # 前景：加了 k 個後的新位置，顏色 = 移動距離
            zk = z_new_norm[c, k]
            ax.scatter(zk[:, 0], zk[:, 1], s=DOT_FG, c=dist_norm[c, k],
                       cmap="inferno", vmin=0, vmax=vmax,
                       linewidths=0, alpha=0.9, rasterized=True)
            zoom_01(ax)
            style(ax, f"{CAT_ZH[c]}  +{k}")
            if j == 0:
                ax.set_ylabel("z2 (min-max)", fontsize=7)
            if row == n_cats - 1:
                ax.set_xlabel("z1 (min-max)", fontsize=7)

    fig.suptitle(f"{VERSION}｜去除 {OUTLIER_DROP_PCT}% 離群後 min-max 正規化"
                 f"｜offset: {OFFSETS}", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    out = os.path.join(OUT_DIR, f"v2_ddae_experience_{VERSION}.png")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"已存 {out}")


# ── entry ────────────────────────────────────────────────────────────────

def main():
    cat_ids = [CAT_ZH.index(c) for c in CATS]
    p = np.load(PATCHES)
    base_counts_all = all_base_counts(p)

    # 用全體 latent 算 robust 距離，丟掉最外圍 5%
    d = np.load(result(VERSION, "latents.npz"))
    z_all = d["z"]
    rd = robust_distance(z_all, z_all)
    thr = np.percentile(rd, 100 - OUTLIER_DROP_PCT)
    keep_mask = rd <= thr
    n_all = len(z_all)
    n_keep = keep_mask.sum()
    print(f"全部 {n_all} 個 patch，丟掉 robust 距離 > {thr:.2f} 的 "
          f"{n_all - n_keep} 個（{OUTLIER_DROP_PCT}%），保留 {n_keep} 個")

    res = run(base_counts_all, keep_mask, cat_ids)

    # 印表 + 存 CSV
    df = res["table"]
    print(f"\n=== {VERSION}（{n_keep} 個 patch 的平均）===")
    print(df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    csv_out = os.path.join(OUT_DIR, f"v2_ddae_experience_{VERSION}.csv")
    df.to_csv(csv_out, index=False, float_format="%.4f")
    print(f"已存 {csv_out}")

    # 畫圖
    plot(res, cat_ids)


main()
