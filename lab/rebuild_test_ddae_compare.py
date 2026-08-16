"""挑一個 patch，比較 v2_ddae 跟 v2_ddae_fsce 兩個模型的重建效果。

兩個模型的 Patches / MLPAE(x) 介面完全一樣（見各自的 model/*/ae.py），
差異只在 v2_ddae_fsce 訓練時多疊了 FSCE loss。用 --n 指定 patch 編號
（0 起算），畫出兩個模型「真實 count vs 重建 λ」的長條圖並排比較，
跟 .bak/v2_perceiver/analyze/rebuild_test.py 是同一種圖。
"""

import argparse
import importlib.util
import os
import sys

import numpy as np
import torch
import matplotlib as mpl
import matplotlib.pyplot as plt

ROOT = os.path.abspath(f"{os.path.dirname(__file__)}/..")
sys.path.insert(0, ROOT)
from config.dataset import CAT_COLORS, CAT_ZH, N_CAT, PATCHES, result  # noqa: E402

VERSIONS = ["v2_ddae", "v2_ddae_fsce"]
LATENT_DIM = 2   # v2 系列固定 latent_dim=2
OUT = os.path.join(ROOT, "lab", "rebuild_test_ddae_compare.png")

mpl.rcParams["font.family"] = ["Heiti TC"]
mpl.rcParams["axes.unicode_minus"] = False
mpl.rcParams["figure.dpi"] = 130


def load_ae_module(version):
    """用 importlib 從 model/<version>/ae.py 載入獨立模組，避免兩個版本的
    ae.py 都叫 "ae" 互相覆蓋。回傳該模組物件（有 MLPAE、Patches...）。
    """
    path = os.path.join(ROOT, "model", version, "ae.py")
    spec = importlib.util.spec_from_file_location(f"ae_{version}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def rebuild_one(version, n):
    """載入 version 對應的模型與 checkpoint，重建 patch n。
    回傳 dict：cnt（真實 count）、lam（重建 λ）、dev、nll、z。
    """
    ae = load_ae_module(version)
    data = ae.Patches(PATCHES)
    assert 0 <= n < data.n, f"--n 要在 0~{data.n - 1}"

    model = ae.MLPAE(LATENT_DIM)
    model.load_state_dict(torch.load(result(version, "ae.pt"), map_location="cpu"))
    model.eval()

    idx = torch.tensor([n])
    x = data.agg(idx)
    with torch.no_grad():
        z, log_lam = model(x)
    dev = ae.poisson_deviance(log_lam, x).item()
    nll = ae.poisson_nll(log_lam, x).item()

    return {
        "cnt": x[0].numpy(),
        "lam": np.round(torch.exp(log_lam)[0].numpy()),
        "z": z[0].numpy(),
        "dev": dev,
        "nll": nll,
        "n_poi": int(data.n_poi[n]),
        "lat": float(data.lat[n]),
        "lon": float(data.lon[n]),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=0, help="patch 編號（0 起算）")
    n = ap.parse_args().n

    out = {v: rebuild_one(v, n) for v in VERSIONS}
    ref = out[VERSIONS[0]]
    print(f"patch {n}：POI {ref['n_poi']}  ({ref['lat']:.5f}, {ref['lon']:.5f})")

    err_cache = {}
    for v in VERSIONS:
        err_cache[v] = np.load(result(v, "latents.npz"))["err"]

    fig, axes = plt.subplots(1, 2, figsize=(16, 5.5), sharey=True)
    idx_c = np.arange(N_CAT)
    w = 0.38
    for ax, v in zip(axes, VERSIONS):
        r = out[v]
        pct = (err_cache[v] < r["dev"]).mean() * 100
        print(f"\n[{v}] latent z = ({r['z'][0]:+.3f}, {r['z'][1]:+.3f})")
        print(f"[{v}] deviance = {r['dev']:.6f}  "
              f"（全體中位數 {np.median(err_cache[v]):.6f}，此 patch 排在第 {pct:.1f} 百分位）")
        print(f"[{v}] NLL = {r['nll']:.6f}（省略 log(y!) 常數項，可能為負）")

        ax.bar(idx_c - w / 2, r["cnt"], width=w, color=CAT_COLORS, alpha=0.55,
               label="真實 count")
        ax.bar(idx_c + w / 2, r["lam"], width=w, color=CAT_COLORS, alpha=0.95,
               hatch="//", edgecolor="white", label="重建 λ")
        ax.set_xticks(idx_c)
        ax.set_xticklabels(CAT_ZH, fontsize=9, rotation=20, ha="right")
        ax.set_ylabel("個數", fontsize=9)
        ax.legend(fontsize=9, framealpha=0.9)
        ax.grid(alpha=0.15, linewidth=0.5, axis="y")
        for s in ax.spines.values():
            s.set_alpha(0.3)
        ax.set_title(f"{v}（deviance {r['dev']:.4f}）", fontsize=11)

    fig.suptitle(f"v2_ddae vs v2_ddae_fsce patch {n}"
                 f"（latent_dim={LATENT_DIM}，POI {ref['n_poi']} 個）", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT, bbox_inches="tight")
    print(f"\n已存 {OUT}")


main()
