"""v2_ae_tanh / v2_vae_tanh / v2_tanh_perceiver 三個「latent 訓練時就夾在
(-1,1)」的版本，各自從全體 latent 中最典型（robust 距離最小）的 patch
出發，對每個類別分別加入 1 / 5 / 10 個該類別的 POI，看 latent 怎麼移動
——對照三種架構（MLP / 機率化 mu / cross-attention）「訓練時就限制範圍」
對移動軌跡的影響是不是一致。

v2 系列的輸入本來就沒有座標，「加 POI」單純是把聚合 count 向量（或
perceiver 的 token 列表）裡某一類加上 k 個。
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
ADD_STEPS = [0,1,5, 10, 30, 50]   # 累積加入的點數
ZOOM_PCT = (1, 99)          # latent 圖初始視野
DOT = 3.0
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


def robust_distance(z, ref):
    """用全體 latent 的 中位數 / MAD 標準化後，算 z 到 latent 中心的距離。"""
    med = np.median(ref, axis=0)
    mad = np.median(np.abs(ref - med), axis=0) * 1.4826
    return np.linalg.norm((z - med) / mad, axis=-1)


def encode(model, version, base_cat, cat_id, k):
    """base_cat 是 base patch 每個 POI 的類別 id；回傳加了 k 個 cat_id 後的 z。"""
    cat = np.concatenate([base_cat, np.full(k, cat_id, dtype=base_cat.dtype)])
    with torch.no_grad():
        if "perceiver" in version:
            tok = torch.from_numpy(cat.astype(np.int64)).unsqueeze(0)
            pad_mask = torch.zeros(1, len(cat), dtype=torch.bool)
            z, _ = model(tok, pad_mask)
        else:
            counts = np.bincount(cat, minlength=N_CAT).astype(np.float32)
            x = torch.from_numpy(counts).unsqueeze(0)
            out = model(x)
            z = out[2] if "vae" in version else out[0]   # vae: (mu,logvar,z,log_lam)
    return z[0].numpy()


def run_version(version):
    mod = load_ae(version)
    d = np.load(result(version, "latents.npz"))
    p = np.load(PATCHES)
    z_all = d["z"]

    model = model_class(mod, version)(2)
    model.load_state_dict(torch.load(result(version, "ae.pt"), map_location="cpu"))
    model.eval()

    dist_all = robust_distance(z_all, z_all)
    bi = int(np.argmin(dist_all))
    s, e = p["offsets"][bi], p["offsets"][bi + 1]
    base_cat = p["cat"][s:e].astype(np.int64)

    trajs = np.zeros((N_CAT, len(ADD_STEPS), 2))
    for c in range(N_CAT):
        for j, k in enumerate(ADD_STEPS):
            trajs[c, j] = encode(model, version, base_cat, c, k)

    return dict(version=version, bi=bi, n_poi=int(d["n_poi"][bi]), z_all=z_all,
                trajs=trajs)


def zoom(ax, z_all):
    lo = np.percentile(z_all, ZOOM_PCT[0], axis=0)
    hi = np.percentile(z_all, ZOOM_PCT[1], axis=0)
    pad = (hi - lo) * 0.05
    ax.set_xlim(lo[0] - pad[0], hi[0] + pad[0])
    ax.set_ylim(lo[1] - pad[1], hi[1] + pad[1])


def style(ax, title):
    ax.set_title(title, fontsize=10)
    ax.tick_params(labelsize=7)
    ax.grid(alpha=0.15, linewidth=0.5)
    for sp in ax.spines.values():
        sp.set_alpha(0.3)


NCOLS = 5


def plot_version(res):
    """一個版本一張圖，N_CAT 個子圖，每個子圖是單一類別加 0/1/5/10 個的軌跡。"""
    trajs, z_all = res["trajs"], res["z_all"]
    nrows = -(-N_CAT // NCOLS)   # ceil

    fig, axes = plt.subplots(nrows, NCOLS, figsize=(4.2 * NCOLS, 4.2 * nrows))
    sc = None
    for c, ax in enumerate(axes.flat):
        if c >= N_CAT:
            ax.axis("off")
            continue
        ax.scatter(z_all[:, 0], z_all[:, 1], s=DOT, c="#b8bcc4",
                   linewidths=0, alpha=0.4, rasterized=True)
        t = trajs[c]
        ax.plot(t[:, 0], t[:, 1], c="#c0392b", lw=1.5, zorder=3)
        sc = ax.scatter(t[:, 0], t[:, 1], c=ADD_STEPS, cmap="autumn_r", s=34,
                        zorder=4, edgecolors="k", linewidths=0.4)
        ax.scatter([t[0, 0]], [t[0, 1]], s=130, marker="*", c="#1a1a1a", zorder=5)
        zoom(ax, z_all)

        style(ax, CAT_ZH[c])
        ax.set_xlabel("z1", fontsize=7)
        ax.set_ylabel("z2", fontsize=7)

    cb = fig.colorbar(sc, ax=axes, fraction=0.015, pad=0.01)
    cb.set_label("加入的點數", fontsize=9)
    cb.ax.tick_params(labelsize=8)

    fig.suptitle(f"{res['version']}｜patch {res['bi']}（原始 POI {res['n_poi']}）"
                f"各類別加入 {ADD_STEPS[1:]} 個 POI 的 latent 移動軌跡", fontsize=13)
    out = os.path.join(OUT_DIR, f"v2_tah_outlier_experience_{res['version']}.png")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"已存 {out}")


def main():
    for v in VERSIONS:
        res = run_version(v)
        print(f"{res['version']}：base patch {res['bi']}（POI {res['n_poi']}）")
        for c in range(N_CAT):
            dist = robust_distance(res["trajs"][c], res["z_all"])
            print(f"  {CAT_ZH[c]:<6} 距離 "
                  + " -> ".join(f"{v:.2f}" for v in dist))
        plot_version(res)


main()
