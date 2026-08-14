"""人工製造過飽和：挑一個 latent 正中央（最典型）的 patch，
持續往它的聚合 count 向量加同一類別的個數，看 latent 怎麼移動。

v2 系列的輸入本來就沒有座標，「加 POI」在這裡單純是把 (N_CAT,) 向量
裡 ADD_CAT 那一維加上 k——不必真的灑點、也不需要 v0 版本那種
REPEATS（同一個 k 灌幾次都是同一個向量，結果必然相同，隨機性已經
不存在了），這是拿掉空間結構後最直接的簡化。

跟 v2_ae 同一支腳本比較：encoder 換成機率化的 mu/logvar head 後，
同樣「加某一類 POI」的動作會不會讓 latent 移動得更乾脆、
離群門檻跨越得更早或更晚。畫的 z 是 mu（eval 模式下的決定性輸出）。

ADD_STEPS 依這份資料集的規模校準：目前 1233 個 patch 的 n_poi
中位數約 15、最大約 85，所以 0~200 已經涵蓋「正常」到「明顯異常」。
"""

import os
import sys

import numpy as np
import torch
import matplotlib as mpl
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(f"{os.path.dirname(__file__)}/../../.."))
from ae import VAE, N_CAT, poisson_deviance  # noqa: E402
from config.dataset import CAT_COLORS, CAT_ZH, PATCHES, result  # noqa: E402

VERSION = "v2_vae"
LATENTS = result(VERSION, "latents.npz")
CKPT = result(VERSION, "ae.pt")
OUT = result(VERSION, "make_outlier.png")

LATENT_DIM = 2
ADD_CAT = 2
ADD_STEPS = list(range(0, 201, 20))   # 累積加入的點數
OUTLIER_PCT = 99.5   # 全體 robust 距離的離群門檻
ZOOM_PCT = (1, 99)
DOT = 3.0

mpl.rcParams["font.family"] = ["Heiti TC"]
mpl.rcParams["axes.unicode_minus"] = False
mpl.rcParams["figure.dpi"] = 130


def robust_distance(z, ref):
    """用全體 latent 的 中位數 / MAD 標準化後，算 z 到 latent 中心的距離。"""
    med = np.median(ref, axis=0)
    mad = np.median(np.abs(ref - med), axis=0) * 1.4826
    return np.linalg.norm((z - med) / mad, axis=-1)


def run_cat(model, base_counts, cat_id):
    """對某一類跑完整條軌跡，回傳 (steps, 2) 的 z 與 deviance。"""
    z = np.zeros((len(ADD_STEPS), LATENT_DIM))
    err = np.zeros(len(ADD_STEPS))
    for j, k in enumerate(ADD_STEPS):
        counts = base_counts.copy()
        counts[cat_id] += k
        x = torch.from_numpy(counts.reshape(1, N_CAT).astype(np.float32))
        with torch.no_grad():
            _, _, zi, log_lam = model(x)   # eval 模式下 zi = mu，決定性
        z[j] = zi[0].numpy()
        err[j] = poisson_deviance(log_lam, x).item()
    return z, err


def style(ax, title):
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("z1", fontsize=8)
    ax.set_ylabel("z2", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.grid(alpha=0.15, linewidth=0.5)
    for s in ax.spines.values():
        s.set_alpha(0.3)


def zoom(ax, z_all):
    lo = np.percentile(z_all, ZOOM_PCT[0], axis=0)
    hi = np.percentile(z_all, ZOOM_PCT[1], axis=0)
    pad = (hi - lo) * 0.05
    ax.set_xlim(lo[0] - pad[0], hi[0] + pad[0])
    ax.set_ylim(lo[1] - pad[1], hi[1] + pad[1])


def background(ax, z_all):
    ax.scatter(z_all[:, 0], z_all[:, 1], s=DOT, c="#b8bcc4",
               linewidths=0, alpha=0.4, rasterized=True, label="全體 patch")


def main():
    d = np.load(LATENTS)
    p = np.load(PATCHES)
    z_all, n_poi = d["z"], d["n_poi"]

    dist_all = robust_distance(z_all, z_all)
    bi = int(np.argmin(dist_all))
    s, e = p["offsets"][bi], p["offsets"][bi + 1]
    base_counts = np.bincount(p["cat"][s:e].astype(np.int64), minlength=N_CAT)
    thr = np.percentile(dist_all, OUTLIER_PCT)

    print(f"base patch {bi}：POI {n_poi[bi]}，robust 距離 {dist_all[bi]:.3f}，"
          f"({d['lat'][bi]:.5f}, {d['lon'][bi]:.5f})")
    print(f"主角類別 {CAT_ZH[ADD_CAT]}（channel {ADD_CAT}），"
          f"原本有 {base_counts[ADD_CAT]} 個")
    print(f"離群門檻（全體 {OUTLIER_PCT}%）= {thr:.2f}\n")

    model = VAE(LATENT_DIM)
    model.load_state_dict(torch.load(CKPT, map_location="cpu"))
    model.eval()

    trajs, errs = [], []
    for c in range(N_CAT):
        z, err = run_cat(model, base_counts, c)
        trajs.append(z)
        errs.append(err)
        dist = robust_distance(z, z_all)
        cross = [k for j, k in enumerate(ADD_STEPS) if dist[j] > thr]
        print(f"  {CAT_ZH[c]:<6} 加 {ADD_STEPS[-1]} 個後 "
              f"z=({z[-1, 0]:+6.2f}, {z[-1, 1]:+6.2f})  "
              f"robust 距離 {dist[0]:.2f} -> {dist[-1]:.2f}  "
              + (f"{cross[0]} 個時越過門檻" if cross else "沒越過門檻"))

    traj = trajs[ADD_CAT]

    fig, ((a, b), (c, e)) = plt.subplots(2, 2, figsize=(13, 11))

    background(a, z_all)
    a.plot(traj[:, 0], traj[:, 1], c="#c0392b", lw=1.8,
           label=f"軌跡（加 0~{ADD_STEPS[-1]} 個{CAT_ZH[ADD_CAT]}）")
    sc = a.scatter(traj[:, 0], traj[:, 1], c=ADD_STEPS, s=28,
                   cmap="autumn_r", zorder=3, edgecolors="k", linewidths=0.3)
    a.scatter(traj[0, 0], traj[0, 1], s=110, marker="*", c="#1a1a1a",
              zorder=4, label="原始 patch")
    cb = fig.colorbar(sc, ax=a, fraction=0.046, pad=0.02)
    cb.set_label("加入的點數", fontsize=8)
    cb.ax.tick_params(labelsize=7)
    zoom(a, z_all)
    a.legend(fontsize=7.5, markerscale=1.3, framealpha=0.9)
    style(a, f"加入{CAT_ZH[ADD_CAT]}的移動軌跡（放大 "
             f"{ZOOM_PCT[0]}~{ZOOM_PCT[1]}%）")

    background(b, z_all)
    for k in range(N_CAT):
        mt = trajs[k]
        b.plot(mt[:, 0], mt[:, 1], c=CAT_COLORS[k],
               lw=2.2 if k == ADD_CAT else 1.2,
               alpha=0.95 if k == ADD_CAT else 0.7, label=CAT_ZH[k])
        b.scatter(mt[-1, 0], mt[-1, 1], s=26, c=CAT_COLORS[k], zorder=3,
                  edgecolors="k", linewidths=0.3)
    b.scatter(traj[0, 0], traj[0, 1], s=110, marker="*", c="#1a1a1a", zorder=4)
    b.legend(fontsize=7, ncol=2, framealpha=0.9)
    style(b, f"{N_CAT} 類各加 {ADD_STEPS[-1]} 個的軌跡對照（全域）")

    for k in range(N_CAT):
        dist = robust_distance(trajs[k], z_all)
        c.plot(ADD_STEPS, dist, c=CAT_COLORS[k],
               lw=2.2 if k == ADD_CAT else 1.2,
               alpha=0.95 if k == ADD_CAT else 0.7, label=CAT_ZH[k])
        e.plot(ADD_STEPS, errs[k], c=CAT_COLORS[k],
               lw=2.2 if k == ADD_CAT else 1.2,
               alpha=0.95 if k == ADD_CAT else 0.7, label=CAT_ZH[k])

    c.axhline(thr, c="#555", ls="--", lw=1,
              label=f"離群門檻 {OUTLIER_PCT}% = {thr:.2f}")
    c.set_xlabel("加入的點數", fontsize=8)
    c.set_ylabel("latent robust 距離", fontsize=8)
    c.set_title("離 latent 中心的距離", fontsize=10)
    c.legend(fontsize=7, ncol=2, framealpha=0.9)
    c.tick_params(labelsize=7)
    c.grid(alpha=0.15, linewidth=0.5)

    e.set_xlabel("加入的點數", fontsize=8)
    e.set_ylabel("Poisson deviance", fontsize=8)
    e.set_title("重建誤差", fontsize=10)
    e.legend(fontsize=7, ncol=2, framealpha=0.9)
    e.tick_params(labelsize=7)
    e.grid(alpha=0.15, linewidth=0.5)

    fig.suptitle(f"{VERSION} 人工過飽和：patch {bi}（原始 POI {n_poi[bi]}，latent 正中央）"
                 f"逐步加入單一類別", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT, bbox_inches="tight")
    print(f"\n已存 {OUT}")


main()
