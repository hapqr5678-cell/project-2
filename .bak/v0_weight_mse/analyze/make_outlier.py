"""人工製造過飽和：挑一個 latent 正中央（最典型）的 patch，
持續往裡面塞同一類別的 POI，看它在 latent space 上怎麼移動。

假設是「城市 POI 已經飽和，硬塞會過飽和 -> latent 出現離群」，
這支就是直接把那件事做出來：
  base patch = robust 距離最小的 patch（latent 正中央）
  每步在半徑內隨機灑 k 個點、重新 encode，畫出軌跡

主角是 ADD_CAT（學校暫時歸在「社區/政府」），
另外把 10 類都各跑一次當對照，看 latent 對哪些類別有反應。
灑點位置有隨機性，所以每個 k 重複 REPEATS 次，畫細線 + 平均軌跡。
"""

import os
import sys

import numpy as np
import torch
import matplotlib as mpl
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(f"{os.path.dirname(__file__)}/../../.."))
from ae import ConvAE, CELL, GRID, N_CAT, HALF_WIDTH  # noqa: E402
from config.dataset import CAT_COLORS, CAT_ZH, PATCHES, result  # noqa: E402

LATENTS = result("v0_weight_mse", "latents.npz")
CKPT = result("v0_weight_mse", "ae.pt")
OUT = result("v0_weight_mse", "make_outlier.png")

LATENT_DIM = 2
ADD_CAT = 2       
ADD_STEPS = list(range(0, 5001, 500))   # 累積加入的點數
REPEATS = 5          # 每個點數重複幾次隨機灑點
SEED = 0
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


def render(dx, dy, cat):
    """把一組點列表 binning 成 (1,10,40,40) 的 log1p(count) 矩陣。"""
    ix = np.clip(np.floor(dx / CELL + GRID / 2), 0, GRID - 1).astype(np.int64)
    iy = np.clip(np.floor(dy / CELL + GRID / 2), 0, GRID - 1).astype(np.int64)
    flat = (cat.astype(np.int64) * GRID + iy) * GRID + ix
    counts = np.bincount(flat, minlength=N_CAT * GRID * GRID)
    counts = counts.reshape(1, N_CAT, GRID, GRID).astype(np.float32)
    return torch.from_numpy(np.log1p(counts))


def sample_disk(rng, k):
    """在半徑 HALF_WIDTH 的圓內均勻灑 k 個點。"""
    r = HALF_WIDTH * np.sqrt(rng.random(k))
    t = rng.random(k) * 2 * np.pi
    return r * np.cos(t), r * np.sin(t)


def run_cat(model, base, cat_id, rng):
    """對某一類跑完整條軌跡，回傳 (steps, REPEATS, 2) 的 z 與 MSE。"""
    dx0, dy0, cat0 = base
    z = np.zeros((len(ADD_STEPS), REPEATS, LATENT_DIM))
    err = np.zeros((len(ADD_STEPS), REPEATS))
    for r in range(REPEATS):
        ax, ay = sample_disk(rng, ADD_STEPS[-1])   # 同一串點逐步加入
        for j, k in enumerate(ADD_STEPS):
            x = render(np.concatenate([dx0, ax[:k]]),
                       np.concatenate([dy0, ay[:k]]),
                       np.concatenate([cat0,
                                       np.full(k, cat_id, dtype=cat0.dtype)]))
            with torch.no_grad():
                zi, recon = model(x)
            z[j, r] = zi[0].numpy()
            err[j, r] = ((recon - x) ** 2).mean().item()
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
    base = (p["dx"][s:e], p["dy"][s:e], p["cat"][s:e])
    thr = np.percentile(dist_all, OUTLIER_PCT)

    print(f"base patch {bi}：POI {n_poi[bi]}，robust 距離 {dist_all[bi]:.3f}，"
          f"({d['lat'][bi]:.5f}, {d['lon'][bi]:.5f})")
    print(f"主角類別 {CAT_ZH[ADD_CAT]}（channel {ADD_CAT}），"
          f"原本有 {(base[2] == ADD_CAT).sum()} 個")
    print(f"離群門檻（全體 {OUTLIER_PCT}%）= {thr:.2f}\n")

    model = ConvAE(LATENT_DIM)
    model.load_state_dict(torch.load(CKPT, map_location="cpu"))
    model.eval()

    rng = np.random.default_rng(SEED)
    trajs, errs = [], []
    for c in range(N_CAT):
        z, err = run_cat(model, base, c, rng)
        trajs.append(z)
        errs.append(err)
        dist = robust_distance(z.mean(1), z_all)
        cross = [k for j, k in enumerate(ADD_STEPS) if dist[j] > thr]
        print(f"  {CAT_ZH[c]:<6} 加 {ADD_STEPS[-1]} 個後 "
              f"z=({z.mean(1)[-1, 0]:+6.2f}, {z.mean(1)[-1, 1]:+6.2f})  "
              f"robust 距離 {dist[0]:.2f} -> {dist[-1]:.2f}  "
              + (f"{cross[0]} 個時越過門檻" if cross else "沒越過門檻"))

    traj = trajs[ADD_CAT]
    mean_traj = traj.mean(1)

    fig, ((a, b), (c, e)) = plt.subplots(2, 2, figsize=(13, 11))

    # 主角類別：全體 + 每次重複的細線 + 平均軌跡
    background(a, z_all)
    for r in range(REPEATS):
        a.plot(traj[:, r, 0], traj[:, r, 1], c="#c0392b", lw=0.6, alpha=0.35)
    a.plot(mean_traj[:, 0], mean_traj[:, 1], c="#c0392b", lw=1.8,
           label=f"平均軌跡（加 0~{ADD_STEPS[-1]} 個{CAT_ZH[ADD_CAT]}）")
    sc = a.scatter(mean_traj[:, 0], mean_traj[:, 1], c=ADD_STEPS, s=28,
                   cmap="autumn_r", zorder=3, edgecolors="k", linewidths=0.3)
    a.scatter(mean_traj[0, 0], mean_traj[0, 1], s=110, marker="*", c="#1a1a1a",
              zorder=4, label="原始 patch")
    cb = fig.colorbar(sc, ax=a, fraction=0.046, pad=0.02)
    cb.set_label("加入的點數", fontsize=8)
    cb.ax.tick_params(labelsize=7)
    zoom(a, z_all)
    a.legend(fontsize=7.5, markerscale=1.3, framealpha=0.9)
    style(a, f"加入{CAT_ZH[ADD_CAT]}的移動軌跡（放大 "
             f"{ZOOM_PCT[0]}~{ZOOM_PCT[1]}%）")

    # 10 類對照
    background(b, z_all)
    for k in range(N_CAT):
        mt = trajs[k].mean(1)
        b.plot(mt[:, 0], mt[:, 1], c=CAT_COLORS[k],
               lw=2.2 if k == ADD_CAT else 1.2,
               alpha=0.95 if k == ADD_CAT else 0.7, label=CAT_ZH[k])
        b.scatter(mt[-1, 0], mt[-1, 1], s=26, c=CAT_COLORS[k], zorder=3,
                  edgecolors="k", linewidths=0.3)
    b.scatter(mean_traj[0, 0], mean_traj[0, 1], s=110, marker="*", c="#1a1a1a",
              zorder=4)
    b.legend(fontsize=7, ncol=2, framealpha=0.9)
    style(b, f"10 類各加 {ADD_STEPS[-1]} 個的軌跡對照（全域）")

    for k in range(N_CAT):
        dist = robust_distance(trajs[k].mean(1), z_all)
        c.plot(ADD_STEPS, dist, c=CAT_COLORS[k],
               lw=2.2 if k == ADD_CAT else 1.2,
               alpha=0.95 if k == ADD_CAT else 0.7, label=CAT_ZH[k])
        e.plot(ADD_STEPS, errs[k].mean(1), c=CAT_COLORS[k],
               lw=2.2 if k == ADD_CAT else 1.2,
               alpha=0.95 if k == ADD_CAT else 0.7, label=CAT_ZH[k])

    dist_main = robust_distance(traj, z_all)
    c.fill_between(ADD_STEPS, dist_main.min(1), dist_main.max(1),
                   color=CAT_COLORS[ADD_CAT], alpha=0.15)
    c.axhline(thr, c="#555", ls="--", lw=1,
              label=f"離群門檻 {OUTLIER_PCT}% = {thr:.2f}")
    c.set_xlabel("加入的點數", fontsize=8)
    c.set_ylabel("latent robust 距離", fontsize=8)
    c.set_title("離 latent 中心的距離", fontsize=10)
    c.legend(fontsize=7, ncol=2, framealpha=0.9)
    c.tick_params(labelsize=7)
    c.grid(alpha=0.15, linewidth=0.5)

    e.set_xlabel("加入的點數", fontsize=8)
    e.set_ylabel("MSE", fontsize=8)
    e.set_title("重建誤差", fontsize=10)
    e.legend(fontsize=7, ncol=2, framealpha=0.9)
    e.tick_params(labelsize=7)
    e.grid(alpha=0.15, linewidth=0.5)

    fig.suptitle(f"人工過飽和：patch {bi}（原始 POI {n_poi[bi]}，latent 正中央）"
                 f"逐步加入單一類別，每點重複 {REPEATS} 次", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT, bbox_inches="tight")
    print(f"\n已存 {OUT}")


main()
