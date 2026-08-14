"""人工製造過飽和：挑一個最典型的 patch，持續塞同一類別的 POI，
看它在 latent（形狀）上怎麼移動、以及飽和度 z-score 怎麼變。

跟 v0_poisson_nll 版的差別是判定標準換了，因為這一版的 latent 不含密度：

  v0_poisson_nll：塞點 -> n_poi 變大 -> latent 直接被推走 -> robust 距離變大
                  但這只是在說「POI 變多了」，用不著 autoencoder。
  v0_sizefactor：塞點只透過「組成比例」與「空間排列」影響 latent。
                  真正的過飽和訊號改看飽和度 z-score（見 saturation.py）：
                  n_poi 相對於「同形狀鄰居」的密度高多少。

所以主圖多一張 z-score 曲線，那條才是直接對應專題目標的東西。
robust 距離仍然畫，但它現在只代表「形狀變得多不尋常」。

ADD_STEPS 改成前疏後密的排程而不是等距 500。原因是這一版對前段敏感得多：
base patch 只有幾十個 POI，加 100 個就足以把組成整個翻掉，
沿用等距 500 的話前面那段全部糊在一起看不到。

已知風險：軌跡上的 z-score 是拿「原始 23700 個 patch 的 latent」當鄰居池算的，
但軌跡點本身是模型沒看過的合成 patch。塞到後期組成極端（單一類別 99%），
latent 會跑到鄰居池的稀疏處，那裡的 z-score 跟 saturation.py 裡標記的
「latent 邊緣不可信」是同一個問題。所以曲線也印出鄰居平均距離。
"""

import os
import sys

import numpy as np
import torch
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(f"{os.path.dirname(__file__)}/../../.."))
from ae import CELL, GRID, N_CAT, HALF_WIDTH, ConvAE, poisson_deviance  # noqa: E402
from config.dataset import CAT_COLORS, CAT_ZH, PATCHES, result  # noqa: E402

LATENTS = result("v0_sizefactor", "latents.npz")
CKPT = result("v0_sizefactor", "ae.pt")
OUT = result("v0_sizefactor", "make_outlier.png")

LATENT_DIM = 2
ADD_CAT = 2
ADD_STEPS = [0, 10, 20, 50, 100, 200, 300, 500, 1000, 2000, 5000]
REPEATS = 5          # 每個點數重複幾次隨機灑點
SEED = 0
OUTLIER_PCT = 99.5   # 全體 robust 距離的離群門檻
KNN = 50             # 算飽和度用幾個形狀鄰居，跟 saturation.py 一致
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


class Saturation:
    """把 saturation.py 的 z-score 包成可以查詢任意 latent 點的形式。"""

    def __init__(self, z_all, n_poi):
        self.mean = z_all.mean(0)
        self.std = z_all.std(0)
        self.tree = cKDTree((z_all - self.mean) / self.std)
        self.y = np.log(n_poi.astype(float))

    def score(self, z, n):
        """z 形狀 (...,2)、n 是對應的 POI 數，回傳 (z-score, 鄰居平均距離)。"""
        q = (z.reshape(-1, 2) - self.mean) / self.std
        dist, idx = self.tree.query(q, k=KNN)
        mu = self.y[idx].mean(1)
        sd = np.maximum(self.y[idx].std(1), 1e-6)
        s = (np.log(np.asarray(n, dtype=float).reshape(-1)) - mu) / sd
        return s.reshape(z.shape[:-1]), dist.mean(1).reshape(z.shape[:-1])


def render(dx, dy, cat):
    """把一組點列表 binning 成 (1,10,40,40) 的原始 count 矩陣。

    正規化不在這裡做——ConvAE.forward 吃 raw count，內部才算 size factor。
    """
    ix = np.clip(np.floor(dx / CELL + GRID / 2), 0, GRID - 1).astype(np.int64)
    iy = np.clip(np.floor(dy / CELL + GRID / 2), 0, GRID - 1).astype(np.int64)
    flat = (cat.astype(np.int64) * GRID + iy) * GRID + ix
    counts = np.bincount(flat, minlength=N_CAT * GRID * GRID)
    counts = counts.reshape(1, N_CAT, GRID, GRID).astype(np.float32)
    return torch.from_numpy(counts)


def sample_disk(rng, k):
    """在半徑 HALF_WIDTH 的圓內均勻灑 k 個點。"""
    r = HALF_WIDTH * np.sqrt(rng.random(k))
    t = rng.random(k) * 2 * np.pi
    return r * np.cos(t), r * np.sin(t)


def run_cat(model, base, cat_id, rng):
    """對某一類跑完整條軌跡，回傳 (steps,REPEATS,2) 的 z 與 (steps,REPEATS) 的 deviance。"""
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
                zi, log_lam = model(x)
            z[j, r] = zi[0].numpy()
            err[j, r] = poisson_deviance(log_lam, x).item()
    return z, err


def style(ax, title, xlabel="z1", ylabel="z2"):
    ax.set_title(title, fontsize=10)
    ax.set_xlabel(xlabel, fontsize=8)
    ax.set_ylabel(ylabel, fontsize=8)
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
    n0 = int(n_poi[bi])

    sat = Saturation(z_all, n_poi)
    counts = np.array([n0 + k for k in ADD_STEPS], dtype=float)

    print(f"base patch {bi}：POI {n0}，robust 距離 {dist_all[bi]:.3f}，"
          f"({d['lat'][bi]:.5f}, {d['lon'][bi]:.5f})")
    print(f"主角類別 {CAT_ZH[ADD_CAT]}（channel {ADD_CAT}），"
          f"原本有 {(base[2] == ADD_CAT).sum()} 個")
    print(f"形狀離群門檻（全體 {OUTLIER_PCT}%）= {thr:.2f}\n")

    model = ConvAE(LATENT_DIM)
    model.load_state_dict(torch.load(CKPT, map_location="cpu"))
    model.eval()

    rng = np.random.default_rng(SEED)
    trajs, errs, scores, edges = [], [], [], []
    for c in range(N_CAT):
        z, err = run_cat(model, base, c, rng)
        mt = z.mean(1)
        sc, ed = sat.score(mt, counts)
        trajs.append(z)
        errs.append(err)
        scores.append(sc)
        edges.append(ed)
        dist = robust_distance(mt, z_all)
        cross = [k for j, k in enumerate(ADD_STEPS) if sc[j] > 3.0]
        print(f"  {CAT_ZH[c]:<6} 形狀距離 {dist[0]:5.2f} -> {dist[-1]:6.2f}   "
              f"飽和 z {sc[0]:+5.2f} -> {sc[-1]:+6.2f}   "
              + (f"z>3 於加入 {cross[0]} 個時" if cross else "z 未超過 3"))

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
    style(a, f"加入{CAT_ZH[ADD_CAT]}的形狀軌跡（放大 "
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
    style(b, f"10 類各加 {ADD_STEPS[-1]} 個的形狀軌跡（全域）")

    # 飽和度 z-score：這一版真正對應專題目標的曲線
    for k in range(N_CAT):
        c.plot(ADD_STEPS, scores[k], c=CAT_COLORS[k],
               lw=2.2 if k == ADD_CAT else 1.2,
               alpha=0.95 if k == ADD_CAT else 0.7, label=CAT_ZH[k])
    c.axhline(3.0, c="#555", ls="--", lw=1, label="z = 3")
    c.axhline(0.0, c="#aaa", lw=0.8)
    c.set_xscale("symlog", linthresh=10)
    c.set_xlim(0, ADD_STEPS[-1] * 1.15)
    c.set_xlabel("加入的點數（symlog）", fontsize=8)
    c.set_ylabel("飽和度 z-score", fontsize=8)
    c.set_title("飽和度：密度相對於「同形狀鄰居」高多少", fontsize=10)
    c.legend(fontsize=7, ncol=2, framealpha=0.9)
    c.tick_params(labelsize=7)
    c.grid(alpha=0.15, linewidth=0.5)

    # 形狀離群程度，對照組
    for k in range(N_CAT):
        e.plot(ADD_STEPS, robust_distance(trajs[k].mean(1), z_all),
               c=CAT_COLORS[k], lw=2.2 if k == ADD_CAT else 1.2,
               alpha=0.95 if k == ADD_CAT else 0.7, label=CAT_ZH[k])
    dist_main = robust_distance(traj, z_all)
    e.fill_between(ADD_STEPS, dist_main.min(1), dist_main.max(1),
                   color=CAT_COLORS[ADD_CAT], alpha=0.15)
    e.axhline(thr, c="#555", ls="--", lw=1,
              label=f"形狀離群門檻 {OUTLIER_PCT}% = {thr:.2f}")
    e.set_xscale("symlog", linthresh=10)
    e.set_xlim(0, ADD_STEPS[-1] * 1.15)
    e.set_xlabel("加入的點數（symlog）", fontsize=8)
    e.set_ylabel("latent robust 距離", fontsize=8)
    e.set_title("形狀有多不尋常（不含密度）", fontsize=10)
    e.legend(fontsize=7, ncol=2, framealpha=0.9)
    e.tick_params(labelsize=7)
    e.grid(alpha=0.15, linewidth=0.5)

    fig.suptitle(f"v0_sizefactor 人工過飽和：patch {bi}（原始 POI {n0}）"
                 f"逐步加入單一類別，每點重複 {REPEATS} 次", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT, bbox_inches="tight")
    print(f"\n已存 {OUT}")


main()
