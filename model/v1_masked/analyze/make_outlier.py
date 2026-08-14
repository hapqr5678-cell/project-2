"""人工製造過飽和：挑一個 latent 正中央（最典型）的 patch，
持續往裡面塞同一類別的 POI，看它在 latent space 上怎麼移動。

假設是「城市 POI 已經飽和，硬塞會過飽和 -> latent 出現離群」，
這支就是直接把那件事做出來：
  base patch = robust 距離最小的 patch（latent 正中央）
  每步在半徑內隨機灑 k 個點、重新挑每格代表、重新 encode，畫出軌跡

主角是 ADD_CAT，另外把 10 類都各跑一次當對照。

v1 特有的限制：矩陣只有 40x40 = 1600 格，而且每格只留一個代表，
所以塞到某個量之後整張圖被新類別佔滿，輸入就不會再變，軌跡會停住。
「軌跡在哪個點數飽和」本身就是這一版的觀察重點。
新灑的點是跟 base 的「代表點」搶格子（不是跟原始的每一個 POI 搶），
所以對高密度格子來說，新點搶贏的機率比訓練時的抽樣稍高一點。
"""

import os
import sys

import numpy as np
import torch
import matplotlib as mpl
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(f"{os.path.dirname(__file__)}/../../.."))
from ae import CELL, GRID, N_CAT, HALF_WIDTH, ConvAE, Patches, mse_loss  # noqa: E402
from config.dataset import CAT_COLORS, CAT_ZH, PATCHES, result  # noqa: E402

LATENTS = result("v1_masked", "latents.npz")
CKPT = result("v1_masked", "ae.pt")
OUT = result("v1_masked", "make_outlier.png")

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


def render(base_cell, base_cat, add_cell, cat_id, pri_base, pri_add):
    """base 代表點 + 新灑的點，每格依優先度留一個，回傳 (1,40,40) 類別編號矩陣。"""
    cell = np.concatenate([base_cell, add_cell])
    cat = np.concatenate([base_cat,
                          np.full(len(add_cell), cat_id, dtype=base_cat.dtype)])
    pri = np.concatenate([pri_base, pri_add])
    order = np.lexsort((pri, cell))
    c = cell[order]
    keep = order[np.append(c[1:] != c[:-1], True)]
    g = np.zeros(GRID * GRID, dtype=np.int64)
    g[cell[keep]] = cat[keep] + 1
    return torch.from_numpy(g).view(1, GRID, GRID)


def sample_disk(rng, k):
    """在半徑 HALF_WIDTH 的圓內均勻灑 k 個點，回傳格子編號。"""
    r = HALF_WIDTH * np.sqrt(rng.random(k))
    t = rng.random(k) * 2 * np.pi
    x, y = r * np.cos(t), r * np.sin(t)
    ix = np.clip(np.floor(x / CELL + GRID / 2), 0, GRID - 1).astype(np.int64)
    iy = np.clip(np.floor(y / CELL + GRID / 2), 0, GRID - 1).astype(np.int64)
    return iy * GRID + ix


def run_cat(model, base, cat_id, rng):
    """對某一類跑完整條軌跡，回傳 (steps, REPEATS, ...) 的 z、MSE、佔用格數。"""
    base_cell, base_cat = base
    z = np.zeros((len(ADD_STEPS), REPEATS, LATENT_DIM))
    err = np.zeros((len(ADD_STEPS), REPEATS))
    occ = np.zeros((len(ADD_STEPS), REPEATS))
    for r in range(REPEATS):
        add_cell = sample_disk(rng, ADD_STEPS[-1])   # 同一串點逐步加入
        pri_base = rng.random(len(base_cell))
        pri_add = rng.random(ADD_STEPS[-1])
        for j, k in enumerate(ADD_STEPS):
            g = render(base_cell, base_cat, add_cell[:k], cat_id,
                       pri_base, pri_add[:k])
            with torch.no_grad():
                x, zi, recon = model(g)
            z[j, r] = zi[0].numpy()
            err[j, r] = mse_loss(recon, x, g).item()
            occ[j, r] = int((g > 0).sum())
    return z, err, occ


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
    data = Patches(PATCHES)
    z_all, n_poi = d["z"], d["n_poi"]

    dist_all = robust_distance(z_all, z_all)
    bi = int(np.argmin(dist_all))
    s, e = data.offsets[bi], data.offsets[bi + 1]
    base = (data.cell[s:e].numpy(), data.cat[s:e].numpy())
    thr = np.percentile(dist_all, OUTLIER_PCT)

    print(f"base patch {bi}：POI {n_poi[bi]}（佔 {len(base[0])} 格），"
          f"robust 距離 {dist_all[bi]:.3f}，"
          f"({d['lat'][bi]:.5f}, {d['lon'][bi]:.5f})")
    print(f"主角類別 {CAT_ZH[ADD_CAT]}（channel {ADD_CAT}），"
          f"代表點裡原本有 {(base[1] == ADD_CAT).sum()} 個")
    print(f"離群門檻（全體 {OUTLIER_PCT}%）= {thr:.2f}\n")

    model = ConvAE(LATENT_DIM)
    model.load_state_dict(torch.load(CKPT, map_location="cpu"))
    model.eval()

    rng = np.random.default_rng(SEED)
    trajs, errs, occs = [], [], []
    for c in range(N_CAT):
        z, err, occ = run_cat(model, base, c, rng)
        trajs.append(z)
        errs.append(err)
        occs.append(occ)
        dist = robust_distance(z.mean(1), z_all)
        cross = [k for j, k in enumerate(ADD_STEPS) if dist[j] > thr]
        print(f"  {CAT_ZH[c]:<6} 加 {ADD_STEPS[-1]} 個後 "
              f"z=({z.mean(1)[-1, 0]:+6.2f}, {z.mean(1)[-1, 1]:+6.2f})  "
              f"robust 距離 {dist[0]:.2f} -> {dist[-1]:.2f}  "
              f"佔用格 {occ.mean(1)[0]:.0f} -> {occ.mean(1)[-1]:.0f}  "
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

    # 佔用格數畫在 MSE 那張的第二軸，看軌跡停住是不是因為格子塞滿了
    e2 = e.twinx()
    e2.plot(ADD_STEPS, occs[ADD_CAT].mean(1), c="#555", ls="--", lw=1.2,
            label="佔用格數")
    e2.axhline(GRID * GRID, c="#999", ls=":", lw=1)
    e2.set_ylabel(f"佔用格數（滿 = {GRID * GRID}）", fontsize=8)
    e2.tick_params(labelsize=7)
    e2.legend(fontsize=7, loc="center right", framealpha=0.9)

    e.set_xlabel("加入的點數", fontsize=8)
    e.set_ylabel("MSE", fontsize=8)
    e.set_title("重建誤差與飽和程度", fontsize=10)
    e.legend(fontsize=7, ncol=2, framealpha=0.9)
    e.tick_params(labelsize=7)
    e.grid(alpha=0.15, linewidth=0.5)

    fig.suptitle(f"v1_masked 人工過飽和：patch {bi}（原始 POI {n_poi[bi]}，"
                 f"latent 正中央）逐步加入單一類別，每點重複 {REPEATS} 次",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT, bbox_inches="tight")
    print(f"\n已存 {OUT}")


main()
