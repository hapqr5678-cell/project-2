"""注入實驗：在指定地點灑進大量餐廳，看 latent space 有什麼反應。

直接驗證專題假設「城市 POI 已飽和，再加點會過飽和 -> latent 出現離群值」。
如果假設成立，注入量越大，latent 應該單調地離開真實資料的分佈。

同時量測「旋轉噪音底線」：同一個 patch 隨機旋轉後 latent 本來就會抖動，
注入造成的位移若小於這個抖動，就代表訊號其實是噪音，假設不成立。

從專案根目錄執行：  .venv/bin/python scratch/3.py
"""

import os
import sys

import matplotlib
import numpy as np
import torch
from pyproj import Transformer
from scipy.spatial import cKDTree

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

matplotlib.rcParams["font.family"] = "Arial Unicode MS"
matplotlib.rcParams["axes.unicode_minus"] = False

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.abspath(__file__)), os.pardir, "model"))
from ae import GRID, N_CAT, ConvAE, circle_mask, poisson_loss  # noqa: E402

PATCHES = "data/patches.npz"
LATENTS = "data/latents.npz"
WEIGHTS = "model/ae2.pt"
FIG = "scratch/inject.png"

CRS = "EPSG:6677"
CELL = 15.0
RADIUS = 300.0
LATENT_DIM = 2
DINING = 0              # "Dining and Drinking" 在 CATEGORIES 裡的 channel

# 要做實驗的地點（名稱, lat, lon）；會自動抓最近的 patch 中心
TARGETS = [
    ("新宿站東口（已極飽和）", 35.6917, 139.7036),
    ("東京站丸之內（商辦）", 35.6812, 139.7671),
    ("練馬住宅區（低密度）", 35.7357, 139.6517),
]

INJECT_COUNTS = [0, 10, 25, 50, 100, 200, 400, 800]
INJECT_RADIUS = 100.0   # 新餐廳灑在距 patch 中心多遠的圓內
ROT_TRIALS = 16         # 量旋轉噪音底線用的隨機旋轉次數
KNN = 50                # 離群指標：到第 KNN 個真實 patch latent 的距離
SEED = 0

device = ("mps" if torch.backends.mps.is_available()
          else "cuda" if torch.cuda.is_available() else "cpu")


def render(dx, dy, cat):
    """點列 -> (counts (1,10,40,40), x (1,11,40,40))，與 train.py 的 render 一致。"""
    ix = np.clip(np.floor(dx / CELL + GRID / 2).astype(int), 0, GRID - 1)
    iy = np.clip(np.floor(dy / CELL + GRID / 2).astype(int), 0, GRID - 1)
    counts = np.zeros((N_CAT, GRID, GRID), dtype=np.float32)
    np.add.at(counts, (cat, iy, ix), 1.0)
    counts = torch.from_numpy(counts)[None]
    occ = (counts.sum(1, keepdim=True) > 0).float()
    return counts, torch.cat([torch.log1p(counts), occ], dim=1)


def spin(dx, dy, theta, flip):
    """繞中心旋轉 theta，flip 時再左右鏡射（與訓練時的增強相同）。"""
    cos, sin = np.cos(theta), np.sin(theta)
    rx, ry = dx * cos - dy * sin, dx * sin + dy * cos
    return (-rx if flip else rx), ry


def scatter_dining(rng, n):
    """在中心附近的圓內均勻灑 n 家餐廳，回傳相對座標（公尺）。"""
    r = INJECT_RADIUS * np.sqrt(rng.random(n))
    a = rng.random(n) * 2 * np.pi
    return r * np.cos(a), r * np.sin(a)


def oracle_nll(counts, cells):
    """基線：已知本 patch 每類真實總數，但空間完全均勻鋪開時的 Poisson NLL。"""
    per = counts.sum(dim=(2, 3)).numpy()[0]
    rate = per / cells
    log_rate = np.where(rate > 0, np.log(np.maximum(rate, 1e-12)), -8.0)
    return float((rate * cells - per * log_rate).sum() / cells / N_CAT)


def encode(model, mask, counts, x):
    """回傳 (latent, 重建誤差, AE 解碼出的期望 POI 總數)。"""
    with torch.no_grad():
        z, log_rate = model(x.to(device))
        err = poisson_loss(log_rate, counts.to(device), mask)
        pred = (torch.exp(log_rate) * mask).sum(dim=(1, 2, 3))
    return z[0].cpu().numpy(), float(err[0]), float(pred[0])


def main():
    p = np.load(PATCHES)
    z_all = np.load(LATENTS)["z2"]
    dx_all, dy_all, cat_all = p["dx"], p["dy"], p["cat"].astype(np.int64)
    off, cx, cy = p["offsets"], p["center_x"], p["center_y"]

    model = ConvAE(LATENT_DIM).to(device)
    model.load_state_dict(torch.load(WEIGHTS, map_location=device))
    model.eval()
    mask = circle_mask(RADIUS, CELL, device)
    cells = float(mask.sum())

    # 真實 latent 的分佈，用來判斷「注入後跑出去多遠算離群」
    mu, cov = z_all.mean(0), np.cov(z_all.T)
    inv_cov = np.linalg.inv(cov)
    ztree = cKDTree(z_all)
    bg_knn = ztree.query(z_all, k=KNN + 1)[0][:, KNN]
    knn_p99 = np.percentile(bg_knn, 99)

    def maha(z):
        d = z - mu
        return float(np.sqrt(d @ inv_cov @ d))

    print(f"device={device}  權重={WEIGHTS}")
    print(f"真實 latent 的 {KNN}-NN 距離：中位數 {np.median(bg_knn):.3f}  "
          f"p99 {knn_p99:.3f}  max {bg_knn.max():.3f}")

    fwd = Transformer.from_crs("EPSG:4326", CRS, always_xy=True)
    ctree = cKDTree(np.column_stack([cx, cy]))
    results = []

    for name, lat, lon in TARGETS:
        tx, ty = fwd.transform(lon, lat)
        dist, i = ctree.query([tx, ty])
        a, b = off[i], off[i + 1]
        bdx, bdy, bcat = dx_all[a:b].astype(np.float64), \
            dy_all[a:b].astype(np.float64), cat_all[a:b]
        n_base = len(bcat)
        n_dine = int((bcat == DINING).sum())

        print(f"\n{'=' * 78}")
        print(f"{name}   最近 patch 距離 {dist:.0f} m   "
              f"({p['center_lat'][i]:.4f}, {p['center_lon'][i]:.4f})")
        print(f"  原有 POI {n_base} 個，其中餐廳 {n_dine} 個 "
              f"（{n_dine / n_base:.0%}）")

        # 噪音底線：同一個 patch 不動，只隨機旋轉，latent 本身抖多少
        rng = np.random.default_rng(SEED)
        zr = []
        for _ in range(ROT_TRIALS):
            rx, ry = spin(bdx, bdy, rng.random() * 2 * np.pi, rng.random() < .5)
            zr.append(encode(model, mask, *render(rx, ry, bcat))[0])
        zr = np.array(zr)
        noise = float(np.linalg.norm(zr - zr.mean(0), axis=1).mean())
        print(f"  旋轉噪音底線：latent 平均抖動 {noise:.3f}"
              f"（位移小於這個數字就等於沒訊號）")

        rng = np.random.default_rng(SEED)
        rows = []
        z0 = None
        print(f"\n  {'注入':>5} {'總POI':>6} {'餐廳佔比':>8} "
              f"{'latent':>17} {'位移':>7} {'Maha':>6} {'kNN':>7} "
              f"{'AE預測POI':>9} {'ratio':>6}")
        for k in INJECT_COUNTS:
            jx, jy = scatter_dining(rng, k)
            dx = np.concatenate([bdx, jx])
            dy = np.concatenate([bdy, jy])
            cat = np.concatenate([bcat, np.full(k, DINING, dtype=np.int64)])

            counts, x = render(dx, dy, cat)
            z, err, pred = encode(model, mask, counts, x)
            if z0 is None:
                z0 = z
            shift = float(np.linalg.norm(z - z0))
            knn = float(ztree.query(z, k=KNN)[0][-1])
            ratio = err / oracle_nll(counts, cells)
            total = n_base + k
            rows.append((k, total, z, shift, maha(z), knn, pred, ratio))
            print(f"  {k:5d} {total:6d} {(n_dine + k) / total:8.0%} "
                  f"({z[0]:+7.3f},{z[1]:+7.3f}) {shift:7.3f} "
                  f"{maha(z):6.2f} {knn:7.3f} {pred:9.0f} {ratio:6.3f}")

        results.append((name, noise, rows))

    # ---- 圖 ----
    fig, ax = plt.subplots(2, 2, figsize=(14, 12))
    colors = plt.cm.tab10(np.arange(len(TARGETS)))

    ax[0, 0].scatter(z_all[:, 0], z_all[:, 1], s=2, c="lightgrey", alpha=.5,
                     label="真實 patch")
    for (name, _, rows), c in zip(results, colors):
        zs = np.array([r[2] for r in rows])
        ax[0, 0].plot(zs[:, 0], zs[:, 1], "-o", color=c, ms=4, lw=1.5, label=name)
        for k, _, z, *_ in rows:
            if k in (0, INJECT_COUNTS[-1]):
                ax[0, 0].annotate(f"+{k}", z, fontsize=8, color=c)
    ax[0, 0].legend(fontsize=8)
    ax[0, 0].set(xlabel="z0", ylabel="z1")
    # 軌跡尺度遠小於整體分佈，放大到軌跡附近才看得見
    allz = np.array([r[2] for _, _, rows in results for r in rows])
    pad = 2.0
    ax[0, 0].set_xlim(allz[:, 0].min() - pad, allz[:, 0].max() + pad)
    ax[0, 0].set_ylim(allz[:, 1].min() - pad, allz[:, 1].max() + pad)
    ax[0, 0].set_title("注入餐廳後 latent 的移動軌跡（已放大到軌跡附近）")

    for (name, noise, rows), c in zip(results, colors):
        ks = [r[0] for r in rows]
        ax[0, 1].plot(ks, [r[3] for r in rows], "-o", color=c, ms=4, label=name)
        ax[0, 1].axhline(noise, color=c, ls=":", lw=1)
    ax[0, 1].set(xlabel="注入餐廳數", ylabel="latent 位移")
    ax[0, 1].legend(fontsize=8)
    ax[0, 1].set_title("位移 vs 注入量（虛線=該地點的旋轉噪音底線）")

    for (name, _, rows), c in zip(results, colors):
        ax[1, 0].plot([r[0] for r in rows], [r[5] for r in rows], "-o",
                      color=c, ms=4, label=name)
    ax[1, 0].axhline(knn_p99, color="k", ls="--", lw=1,
                     label=f"真實資料 {KNN}-NN 距離 p99")
    ax[1, 0].set(xlabel="注入餐廳數", ylabel=f"到第 {KNN} 個真實 latent 的距離")
    ax[1, 0].legend(fontsize=8)
    ax[1, 0].set_title("離群程度：超過黑線才算真的跑出真實分佈")

    for (name, _, rows), c in zip(results, colors):
        ax[1, 1].plot([r[1] for r in rows], [r[6] for r in rows], "-o",
                      color=c, ms=4, label=name)
    lim = max(max(r[1] for r in rows) for _, _, rows in results)
    ax[1, 1].plot([0, lim], [0, lim], "k--", lw=1, label="y = x")
    ax[1, 1].set(xlabel="實際 POI 總數", ylabel="AE 解碼出的期望 POI 總數")
    ax[1, 1].legend(fontsize=8)
    ax[1, 1].set_title("AE 認為「這裡該有幾個 POI」：低於對角線=模型認定過飽和")

    plt.tight_layout()
    plt.savefig(FIG, dpi=110)
    print(f"\n圖已存 {FIG}")


main()
