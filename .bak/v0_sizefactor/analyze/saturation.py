"""飽和度＝條件式的密度殘差：給定這個 patch 的形狀，它的 POI 數異常嗎。

為什麼要換一套判定：在 v0 / v0_poisson_nll 上，「latent 離群」同時混著
兩種東西——密度特別大、以及形狀特別怪——因為密度本身就佔了 latent 的一整維
（v0 的 R²密度 = 0.934）。分不出來的話，「過飽和」這個結論就沒有內容，
它可能只是在說「這裡 POI 很多」，那不用 autoencoder 也知道。

v0_sizefactor 把密度從 latent 拿掉之後，兩者才分得開：
    z            = 形狀（組成 + 空間排列），跟總量無關
    log n_poi    = 總量，獨立的一個純量
於是飽和度可以問得很乾淨：

    在 latent 上找 k 個形狀最像的鄰居，看它們的 log n_poi 分布，
    這個 patch 落在分布的哪裡。

    residual = log n_poi - mean(鄰居的 log n_poi)
    z-score  = residual / std(鄰居的 log n_poi)

z-score 高 = 同樣長相的地方通常沒這麼多 POI -> 過飽和
z-score 低 = 有這種結構卻還很空 -> 未飽和（可開發）

用 z-score 而不是原始 residual，是因為不同形狀區域的密度變異天差地遠：
車站周邊本來就散得很開，用同一把尺量會全部判成過飽和。

已知風險：
  * 鄰居是在 2 維 latent 上找的，形狀「像」的定義完全由模型決定。
    latent 若塌成一條線，鄰居就沒有意義，要先看 latent_plot 的散布。
  * k=50 在 23700 個 patch 上約佔 0.2%，但 latent 邊緣的點鄰居會被拉得很遠，
    那裡的 z-score 不可信。所以圖上另外標出「鄰居平均距離」最大的那批。
  * patch 中心是規則格點（CENTER_STEP=50m），相鄰 patch 高度重疊，
    kNN 很可能挑到地理上的鄰居而不是形狀上的同類。這會讓 z-score 偏保守
    （自己跟自己比）。要根治得在取鄰居時排除地理距離太近的，這一版沒做。
"""

import os
import sys

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.abspath(f"{os.path.dirname(__file__)}/../../.."))
from config.dataset import result  # noqa: E402

LATENTS = result("v0_sizefactor", "latents.npz")
OUT = result("v0_sizefactor", "saturation.png")

KNN = 50
TOP_PCT = 99.0      # z-score 超過這個百分位算過飽和
BOT_PCT = 1.0       # 低於這個百分位算未飽和
EDGE_PCT = 95.0     # 鄰居平均距離超過這個百分位視為 latent 邊緣、不可信
DOT = 3.0

mpl.rcParams["font.family"] = ["Heiti TC"]
mpl.rcParams["axes.unicode_minus"] = False
mpl.rcParams["figure.dpi"] = 130


def style(ax, title):
    ax.set_title(title, fontsize=10)
    ax.tick_params(labelsize=7)
    ax.grid(alpha=0.15, linewidth=0.5)
    for s in ax.spines.values():
        s.set_alpha(0.3)


def main():
    d = np.load(LATENTS)
    z, n_poi = d["z"], d["n_poi"].astype(float)
    lat, lon = d["lat"], d["lon"]
    y = np.log(n_poi)

    zs = (z - z.mean(0)) / z.std(0)
    dist, idx = cKDTree(zs).query(zs, k=KNN + 1)
    nb = idx[:, 1:]                      # 排除自己
    mu = y[nb].mean(1)
    sd = y[nb].std(1)
    resid = y - mu
    score = resid / np.maximum(sd, 1e-6)
    edge = dist[:, 1:].mean(1)

    ok = edge < np.percentile(edge, EDGE_PCT)
    hi = (score > np.percentile(score[ok], TOP_PCT)) & ok
    lo = (score < np.percentile(score[ok], BOT_PCT)) & ok

    print(f"{len(z)} 個 patch，k={KNN}")
    print(f"形狀鄰居能解釋多少密度：R² = "
          f"{1 - np.var(y - mu) / np.var(y):+.3f}")
    print(f"  （這個數字高不代表 latent 編碼了密度，而是形狀本身就跟密度相關，"
          f"例如商店街必然密）")
    print(f"z-score: 中位數 {np.median(score):+.3f}  "
          f"IQR [{np.percentile(score, 25):+.2f}, "
          f"{np.percentile(score, 75):+.2f}]  "
          f"範圍 [{score.min():+.2f}, {score.max():+.2f}]")
    print()
    print(f"過飽和 {hi.sum()} 個（z>{np.percentile(score[ok], TOP_PCT):.2f}）"
          f"：POI 中位數 {np.median(n_poi[hi]):.0f}，"
          f"同型鄰居中位數 {np.median(np.exp(mu[hi])):.0f}")
    print(f"未飽和 {lo.sum()} 個（z<{np.percentile(score[ok], BOT_PCT):.2f}）"
          f"：POI 中位數 {np.median(n_poi[lo]):.0f}，"
          f"同型鄰居中位數 {np.median(np.exp(mu[lo])):.0f}")
    print(f"（{(~ok).sum()} 個 latent 邊緣 patch 已排除）")
    print()
    print("最過飽和的 10 個：")
    for i in np.argsort(-np.where(ok, score, -np.inf))[:10]:
        print(f"  patch {i:6d}  ({lat[i]:.5f}, {lon[i]:.5f})  "
              f"POI {n_poi[i]:5.0f}  同型鄰居 {np.exp(mu[i]):6.1f}  "
              f"z = {score[i]:+.2f}")

    fig, ((a, b), (c, e)) = plt.subplots(2, 2, figsize=(13, 11))

    v = np.percentile(np.abs(score), 99)
    sc = a.scatter(z[:, 0], z[:, 1], c=score, s=DOT, cmap="coolwarm",
                   linewidths=0, alpha=0.7, rasterized=True, vmin=-v, vmax=v)
    cb = fig.colorbar(sc, ax=a, fraction=0.046, pad=0.02)
    cb.set_label("飽和度 z-score", fontsize=8)
    cb.ax.tick_params(labelsize=7)
    a.set_xlabel("z1", fontsize=8)
    a.set_ylabel("z2", fontsize=8)
    style(a, "latent（形狀）上的飽和度：紅=過飽和 藍=未飽和")

    b.scatter(mu[ok], y[ok], s=DOT, c="#b8bcc4", linewidths=0, alpha=0.4,
              rasterized=True, label="全體")
    b.scatter(mu[hi], y[hi], s=DOT * 4, c="#c0392b", linewidths=0, alpha=0.85,
              rasterized=True, label=f"過飽和 (n={hi.sum()})")
    b.scatter(mu[lo], y[lo], s=DOT * 4, c="#2471a3", linewidths=0, alpha=0.85,
              rasterized=True, label=f"未飽和 (n={lo.sum()})")
    lim = [y.min(), y.max()]
    b.plot(lim, lim, c="#555", ls="--", lw=1, label="y = x")
    b.set_xlabel("同型鄰居的 log(POI 數) 平均", fontsize=8)
    b.set_ylabel("自己的 log(POI 數)", fontsize=8)
    b.legend(fontsize=7.5, markerscale=2, framealpha=0.9)
    style(b, "實際密度 vs 形狀所預期的密度")

    c.hist(score[ok], bins=120, color="#7f8c8d", alpha=0.85)
    c.axvline(np.percentile(score[ok], TOP_PCT), c="#c0392b", ls="--", lw=1,
              label=f"過飽和門檻 {TOP_PCT}%")
    c.axvline(np.percentile(score[ok], BOT_PCT), c="#2471a3", ls="--", lw=1,
              label=f"未飽和門檻 {BOT_PCT}%")
    c.set_xlabel("飽和度 z-score", fontsize=8)
    c.set_ylabel("patch 數", fontsize=8)
    c.set_yscale("log")
    c.legend(fontsize=7.5, framealpha=0.9)
    style(c, "飽和度分布")

    e.scatter(lon, lat, s=1.5, c="#d5d8dc", linewidths=0, alpha=0.5,
              rasterized=True)
    e.scatter(lon[lo], lat[lo], s=9, c="#2471a3", linewidths=0, alpha=0.85,
              rasterized=True, label="未飽和")
    e.scatter(lon[hi], lat[hi], s=9, c="#c0392b", linewidths=0, alpha=0.9,
              rasterized=True, label="過飽和")
    e.set_aspect(1 / np.cos(np.deg2rad(float(lat.mean()))))
    e.set_xlabel("經度", fontsize=8)
    e.set_ylabel("緯度", fontsize=8)
    e.legend(fontsize=7.5, markerscale=2, framealpha=0.9)
    style(e, "地理分布")

    fig.suptitle("v0_sizefactor 飽和度：latent 只管形狀，密度單獨比較\n"
                 f"（{len(z)} 個 patch，k={KNN} 個形狀鄰居）", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT, bbox_inches="tight")
    print(f"\n已存 {OUT}")


main()
