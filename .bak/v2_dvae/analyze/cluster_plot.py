"""看 v2_dvae 的 Dirichlet 先驗到底把 patch 分成了什麼。

K=3，所以 latent 直接畫成三角座標圖，沒有任何投影損失（見 ternary.py），
可以跟 v2_gvae 的同名圖一張對一張比。

六張圖：
  1. 三角形上的主 archetype 上色（argmax theta）
  2. 主 archetype 的佔比 max theta（越低代表這個 patch 混得越均勻）
  3. 各 archetype 的平均類別組成（archetype 到底對應什麼樣的地區）
  4. 各 archetype 的 POI 數分布（分的是「型態」還是只是「密度」）
  5. archetype 使用率：整體平均 theta_k（有分量接近 0 就是 collapse）
  6. 主 archetype 的地理分布

第 1 張要小心讀：三色分區是 argmax 沿著三條中線切出來的，不管資料長怎樣
都會是這個形狀，不是學到的結構。真正該看的是第 2 張——點有沒有往角落
聚集（max theta 高），那才是先驗有沒有在推動的證據。

第 4 張跟 v2_gvae 一樣是最該看的檢查：如果三個 archetype 的 POI 數 box
幾乎完全分層、彼此不重疊，代表模型只是把密度切成三段。
第 5 張是 Dirichlet 版特有的：稀疏先驗（alpha < 1）最典型的失敗就是
只有一兩個 archetype 活著，其餘的平均 theta 掉到 0。
"""

import os
import sys

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.abspath(f"{os.path.dirname(__file__)}/../../.."))
from ternary import frame, to_xy  # noqa: E402
from config.dataset import CAT_ZH, N_CAT, PATCHES, result  # noqa: E402

VERSION = "v2_dvae"
LATENTS = result(VERSION, "latents.npz")
OUT = result(VERSION, "cluster_plot.png")

DOT = 3.0
CLUSTER_COLORS = ["#e6194b", "#3cb44b", "#4363d8", "#f58231",
                  "#911eb4", "#008080", "#f032e6", "#9a6324"]

mpl.rcParams["font.family"] = ["Heiti TC"]
mpl.rcParams["axes.unicode_minus"] = False
mpl.rcParams["figure.dpi"] = 130


def composition(cluster, k):
    """每個 archetype 的平均類別組成，回傳 (K,N_CAT) 的比例矩陣（每列總和 1）。
    cluster：(N,) 的硬分群結果（argmax theta）。k：archetype 數。
    """
    p = np.load(PATCHES)
    owner = np.repeat(np.arange(len(p["n_poi"])), p["n_poi"])
    counts = np.zeros((k, N_CAT))
    for c in range(k):
        m = np.isin(owner, np.flatnonzero(cluster == c))
        counts[c] = np.bincount(p["cat"][m], minlength=N_CAT)
    total = counts.sum(1, keepdims=True)
    return np.divide(counts, total, out=np.zeros_like(counts), where=total > 0)


def main():
    d = np.load(LATENTS)
    theta, cluster, conf = d["z"], d["cluster"], d["conf"]
    n_poi, lat, lon = d["n_poi"], d["lat"], d["lon"]
    k, alpha = int(d["k"]), float(d["alpha"])
    if k != 3:
        raise SystemExit(f"這張圖只支援 K=3，latents.npz 的 K={k}")
    xy = to_xy(theta)

    print(f"K={k}，alpha={alpha:.3f}，主 archetype 佔比中位數 "
          f"{np.median(conf):.3f}")

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    (a, b, c), (e, f, g) = axes

    # 1) 三角形上的主 archetype
    frame(a)
    for i in range(k):
        m = cluster == i
        a.scatter(xy[m, 0], xy[m, 1], s=DOT,
                  c=CLUSTER_COLORS[i % len(CLUSTER_COLORS)], linewidths=0,
                  alpha=0.55, rasterized=True, zorder=2,
                  label=f"a{i} ({m.sum()}, theta={theta[:, i].mean():.2f})")
    a.legend(fontsize=7, markerscale=2, framealpha=0.9, loc="upper left")
    a.set_title("主 archetype（argmax theta）於 2-simplex", fontsize=10)

    # 2) 主 archetype 的佔比
    frame(b)
    sc = b.scatter(xy[:, 0], xy[:, 1], c=conf, s=DOT, cmap="magma",
                   linewidths=0, alpha=0.75, rasterized=True, zorder=2,
                   vmin=1 / k, vmax=1)
    cb = fig.colorbar(sc, ax=b, fraction=0.046, pad=0.02)
    cb.set_label("max theta", fontsize=8)
    cb.ax.tick_params(labelsize=7)
    b.set_title("主 archetype 佔比（低 = 混合得很均勻）", fontsize=10)

    # 3) 各 archetype 的平均類別組成
    comp = composition(cluster, k)
    bottom = np.zeros(k)
    xs = np.arange(k)
    for j in range(N_CAT):
        c.bar(xs, comp[:, j], bottom=bottom, width=0.6, label=CAT_ZH[j])
        bottom += comp[:, j]
    c.set_xticks(xs)
    c.set_xticklabels([f"a{i}" for i in range(k)], fontsize=8)
    c.legend(fontsize=6, ncol=2, loc="upper right", framealpha=0.9)
    c.set_ylim(0, 1.35)
    c.set_title("各 archetype 的平均類別組成", fontsize=10)
    c.set_ylabel("比例", fontsize=8)
    c.tick_params(labelsize=7)

    # 4) 各 archetype 的 POI 數分布
    groups = [n_poi[cluster == i] for i in range(k)]
    e.boxplot([grp if len(grp) else [np.nan] for grp in groups],
              tick_labels=[f"a{i}" for i in range(k)], showfliers=False)
    e.set_yscale("log")
    e.set_title("各 archetype 的 POI 數（分層 = 只在切密度）", fontsize=10)
    e.set_ylabel("POI 數", fontsize=8)
    e.tick_params(labelsize=7)
    e.grid(alpha=0.15, linewidth=0.5)

    # 5) archetype 使用率：整體平均 theta_k
    share = theta.mean(0)
    cols = [CLUSTER_COLORS[i % len(CLUSTER_COLORS)] for i in range(k)]
    f.bar(xs, share, width=0.6, color=cols)
    f.axhline(1 / k, color="#555", linestyle="--", linewidth=1,
              label=f"均勻使用 (1/K={1 / k:.2f})")
    f.set_xticks(xs)
    f.set_xticklabels([f"a{i}" for i in range(k)], fontsize=8)
    f.legend(fontsize=7, framealpha=0.9)
    f.set_title("archetype 使用率（接近 0 = collapse）", fontsize=10)
    f.set_ylabel("平均 theta", fontsize=8)
    f.tick_params(labelsize=7)
    f.grid(alpha=0.15, linewidth=0.5, axis="y")

    # 6) 主 archetype 的地理分布
    for i in range(k):
        m = cluster == i
        if not m.any():
            continue
        g.scatter(lon[m], lat[m], s=1.5,
                  c=CLUSTER_COLORS[i % len(CLUSTER_COLORS)],
                  linewidths=0, alpha=0.6, rasterized=True, label=f"a{i}")
    g.legend(fontsize=7, markerscale=4, framealpha=0.9)
    g.set_aspect(1 / np.cos(np.deg2rad(float(lat.mean()))))
    g.set_title("主 archetype 的地理分布", fontsize=10)
    g.set_xlabel("經度", fontsize=8)
    g.set_ylabel("緯度", fontsize=8)
    g.tick_params(labelsize=7)
    g.grid(alpha=0.15, linewidth=0.5)
    for s in g.spines.values():
        s.set_alpha(0.3)

    fig.suptitle(f"{VERSION} Dirichlet-VAE 分群（{len(theta)} 個 patch，"
                 f"K={k}（自由度 2，對齊 v2_vae 的 latent_dim=2），"
                 f"alpha={alpha:.3f}，Poisson NLL + BETA·Dirichlet KL）",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT, bbox_inches="tight")
    print(f"已存 {OUT}")


main()
