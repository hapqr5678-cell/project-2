"""K=3 的 simplex 是一個正三角形，可以無損地畫成 2D 三角座標圖。
這支放共用的座標轉換與三角框，給 latent_plot_plain.py 和 cluster_plot.py 用。

跟 PCA 投影不同，這個轉換沒有丟掉任何資訊：theta 有 2 個自由度，
三角形也是 2 維，是一對一的。所以圖上看到的形狀就是 latent 真正的形狀，
可以直接跟 v2_vae / v2_gvae 的 latent 散點圖對照。
"""

import numpy as np

# 三角形的三個頂點，對應 theta = (1,0,0)、(0,1,0)、(0,0,1)。
# 置中並縮放到 [-1,1]，好跟其他版本 min-max 正規化過的圖同尺度。
_V = np.array([[-1.0, -np.sqrt(3) / 3],
               [1.0, -np.sqrt(3) / 3],
               [0.0, 2 * np.sqrt(3) / 3]])


def to_xy(theta):
    """把 (N,3) 的 simplex 向量轉成三角座標。

    theta：(N,3)，每列非負且總和 1。
    回傳 (N,2) 的 xy 座標；三個頂點分別落在 a0/a1/a2 的純成分位置。
    """
    return theta @ _V


def frame(ax, labels=("a0", "a1", "a2")):
    """在 ax 上畫三角形外框與三個頂點標籤，並把座標軸關掉。

    ax：matplotlib 的 Axes。
    labels：三個頂點的名稱，依 theta 的維度順序。
    沒有回傳值，就地修改 ax。
    """
    tri = np.vstack([_V, _V[:1]])
    ax.plot(tri[:, 0], tri[:, 1], color="#666", linewidth=1.0, alpha=0.6)
    # 等比例的內部格線：theta_k = 0.25/0.5/0.75 的等值線
    for f in (0.25, 0.5, 0.75):
        for k in range(3):
            i, j = (k + 1) % 3, (k + 2) % 3
            p = np.zeros((2, 3))
            p[0, k] = p[1, k] = f
            p[0, i] = 1 - f
            p[1, j] = 1 - f
            seg = p @ _V
            ax.plot(seg[:, 0], seg[:, 1], color="#999", linewidth=0.4,
                    alpha=0.25, zorder=0)
    off = np.array([[-0.10, -0.12], [0.10, -0.12], [0.0, 0.10]])
    for k in range(3):
        ax.annotate(labels[k], _V[k] + off[k], fontsize=9, color="#444",
                    ha="center", va="center")
    ax.set_xlim(-1.25, 1.25)
    ax.set_ylim(-0.85, 1.35)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")
