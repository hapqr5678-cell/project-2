"""挑一個 patch 丟進訓練好的 v1_masked AE，比較「進去」與「出來」。

三張圖：
  左   原始 POI（半徑 300m 的圓，一點一個，顏色 = 類別）
  中   實際餵給 AE 的輸入：每格只留一個隨機代表，畫在格子中心
  右   AE 重建，只畫有算 loss 的格子：每格是一個 8 維向量，取它跟 10 個類別
       embedding 的內積最大者當「這格 AE 覺得是什麼類」，向量長度當強度

右圖跟著 loss 一起上遮罩。這一版空白格不算分，decoder 在那些位置輸出什麼都不
扣分，畫出來只是雜訊；只畫遮罩內的格子，才是拿模型真正被要求的部分跟輸入對齊。
（想看被排掉的部分長什麼樣，改 SHOW_MASKED_OUT = True，空白格會用灰點淡淡疊上去。）

另外印出這個 patch 的 MSE、它在全體裡的百分位，以及兩個比 MSE 好懂的指標：
  類別正確率    在遮罩內的格子上，重建的最近類別猜對的比例（亂猜 = 1/10）
  佔用格命中率  取重建強度前 N 名的格子（N = 真實佔用格數），有幾成落在真的有
                東西的格子。這一版的 loss 從沒要求過這件事，純粹拿來對照 v1_embeding
"""

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib as mpl
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(f"{os.path.dirname(__file__)}/../../.."))
from ae import GRID, CELL, HALF_WIDTH, ConvAE, Patches, mse_loss  # noqa: E402
from config.result_style import REBUILD_GRID  # noqa: E402
from config.dataset import CAT_COLORS, CAT_ZH, N_CAT, PATCHES, result  # noqa: E402

LATENTS = result("v1_l32_mask", "latents.npz")
CKPT = result("v1_l32_mask", "ae.pt")
OUT = result("v1_l32_mask", "rebuild_test.png")


SHOW_MASKED_OUT = False   # 右圖要不要把被遮罩排掉的空白格用灰點疊上去
DOT = 18          # 點的基本大小
IN_COLOR = "#2c7fb8"
MID_COLOR = "#6a51a3"
OUT_COLOR = "#c0392b"

mpl.rcParams["font.family"] = ["Heiti TC"]
mpl.rcParams["axes.unicode_minus"] = False
mpl.rcParams["figure.dpi"] = 130


def cell_xy():
    """40x40 格子中心換算回公尺。"""
    gy, gx = np.mgrid[0:GRID, 0:GRID]
    return (gx + 0.5 - GRID / 2) * CELL, (gy + 0.5 - GRID / 2) * CELL


def style(ax, title, edge):
    ax.add_patch(plt.Circle((0, 0), HALF_WIDTH, fill=False, lw=1.6,
                            color=edge, alpha=0.8))
    ax.set_xlim(-HALF_WIDTH * 1.05, HALF_WIDTH * 1.05)
    ax.set_ylim(-HALF_WIDTH * 1.05, HALF_WIDTH * 1.05)
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=10, color=edge)
    ax.tick_params(labelsize=7)
    ax.grid(alpha=0.15, linewidth=0.5)
    if REBUILD_GRID:
        import numpy as _np
        ticks = _np.arange(-GRID // 2, GRID // 2 + 1) * CELL
        for t in ticks:
            ax.axhline(t, color='#888', lw=0.3, alpha=0.35)
            ax.axvline(t, color='#888', lw=0.3, alpha=0.35)
    for s in ax.spines.values():
        s.set_alpha(0.3)


def draw_true(ax, p, i, title):
    """原始 POI，一點一個。"""
    s, e = p["offsets"][i], p["offsets"][i + 1]
    dx, dy, cat = p["dx"][s:e], p["dy"][s:e], p["cat"][s:e].astype(np.int64)
    for k in range(N_CAT):
        m = cat == k
        if m.any():
            ax.scatter(dx[m], dy[m], s=DOT, c=CAT_COLORS[k], linewidths=0,
                       alpha=0.85, label=CAT_ZH[k])
    style(ax, title, IN_COLOR)


def draw_grid(ax, g, title, edge):
    """實際輸入：每個非空格畫一點在格子中心，顏色 = 類別。"""
    x, y = cell_xy()
    g = g.numpy()
    for k in range(N_CAT):
        m = g == k + 1
        if m.any():
            ax.scatter(x[m], y[m], s=DOT, c=CAT_COLORS[k], linewidths=0,
                       alpha=0.85, label=CAT_ZH[k])
    style(ax, title, edge)


def draw_recon(ax, score, inten, occ, title):
    """重建：顏色 = 最近的類別，大小/透明度 ∝ 向量長度。只畫遮罩內的格子。"""
    x, y = cell_xy()
    top_cat = score.argmax(0)
    rel = inten / inten[occ].max()
    if SHOW_MASKED_OUT:
        ax.scatter(x[~occ], y[~occ], s=DOT * 0.25, c="#c8ccd2", linewidths=0,
                   alpha=0.35, label="被遮罩排掉（不算 loss）")
    for k in range(N_CAT):
        m = occ & (top_cat == k)
        if m.any():
            ax.scatter(x[m], y[m], s=DOT * (0.15 + 1.5 * rel[m]),
                       c=CAT_COLORS[k], linewidths=0,
                       alpha=np.clip(rel[m], 0, 1) * 0.9, label=CAT_ZH[k])
    style(ax, title, OUT_COLOR)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=0, help="patch 編號（0 起算）")
    n = ap.parse_args().n

    data = Patches(PATCHES)
    assert 0 <= n < data.n, f"--n 要在 0~{data.n - 1}"

    sd = torch.load(CKPT, map_location="cpu")
    LATENT_DIM = sd["encoder.6.weight"].shape[0]
    model = ConvAE(LATENT_DIM)
    model.load_state_dict(sd)
    model.eval()

    g = data.render(torch.tensor([n]))
    with torch.no_grad():
        x, z, recon = model(g)
        emb = F.normalize(model.emb.weight, dim=1)          # (10, 8)
    loss = mse_loss(recon, x, g).item()

    # 每格：跟 10 類 embedding 的內積（誰最像）、以及向量長度（多確定有東西）
    score = torch.einsum("ce,ehw->chw", emb, recon[0]).numpy()   # (10,40,40)
    inten = recon[0].pow(2).sum(0).sqrt().numpy()               # (40,40)

    err = np.load(LATENTS)["err"]
    pct = (err < loss).mean() * 100
    n_poi = int(data.n_poi[n])
    n_occ = int(data.n_occupied[n])

    true_g = g[0].numpy()
    occ = true_g > 0
    true_cat = true_g - 1

    # 佔用格命中率：重建強度前 N 名有幾成落在真的有東西的格子
    top = np.argsort(inten, axis=None)[::-1][:n_occ]
    hit = occ.ravel()[top].mean()
    # 類別正確率：只看真的有東西的格子
    acc = (score.argmax(0)[occ] == true_cat[occ]).mean()

    print(f"patch {n}：POI {n_poi} 個，去重後佔 {n_occ} 格  "
          f"({data.lat[n]:.5f}, {data.lon[n]:.5f})")
    print(f"latent z = ({z[0, 0]:+.3f}, {z[0, 1]:+.3f})")
    print(f"MSE = {loss:.6f}  "
          f"（全體中位數 {np.median(err):.6f}，此 patch 排在第 {pct:.1f} 百分位）")
    print(f"類別正確率 {acc:.1%}（亂猜 {1 / N_CAT:.1%}）  "
          f"佔用格命中率 {hit:.1%}（亂猜 {n_occ / (GRID * GRID):.1%}，"
          f"這一版沒被 loss 要求，僅供對照）")

    print("\n逐類別：輸入格數 -> 重建判給這一類的格數（都只算遮罩內的格子）")
    pred_cat = score.argmax(0)
    for c, name in enumerate(CAT_ZH):
        print(f"  {name:<6}輸入 {int((true_cat == c)[occ].sum()):4d} 格 -> "
              f"重建 {int((pred_cat == c)[occ].sum()):4d} 格")

    fig, (a, b, c) = plt.subplots(1, 3, figsize=(18, 6.5))
    draw_true(a, np.load(PATCHES), n, f"原始 POI（共 {n_poi} 個）")
    draw_grid(b, g[0], f"AE 前：實際輸入（{n_occ} 格，每格一個代表）", MID_COLOR)
    draw_recon(c, score, inten, occ,
               f"AE 後：只畫遮罩內的 {n_occ} 格（MSE {loss:.6f}）")
    a.legend(fontsize=6.5, markerscale=1.4, framealpha=0.9,
             loc="upper left", bbox_to_anchor=(0.0, -0.06), ncol=3)

    print(f"\n重建向量長度：最大 {inten.max():.3f}，"
          f"真實佔用格平均 {inten[occ].mean():.3f}，空白格平均 {inten[~occ].mean():.3f}")

    fig.suptitle(f"v1_l32_mask patch {n}（latent_dim={LATENT_DIM}）", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT, bbox_inches="tight")
    print(f"已存 {OUT}")


main()
