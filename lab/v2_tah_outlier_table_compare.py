"""把 v2_tah_outlier_table_experience.py 存的三個版本 csv 讀進來，畫成一張圖
比較 v2_ae_tanh / v2_vae_tanh / v2_tanh_perceiver 三者的差異：同一個類別、
同一個加入點數下，誰移動得比較多（不穩定性）、誰的重建損失提升得比較多。

兩個指標尺度差很多，分開兩列畫（不共用 y 軸），每欄是一個類別，
每條線是一個版本。先跑 v2_tah_outlier_table_experience.py 才有 csv 可讀。
"""

import os
import sys

import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

ROOT = os.path.abspath(f"{os.path.dirname(__file__)}/..")
sys.path.insert(0, ROOT)

VERSIONS = ["v2_ae_tanh", "v2_vae_tanh", "v2_tanh_perceiver"]
METRICS = ["移動距離_不穩定性", "重建損失提升"]
OUT = os.path.join(ROOT, "lab", "v2_tah_outlier_table_compare.png")

# 驗證過、CVD 安全的三色（dataviz skill 的 categorical 前三格，all-pairs 過關）
COLORS = {"v2_ae_tanh": "#2a78d6", "v2_vae_tanh": "#eb6834", "v2_tanh_perceiver": "#1baf7a"}
LABELS = {"v2_ae_tanh": "ae", "v2_vae_tanh": "vae", "v2_tanh_perceiver": "perceiver"}

mpl.rcParams["font.family"] = ["Heiti TC"]
mpl.rcParams["axes.unicode_minus"] = False
mpl.rcParams["figure.dpi"] = 130


def load_all():
    dfs = []
    for v in VERSIONS:
        path = os.path.join(ROOT, "lab", f"v2_tah_outlier_table_experience_{v}.csv")
        df = pd.read_csv(path)
        df["version"] = v
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)


def style(ax):
    ax.tick_params(labelsize=7, colors="#52514e")
    ax.grid(alpha=0.25, linewidth=0.6, color="#e1e0d9")
    for sp in ax.spines.values():
        sp.set_color("#c3c2b7")
        sp.set_linewidth(0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def main():
    df = load_all()
    cats = list(df["類別"].unique())

    fig, axes = plt.subplots(len(METRICS), len(cats),
                             figsize=(3.6 * len(cats), 3.2 * len(METRICS)),
                             squeeze=False)

    for row, metric in enumerate(METRICS):
        for col, cat in enumerate(cats):
            ax = axes[row, col]
            sub = df[df["類別"] == cat]
            for v in VERSIONS:
                s = sub[sub["version"] == v].sort_values("加入點數")
                ax.plot(s["加入點數"], s[metric], marker="o", ms=5, lw=2,
                        color=COLORS[v], label=LABELS[v])
            style(ax)
            if row == 0:
                ax.set_title(cat, fontsize=11, color="#0b0b0b")
            if col == 0:
                ax.set_ylabel(metric, fontsize=9, color="#0b0b0b")
            if row == len(METRICS) - 1:
                ax.set_xlabel("加入點數", fontsize=8, color="#52514e")
            ax.set_xticks(sorted(df["加入點數"].unique()))

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(VERSIONS),
              bbox_to_anchor=(0.5, 1.02), fontsize=10, frameon=False)

    fig.suptitle("三個 v2_tanh 版本的差異：移動距離 vs 重建損失提升", fontsize=13, y=1.06)
    fig.tight_layout()
    fig.savefig(OUT, bbox_inches="tight", facecolor="#fcfcfb")
    print(f"已存 {OUT}")


main()
