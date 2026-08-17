"""把多個模型版本的 latents.npz 放在同一張表上比較。

用法：
    python lab/eval_latents.py                    # 比較 model/ 底下所有版本
    python lab/eval_latents.py v2_ddae v2_ddae_fsce

每一欄的意義見 config/metrics.py 的 docstring。箭頭標的是方向：↑ 越大越好、
↓ 越小越好；aniso 沒有標，因為它是描述性的（1 是圓形分布，接近 0 是塌成線）。
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(f"{os.path.dirname(__file__)}/.."))
from config.dataset import PATCHES, ROOT  # noqa: E402
from config.metrics import evaluate, patch_counts, val_split  # noqa: E402

K = 15            # 跟 train.py 的 N_NEIGHBORS 對齊
N_CLUSTERS = 8    # 高維 pseudo-label 的群數
VAL_FRAC = 0.1    # 跟 train.py 的 VAL_FRAC 對齊
SEED = 0          # 跟 train.py 的 SEED 對齊

COLUMNS = [
    ("val_dev", "val dev↓"),
    ("trust", "trust↑"),
    ("cont", "cont↑"),
    ("knn_recall", "knn@15↑"),
    ("dist_rho", "dist ρ↑"),
    ("ari", "ARI↑"),
    ("silhouette", "sil↑"),
    ("occupancy", "occup↑"),
    ("aniso", "aniso"),
]


def versions(argv):
    """決定要評估哪些版本。

    argv：命令列參數（不含程式名）。空的話掃 model/ 底下所有有 latents.npz
    的目錄。

    回傳版本名稱的 list，已排序。
    """
    if argv:
        return argv
    model_dir = os.path.join(ROOT, "model")
    return sorted(v for v in os.listdir(model_dir)
                  if os.path.exists(os.path.join(model_dir, v, "result",
                                                 "latents.npz")))


def main():
    names = versions(sys.argv[1:])
    x_hi = patch_counts(PATCHES)
    val_idx, _ = val_split(len(x_hi), VAL_FRAC, SEED)
    print(f"{len(x_hi)} 個 patch，驗證集 {len(val_idx)} 個，k={K}，"
          f"pseudo-label {N_CLUSTERS} 群")

    rows = []
    for name in names:
        path = os.path.join(ROOT, "model", name, "result", "latents.npz")
        with np.load(path) as d:
            z, err = d["z"].astype(float), d["err"].astype(float)
        rows.append((name, evaluate(z, err, x_hi, val_idx, K, N_CLUSTERS, SEED)))

    w = max(len(n) for n, _ in rows) + 2
    print("\n" + "version".ljust(w) + "".join(h.rjust(10) for _, h in COLUMNS))
    for name, m in rows:
        print(name.ljust(w) + "".join(f"{m[k]:10.4f}" for k, _ in COLUMNS))


main()
