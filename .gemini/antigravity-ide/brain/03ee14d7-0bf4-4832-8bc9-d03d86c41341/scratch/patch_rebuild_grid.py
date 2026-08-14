"""批次修改所有 rebuild_test.py：加入 REBUILD_GRID 開關，開啟時顯示 CELL 格線。"""
import glob
import re

TARGETS = sorted(glob.glob("model/*/analyze/rebuild_test.py"))

for path in TARGETS:
    with open(path) as f:
        content = f.read()

    changed = False

    # 1. 加入 import（在 from config.dataset 那行之後）
    if "from config.result_style" not in content:
        content = content.replace(
            "from config.dataset import",
            "from config.result_style import REBUILD_GRID  # noqa: E402\n"
            "from config.dataset import",
        )
        changed = True

    # 2. 在 style() 裡加入 CELL 格線
    old_style_tail = """\
    ax.grid(alpha=0.15, linewidth=0.5)
    for s in ax.spines.values():
        s.set_alpha(0.3)"""

    new_style_tail = """\
    ax.grid(alpha=0.15, linewidth=0.5)
    if REBUILD_GRID:
        import numpy as _np
        ticks = _np.arange(-GRID // 2, GRID // 2 + 1) * CELL
        for t in ticks:
            ax.axhline(t, color='#888', lw=0.3, alpha=0.35)
            ax.axvline(t, color='#888', lw=0.3, alpha=0.35)
    for s in ax.spines.values():
        s.set_alpha(0.3)"""

    if "REBUILD_GRID" not in content.split("def style")[1].split("\ndef ")[0]:
        content = content.replace(old_style_tail, new_style_tail)
        changed = True

    if changed:
        with open(path, "w") as f:
            f.write(content)
        print(f"✓ {path}")
    else:
        print(f"  {path} (already patched)")
