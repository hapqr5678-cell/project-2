import glob
import re

for f in glob.glob("/Users/yentso/developer/StartOver/model/*/analyze/latent_plot.py"):
    with open(f, "r") as file:
        content = file.read()
    
    # 1. Add import argparse
    if "import argparse" not in content:
        content = content.replace("import sys\n", "import sys\nimport argparse\n")

    # 2. Add CAT_ZH to config.dataset import
    if "CAT_ZH" not in content:
        content = content.replace("from config.dataset import ", "from config.dataset import CAT_ZH, ")

    # 3. Add LQ calculation to main()
    lq_code = """
    ap = argparse.ArgumentParser()
    ap.add_argument("--c", type=int, default=0, help="類別 ID (預設 0)")
    c_id = ap.parse_args().c

    p = np.load(PATCHES)
    owner = np.repeat(np.arange(len(n_poi)), n_poi)
    e_i = np.bincount(owner[p["cat"] == c_id], minlength=len(n_poi))
    e_t = n_poi
    with np.errstate(divide='ignore', invalid='ignore'):
        local_ratio = e_i / e_t
    global_ratio = e_i.sum() / e_t.sum()
    lq = np.nan_to_num(local_ratio / global_ratio)
    
    print(f"LQ_i ({CAT_ZH[c_id]}): max={lq.max():.2f}, mean={lq.mean():.2f}")
"""
    if "ap = argparse.ArgumentParser()" not in content:
        # insert after def main():
        content = re.sub(r'(def main\(\):)', r'\1' + lq_code, content)

    # 4. Replace the middle-top scatter
    # It looks like: scatter(b, zp, np.log10(n_poi), f"放大 {ZOOM_PCT[0]}~{ZOOM_PCT[1]}%：色=POI 數", "log10(POI 數)")
    # Or scatter(b, z, ...)
    content = re.sub(
        r'(zoom\(b, (z|zp)\)\n\s+scatter\(b, (z|zp), )np\.log10\(n_poi\),\n\s+f"放大[^"]+", "log10\(POI 數\)"\)',
        r'\1lq, f"放大 {ZOOM_PCT[0]}~{ZOOM_PCT[1]}%：LQ ({CAT_ZH[c_id]})", "LQ", cmap="magma")',
        content
    )

    with open(f, "w") as file:
        file.write(content)
