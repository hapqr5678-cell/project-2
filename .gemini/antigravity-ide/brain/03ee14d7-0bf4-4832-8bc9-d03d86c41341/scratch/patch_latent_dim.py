"""批次修改所有 rebuild_test.py：自動從 checkpoint 偵測 latent_dim，不再寫死。"""
import glob

TARGETS = sorted(glob.glob("model/*/analyze/rebuild_test.py"))

for path in TARGETS:
    with open(path) as f:
        lines = f.readlines()

    new_lines = []
    changed = False

    for i, line in enumerate(lines):
        # 1. 刪除寫死的 LATENT_DIM 常數行
        stripped = line.strip()
        if stripped.startswith("LATENT_DIM") and "=" in stripped and "model" not in stripped.lower():
            # e.g. "LATENT_DIM = 2" or "LATENT_DIM = 16"
            changed = True
            continue  # skip this line

        # 2. 把 main() 裡的 model = ConvAE(LATENT_DIM) 改成自動偵測
        if "model = ConvAE(LATENT_DIM)" in line:
            indent = line[:len(line) - len(line.lstrip())]
            new_lines.append(f"{indent}sd = torch.load(CKPT, map_location=\"cpu\")\n")
            new_lines.append(f"{indent}LATENT_DIM = sd[\"encoder.6.weight\"].shape[0]\n")
            new_lines.append(f"{indent}model = ConvAE(LATENT_DIM)\n")
            changed = True
            continue

        # 3. 刪除緊接在後的 model.load_state_dict(torch.load(CKPT...)) 行，
        #    因為我們已經在上面 load 過了，改成用 sd
        if "model.load_state_dict(torch.load(CKPT" in line:
            indent = line[:len(line) - len(line.lstrip())]
            new_lines.append(f"{indent}model.load_state_dict(sd)\n")
            changed = True
            continue

        new_lines.append(line)

    if changed:
        with open(path, "w") as f:
            f.writelines(new_lines)
        print(f"✓ {path}")
    else:
        print(f"  {path} (no change)")
