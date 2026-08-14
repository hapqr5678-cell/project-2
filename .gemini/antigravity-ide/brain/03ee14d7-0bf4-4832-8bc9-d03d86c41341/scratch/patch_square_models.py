import os
import glob
import re

ROOT = "/Users/yentso/developer/StartOver/model"

# 1. Update ae.py in all models
for path in glob.glob(f"{ROOT}/*/ae.py"):
    with open(path, "r") as f:
        content = f.read()

    # Replace RADIUS import and usage with HALF_WIDTH
    content = content.replace("RADIUS", "HALF_WIDTH")
    
    # Remove MASK and N_VALID definitions
    # Usually looks like:
    # _axis = (torch.arange(GRID, dtype=torch.float32) + 0.5 - GRID / 2) * CELL
    # _yy, _xx = torch.meshgrid(_axis, _axis, indexing="ij")
    # MASK = ((_xx ** 2 + _yy ** 2) <= HALF_WIDTH ** 2).view(1, 1, GRID, GRID)
    # N_VALID = int(MASK.sum()) * N_CAT   # loss 的分母：圓內格數 x 類別數
    
    content = re.sub(r"# 圓形遮罩.*?\n", "", content)
    content = re.sub(r"_axis = .*?\n", "", content)
    content = re.sub(r"_yy, _xx = torch\.meshgrid.*?\n", "", content)
    content = re.sub(r"MASK = .*?\n", "", content)
    content = re.sub(r"N_VALID = .*?\n", "", content)

    # In v0/ae.py and others:
    # return mse(out * MASK.to(out.device), x * MASK.to(x.device)) / (N_VALID / N_CAT) * (GRID * GRID)
    # becomes: return mse(out, x)
    content = re.sub(r"return mse\(out \* MASK.*? x \* MASK.*?\).*?\n", "return mse(out, x)\n", content)
    
    # Also handle mse_loss(out, x) which was masking:
    # m = MASK.to(out.device)
    # return mse(out * m, x * m) * (GRID * GRID / N_VALID)
    content = re.sub(r"m = MASK\.to\(out\.device\)\n\s*return mse\(out \* m, x \* m\) \* \(GRID \* GRID / N_VALID\)", "return mse(out, x)", content)
    content = re.sub(r"m = MASK\.to\(out\.device\)\n\s*return F\.mse_loss\(out \* m, x \* m\) \* \(GRID \* GRID / N_VALID\)", "return F.mse_loss(out, x)", content)
    content = re.sub(r"m = MASK\.to\(out\.device\)\n\s*return F\.mse_loss\(out \* m, x \* m, reduction='sum'\) / N_VALID", "return F.mse_loss(out, x)", content)

    # For poisson_nll, poisson_deviance, nb_nll, nb_deviance:
    # m = MASK.to(log_lam.device)
    # return (-ll * m).sum(dim=(1, 2, 3)) / N_VALID
    # becomes:
    # return -ll.mean(dim=(1, 2, 3))
    
    content = re.sub(r"m = MASK\.to\([^)]+\)\n\s*return \(-ll \* m\)\.sum\(dim=\(1, 2, 3\)\) / N_VALID", "return -ll.mean(dim=(1, 2, 3))", content)
    content = re.sub(r"m = MASK\.to\([^)]+\)\n\s*return \(cell \* m\)\.sum\(dim=\(1, 2, 3\)\) / N_VALID", "return cell.mean(dim=(1, 2, 3))", content)
    
    # Also some might have just .sum() / N_VALID
    content = re.sub(r"m = MASK\.to\([^)]+\)\n\s*return \(-ll \* m\)\.sum\(\) / N_VALID", "return -ll.mean()", content)
    content = re.sub(r"m = MASK\.to\([^)]+\)\n\s*return \(cell \* m\)\.sum\(\) / N_VALID", "return cell.mean()", content)

    with open(path, "w") as f:
        f.write(content)


# 2. Update all analyze/rebuild_test.py
for path in glob.glob(f"{ROOT}/*/analyze/rebuild_test.py"):
    with open(path, "r") as f:
        content = f.read()

    content = content.replace("RADIUS", "HALF_WIDTH")
    content = content.replace("MASK, ", "")
    content = content.replace(" MASK,", "")

    # Remove circle drawing
    # ax.add_patch(plt.Circle((0, 0), HALF_WIDTH, fill=False, lw=1.6, color=edge, alpha=0.8))
    content = re.sub(r"\s*ax\.add_patch\(plt\.Circle\(\(0, 0\), HALF_WIDTH.*?alpha=0\.8\)\)\n", "\n", content)

    # Remove inside mask logic in draw_recon
    # inside = MASK[0, 0].numpy().astype(bool)
    # if not inside[gy, gx]: continue
    content = re.sub(r"\s*inside = MASK.*?astype\(bool\)\n", "\n", content)
    content = re.sub(r"\s*if not inside\[gy, gx\]:\n\s*continue\n", "\n", content)

    # total[inside].sum() -> total.sum()
    content = content.replace("total[inside]", "total")
    content = content.replace("total[~inside].sum()", "0") # we'll just remove the whole line below
    
    # Remove "圓外（loss 沒約束）"
    content = re.sub(r"\s*print\(f\"圓外（loss 沒約束）.*?\n.*?\n", "\n", content)
    content = content.replace("圓內真實共", "真實共")
    content = content.replace("圓內平均", "平均")
    content = content.replace("圓內重建", "重建")
    content = content.replace("圓內 Poisson", "Poisson")
    content = content.replace("圓內 NB", "NB")

    # In main(), per_cat and cnt calculations
    # m = MASK.to(log_lam.dtype)
    # cell = ... * m
    # per_cat = cell[0].sum(dim=(1, 2)) / MASK.sum()
    # cnt_x = (x * m)[0].sum(dim=(1, 2))
    # cnt_r = (lam * m)[0].sum(dim=(1, 2))
    
    # Instead of regex, we'll replace the block manually or with careful regex
    content = re.sub(r"\s*m = MASK\.to\(.*?\)\n", "\n", content)
    content = re.sub(r" \* m\n", "\n", content)
    content = re.sub(r" \* m\)[0]", ")[0]", content)
    content = re.sub(r" / MASK\.sum\(\)", " / (GRID * GRID)", content)

    # Also fix draw_recon return
    content = content.replace("return total, inside", "return total")
    content = content.replace("total, inside = draw_recon", "total = draw_recon")
    content = content.replace("total[inside]", "total")
    content = content.replace("total[~inside]", "total")

    with open(path, "w") as f:
        f.write(content)


# 3. Update all analyze/make_outlier.py
for path in glob.glob(f"{ROOT}/*/analyze/make_outlier.py"):
    with open(path, "r") as f:
        content = f.read()

    content = content.replace("RADIUS", "HALF_WIDTH")
    
    # r = HALF_WIDTH * np.sqrt(rng.random(k))
    # theta = rng.random(k) * 2 * np.pi
    # x = r * np.cos(theta)
    # y = r * np.sin(theta)
    
    square_noise = '''
    # 在正方形內均勻灑 k 個點
    x = rng.uniform(-HALF_WIDTH, HALF_WIDTH, k)
    y = rng.uniform(-HALF_WIDTH, HALF_WIDTH, k)
    '''
    content = re.sub(r"\s*r = HALF_WIDTH \* np\.sqrt\(rng\.random\(k\)\)\n\s*theta = rng\.random\(k\) \* 2 \* np\.pi\n\s*x = r \* np\.cos\(theta\)\n\s*y = r \* np\.sin\(theta\)\n", square_noise, content)

    with open(path, "w") as f:
        f.write(content)


# 4. Update all analyze/outlier_compare.py & latent_plot.py
for path in glob.glob(f"{ROOT}/*/analyze/outlier_compare.py") + glob.glob(f"{ROOT}/*/analyze/latent_plot.py"):
    with open(path, "r") as f:
        content = f.read()

    content = content.replace("RADIUS", "HALF_WIDTH")
    content = re.sub(r"\s*ax\.add_patch\(plt\.Circle\(\(0, 0\), HALF_WIDTH.*?alpha=0\.8\)\)\n", "\n", content)
    # ax.add_patch(plt.Circle((0, 0), HALF_WIDTH, fill=False, lw=1.6, color=IN_COLOR, alpha=0.8))

    with open(path, "w") as f:
        f.write(content)

print("Patching complete!")
