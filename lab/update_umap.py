import glob
import re

for f in glob.glob("/Users/yentso/developer/StartOver/model/*/analyze/latent_plot.py"):
    with open(f, "r") as file:
        content = file.read()
    
    # 1. Remove pca2 function if it exists
    content = re.sub(r'def pca2\(z\):.*?return z_pca, pca\.explained_variance_ratio_\n\n', '', content, flags=re.DOTALL)
    
    # 2. Update style function to accept xlabel and ylabel dynamically
    style_def_regex = r'def style\(ax, title\):\n    ax\.set_title\(title, fontsize=10\)\n    ax\.set_xlabel\("[^"]+", fontsize=8\)\n    ax\.set_ylabel\("[^"]+", fontsize=8\)'
    new_style_def = 'def style(ax, title, xlabel="z1", ylabel="z2"):\n    ax.set_title(title, fontsize=10)\n    ax.set_xlabel(xlabel, fontsize=8)\n    ax.set_ylabel(ylabel, fontsize=8)'
    content = re.sub(style_def_regex, new_style_def, content)

    # 3. Modify the main block right after OUTLIER_PCT
    main_block_regex = r'    out = dist > np\.percentile\(dist, OUTLIER_PCT\).*?fig, axes = plt\.subplots'
    
    new_main_block = """    out = dist > np.percentile(dist, OUTLIER_PCT)
    
    if z.shape[1] > 2:
        print(f"\\n將 {z.shape[1]} 維 latent 降維到 2 維 (UMAP)...")
        import umap
        zp = umap.UMAP(n_components=2, random_state=42).fit_transform(z)
        axes_labels = ("UMAP1", "UMAP2")
    else:
        zp = z
        axes_labels = ("z1", "z2")

    print()
    print(f"離群 {out.sum()} 個：POI 數中位數 {np.median(n_poi[out]):.0f} "
          f"(全體 {np.median(n_poi):.0f})，"
          f"偏心度 {np.median(ecc[out]):.3f} (全體 {np.median(ecc):.3f})")

    fig, axes = plt.subplots"""
    content = re.sub(main_block_regex, new_main_block, content, flags=re.DOTALL)

    # 4. Replace z with zp in plotting, and update style() calls
    content = content.replace("scatter(a, z,", "scatter(a, zp,")
    content = content.replace("zoom(b, z)", "zoom(b, zp)")
    content = content.replace("scatter(b, z,", "scatter(b, zp,")
    content = content.replace("zoom(c, z)", "zoom(c, zp)")
    content = content.replace("scatter(c, z,", "scatter(c, zp,")
    content = content.replace("zoom(e, z)", "zoom(e, zp)")
    content = content.replace("scatter(e, z,", "scatter(e, zp,")
    
    content = content.replace("f.scatter(z[~out, 0], z[~out, 1]", "f.scatter(zp[~out, 0], zp[~out, 1]")
    content = content.replace("f.scatter(z[out, 0], z[out, 1]", "f.scatter(zp[out, 0], zp[out, 1]")
    
    # 5. Update style(f, "...") to pass *axes_labels
    content = re.sub(r'style\(f, (.*?)\)\n', r'style(f, \1, *axes_labels)\n', content)
    
    with open(f, "w") as file:
        file.write(content)
