import glob
import re

for f in glob.glob("/Users/yentso/developer/StartOver/model/*/analyze/latent_plot.py"):
    with open(f, "r") as file:
        content = file.read()
    
    # Check if we already bypassed it
    if "return 0.0  # Bypassed" in content:
        continue

    # Find the knn_r2 function and inject return 0.0
    content = re.sub(
        r'(def knn_r2\(z, y\):\s+(?:\"\"\"[^"]+\"\"\"\s+)?)',
        r'\1return 0.0  # Bypassed for speed\n    ',
        content
    )

    with open(f, "w") as file:
        file.write(content)
