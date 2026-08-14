import glob

for f in glob.glob("/Users/yentso/developer/StartOver/model/*/analyze/latent_plot.py"):
    with open(f, "r") as file:
        content = file.read()
    
    content = content.replace("print(f\"\\n將 {z.shape[1]} 維", "print(f\"\\n將 {z.shape[1]} 維")
    
    # Wait, the problem is literally a newline character in the source file.
    content = content.replace("print(f\"\n將 {z.shape[1]}", "print(f\"\\n將 {z.shape[1]}")
    
    with open(f, "w") as file:
        file.write(content)
