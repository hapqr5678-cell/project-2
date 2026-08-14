import glob

for f in glob.glob("/Users/yentso/developer/StartOver/model/*/analyze/latent_plot.py"):
    with open(f, "r") as file:
        content = file.read()
    
    content = content.replace("len(n_poi)), n_poi)", 'len(p["n_poi"])), p["n_poi"])')
    content = content.replace("minlength=len(n_poi)", 'minlength=len(p["n_poi"])')
    content = content.replace("e_t = n_poi", 'e_t = p["n_poi"]')
    
    with open(f, "w") as file:
        file.write(content)
