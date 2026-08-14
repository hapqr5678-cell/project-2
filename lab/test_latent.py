import sys, os, numpy as np
sys.path.insert(0, "/Users/yentso/developer/StartOver")
from config.dataset import result
d = np.load(result("v0_l32_weight_mse", "latents.npz"))
print("z shape:", d["z"].shape)
print("n_poi shape:", d["n_poi"].shape)
