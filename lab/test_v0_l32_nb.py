import numpy as np
from config.dataset import result
d = np.load(result("v0_l32_nb", "latents.npz"))
print("v0_l32_nb z shape:", d["z"].shape)
