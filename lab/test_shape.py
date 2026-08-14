import sys, os, numpy as np
sys.path.insert(0, "/Users/yentso/developer/StartOver")
from config.dataset import PATCHES, HALF_WIDTH
p = np.load(PATCHES)
print("n_poi shape:", p["n_poi"].shape)
print("offsets shape:", p["offsets"].shape)

def eccentricity(p):
    offs, dx, dy = p["offsets"], p["dx"], p["dy"]
    v = np.zeros(len(offs) - 1)
    for i in range(len(v)):
        s, e = offs[i], offs[i + 1]
        v[i] = np.hypot(dx[s:e].mean(), dy[s:e].mean()) / HALF_WIDTH
    return v

ecc = eccentricity(p)
print("ecc shape:", ecc.shape)
