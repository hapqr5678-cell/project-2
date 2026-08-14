import os, sys
import numpy as np

ROOT = "/Users/yentso/developer/StartOver"
sys.path.insert(0, ROOT)
from config.dataset import PATCHES, CAT_ZH

p = np.load(PATCHES)
n_poi = p["n_poi"]
cat = p["cat"]

owner = np.repeat(np.arange(len(n_poi)), n_poi)
for c_id in range(len(CAT_ZH)):
    e_i = np.bincount(owner[cat == c_id], minlength=len(n_poi))
    e_t = n_poi
    
    with np.errstate(divide='ignore', invalid='ignore'):
        local_ratio = e_i / e_t
    global_ratio = e_i.sum() / e_t.sum()
    lq = np.nan_to_num(local_ratio / global_ratio)
    
    print(f"{CAT_ZH[c_id]}: max LQ = {lq.max():.2f}, mean = {lq.mean():.2f}, min = {lq.min():.2f}")
