import numpy as np
import time
from scipy.spatial import cKDTree

N = 78802
D = 32
K = 51

print(f"Generating data: {N} points in {D} dimensions...")
zs = np.random.randn(N, D)

print("Building KDTree...")
start = time.time()
tree = cKDTree(zs)
print(f"Build time: {time.time() - start:.2f} s")

print(f"Querying KDTree for {K} neighbors...")
start = time.time()
tree.query(zs[:1000], k=K) # Query only 1000 points to estimate
elapsed = time.time() - start
print(f"Query 1000 points time: {elapsed:.2f} s")
print(f"Estimated query all {N} points time: {elapsed * (N/1000):.2f} s")
