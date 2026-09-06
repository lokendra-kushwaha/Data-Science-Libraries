import numpy as np
import time

print("--- 2D Matrix CPU Cache Benchmark ---")
N = 10000
matrix = np.ones((N, N), dtype=np.float32)

start = time.perf_counter()
row_sum = matrix.sum(axis=1)
time_row = time.perf_counter() - start
print(f"Row-wise calculation (axis=1): {time_row:.4f} seconds")

start = time.perf_counter()
col_sum = matrix.sum(axis=0)
time_col = time.perf_counter() - start
print(f"Column-wise calculation (axis=0): {time_col:.4f} seconds")