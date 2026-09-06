import numpy as np
import time

# Create two arrays with 10 Million items each
# Using float32 to optimize RAM and Cache usage
array_A = np.ones(10000000, dtype=np.float32)
array_B = np.ones(10000000, dtype=np.float32)

print("=== THE VECTORIZATION & SIMD TEST ===\n")

# 1. The Slow Way (Python For Loop)
# Fetches one item at a time, checks data types, and adds them.
start = time.time()
py_result = [array_A[i] + array_B[i] for i in range(len(array_A))]
py_time = time.time() - start
print(f"Python Loop Time : {py_time:.5f} seconds")

# 2. The Vectorized Way (NumPy C-Engine + SIMD)
# Bypasses Python entirely. Adds contiguous chunks in parallel.
start = time.time()
np_result = array_A + array_B 
np_time = time.time() - start
print(f"Vectorized Time  : {np_time:.5f} seconds")

# Calculate the speed difference
speedup = py_time / np_time
print(f"\nVectorization is {speedup:.0f}x faster!")