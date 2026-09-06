import numpy as np
import time

# Matrix size: 10,000 x 10,000 (100 Million items each)
# 1. C-Order (Row-Major): Data is stored row-by-row contiguously in RAM
c_matrix = np.ones((10000, 10000), dtype=np.float32, order='C')

# 2. F-Order (Column-Major): Data is stored column-by-column contiguously in RAM
f_matrix = np.ones((10000, 10000), dtype=np.float32, order='F')

print("=== THE MEMORY ALIGNMENT TEST (C-ORDER vs F-ORDER) ===\n")

# ---------------------------------------------------------
# TEST SUITE 1: C-ORDER (Row-Major)
# ---------------------------------------------------------
print("** 1. C-Order Matrix (Row-Major RAM) **")

# Case 1A: axis=0 (Summing down columns)
# CPU fetches Row 1, fetches Row 2, adds them parallelly (Vector + Vector)
start = time.time()
_ = c_matrix.sum(axis=0) 
c_axis0_time = time.time() - start
print(f"axis=0 (Vector Addition) : {c_axis0_time:.5f} seconds")

# Case 1B: axis=1 (Summing across rows)
# CPU fetches a Row (100% Cache Hit) but must reduce it to a single number (Reduction penalty)
start = time.time()
_ = c_matrix.sum(axis=1) 
c_axis1_time = time.time() - start
print(f"axis=1 (Scalar Reduction): {c_axis1_time:.5f} seconds")

print("\n---------------------------------------------------------")

# ---------------------------------------------------------
# TEST SUITE 2: F-ORDER (Column-Major)
# ---------------------------------------------------------
print("** 2. F-Order Matrix (Column-Major RAM) **")

# Case 2A: axis=0 (Summing down columns)
# CPU fetches a Column (100% Cache Hit) but must reduce it to a single number (Reduction penalty)
start = time.time()
_ = f_matrix.sum(axis=0)
f_axis0_time = time.time() - start
print(f"axis=0 (Scalar Reduction): {f_axis0_time:.5f} seconds")

# Case 2B: axis=1 (Summing across rows)
# CPU fetches Column 1, fetches Column 2, adds them parallelly (Vector + Vector)
start = time.time()
_ = f_matrix.sum(axis=1)
f_axis1_time = time.time() - start
print(f"axis=1 (Vector Addition) : {f_axis1_time:.5f} seconds")