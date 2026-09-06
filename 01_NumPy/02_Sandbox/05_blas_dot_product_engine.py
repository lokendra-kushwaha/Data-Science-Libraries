import numpy as np
import time

# Matrix size: 1000 x 1000
c_matrix = np.ones((1000, 1000), dtype=np.float32, order='C')
f_matrix = np.ones((1000, 1000), dtype=np.float32, order='F')

print("=== BLAS ENGINE MATRIX MULTIPLICATION TEST ===\n")

# WARM-UP RUN (To bypass the 'Cold Start' anomaly of loading BLAS into memory)
_ = np.dot(c_matrix, c_matrix)

# Test 1: C-Order * C-Order
start = time.time()
_ = np.dot(c_matrix, c_matrix)
print(f"C-Order * C-Order Time : {time.time() - start:.5f} seconds")

# Test 2: C-Order * F-Order (The Architect's Hypothesis)
start = time.time()
_ = np.dot(c_matrix, f_matrix)
print(f"C-Order * F-Order Time : {time.time() - start:.5f} seconds")

# Test 3: F-Order * C-Order
start = time.time()
_ = np.dot(f_matrix, c_matrix)
print(f"F-Order * C-Order Time : {time.time() - start:.5f} seconds")

# Test 4: F-Order * F-Order
start = time.time()
_ = np.dot(f_matrix, f_matrix)
print(f"F-Order * F-Order Time : {time.time() - start:.5f} seconds")