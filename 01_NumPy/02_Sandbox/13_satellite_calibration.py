import numpy as np

# 1. The Setup (5 Million Points)
raw_data = np.random.uniform(10, 40, 5_000_000).astype(np.float32)
raw_data[::10000] = -99.0
raw_data[::25000] = 999.0

print("Before Calibration:", raw_data[:5])

# 2. Identify Garbage (No numpy.ma module!)
# This creates a simple boolean array, NOT a copied MaskedArray
garbage_condition = (raw_data == -99.0) | (raw_data == 999.0)

# 3. TRUE Zero-Copy Calibration (In-Place)
# - out=raw_data: Forces the result back into the exact same RAM block
# - where=~garbage_condition: The '~' (NOT) applies the math ONLY to valid data
# - np.float32(2.5): Prevents the silent 64-bit RAM upgrade trap
np.add(raw_data, np.float32(2.5), out=raw_data, where=~garbage_condition)

# Proof that the underlying raw_data was modified in-place
print("After Calibration :", raw_data[:5])