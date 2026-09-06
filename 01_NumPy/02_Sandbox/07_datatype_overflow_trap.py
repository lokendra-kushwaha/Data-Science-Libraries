import numpy as np

print("=== 1. THE SILENT OVERFLOW (MEMORY CASTING) ===")
# 1. Data comes in normally as int32/int64
intended_values = np.array([50, 127, 128, 200, 300])
print(f"Original Data (int32) : {intended_values}")

# 2. Trying to save RAM by forcing a downcast to int8
# This bypasses the safety check and causes the silent wrap-around
corrupted_array = intended_values.astype(np.int8)
print(f"Corrupted Data (int8) : {corrupted_array}\n")

print("=== 2. THE SILENT OVERFLOW (ARITHMETIC) ===")
# 3. Math operations on max capacity int8 also silently fail
max_val_array = np.array([127], dtype=np.int8)
print(f"Starting value : {max_val_array}")

# Adding 1 pushes it over the 8-bit limit (01111111 + 1 = 10000000)
overflow_math = max_val_array + 1
print(f"After adding 1 : {overflow_math}\n")

print("=== 3. MEMORY FOOTPRINT (float64 vs float32) ===")
size = 100_000_000 # 100 Million items

# Default float64 (8 Bytes per number)
arr_64 = np.ones(size, dtype=np.float64)
mb_64 = arr_64.nbytes / (1024 * 1024)
print(f"float64 RAM Usage : {mb_64:.2f} MB")

# Optimized float32 (4 Bytes per number)
arr_32 = np.ones(size, dtype=np.float32)
mb_32 = arr_32.nbytes / (1024 * 1024)
print(f"float32 RAM Usage : {mb_32:.2f} MB")