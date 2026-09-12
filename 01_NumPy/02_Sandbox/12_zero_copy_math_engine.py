import numpy as np
import time

print("=== THE TRUE ZERO-COPY MATH ENGINE ===")

# 1. MEMORY ARCHITECTURE
size = 10_000_000
raw_data = np.random.uniform(0, 100, size).astype(np.float32)

# Injecting artificial garbage values (-999.0)
raw_data[::50000] = -999.0 

print(f"\nTotal Data Points: {size:,}")
print(f"RAM Usage        : {raw_data.nbytes / (1024*1024):.2f} MB")

start_time = time.time()

# 2. PURE ZERO-COPY CONDITION (No numpy.ma module!)
# True means the data is valid, False means it's garbage
valid_condition = (raw_data != -999.0)

# 3. MEMORY-SAFE REDUCTION
# Calculate metrics directly from RAM without extracting a copy
mean_val = np.mean(raw_data, where=valid_condition)
std_val = np.std(raw_data, where=valid_condition)

# 4. TRUE IN-PLACE MUTATION (Hardware-Level Overwrite)
# Force 32-bit float to prevent the 64-bit silent RAM trap!
np.subtract(raw_data, np.float32(mean_val), out=raw_data, where=valid_condition)
np.divide(raw_data, np.float32(std_val), out=raw_data, where=valid_condition)

execution_time = time.time() - start_time

print(f"\nEngine Execution Time : {execution_time:.4f} seconds")
print(f"Processed Original RAM: {raw_data[:5]}")