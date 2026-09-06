import numpy as np
import numpy.ma as ma
import time

print("=== THE CUSTOM MATH ENGINE ===")

# 1. MEMORY ARCHITECTURE (float32 saves 50% RAM)
size = 10_000_000
raw_data = np.random.uniform(0, 100, size).astype(np.float32)

# Injecting artificial garbage values (-999.0) to simulate broken sensors
raw_data[::50000] = -999.0 

print(f"\nTotal Data Points: {size:,}")
print(f"RAM Usage        : {raw_data.nbytes / (1024*1024):.2f} MB")

start_time = time.time()

# 2. GARBAGE FILTRATION (Masking without breaking Strides)
# We mask out the -999.0 values so they don't corrupt our math
safe_data = ma.masked_equal(raw_data, -999.0)

# 3. VECTORIZED MATH & ZERO-COPY EXECUTION
# Formula: (Data - Mean) / Standard Deviation
# We use in-place operations to avoid RAM spikes

# Calculate metrics only on valid data (SIMD executed)
mean_val = safe_data.mean()
std_val = safe_data.std()

# 4. IN-PLACE MUTATION
# We subtract the mean and divide by std directly in the existing RAM block
np.subtract(safe_data, mean_val, out=safe_data)
np.divide(safe_data, std_val, out=safe_data)

execution_time = time.time() - start_time

print(f"\nEngine Execution Time : {execution_time:.4f} seconds")
print(f"Processed Array (View): {safe_data[:5]}")