import numpy as np
import time

print("=== CUSTOM UFUNC ENGINE ===")

# 1. A pure Python function with complex if-else logic
def sensor_calibration_logic(val):
    if val < 0:
        return 0.0          # Reset negative spikes
    elif val > 100:
        return val * 0.9    # Scale down extreme heat
    else:
        return val          # Keep normal data

# 2. Creating the NumPy Ufunc
# Syntax: np.frompyfunc(function, number_of_inputs, number_of_outputs)
c_level_ufunc = np.frompyfunc(sensor_calibration_logic, 1, 1)

# 3. The Setup (5 Million points)
raw_data = np.random.uniform(-20, 150, 5_000_000).astype(np.float32)

print(f"\nProcessing {len(raw_data):,} data points...")

start_time = time.time()

# 4. Execution
# The Ufunc automatically broadcasts the Python logic across the C-array.
# NOTE: frompyfunc returns 'object' datatype, so we cast it back to float32
processed_data = c_level_ufunc(raw_data).astype(np.float32)

execution_time = time.time() - start_time

print(f"Engine Execution Time: {execution_time:.4f} seconds")
print(f"Original Data  : {raw_data[:5]}")
print(f"Processed Data : {processed_data[:5]}")