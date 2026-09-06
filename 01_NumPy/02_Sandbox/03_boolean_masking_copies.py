import numpy as np
import time

data = np.arange(1000000)

print("** 1. The Python Loop Way **")
start_time = time.time()
py_filtered = [x for x in data if x > 500000]
py_time = time.time() - start_time
print(f"Time taken: {py_time:.5f} seconds")

print("\n** 2. The NumPy Boolean Masking Way **")
start_time = time.time()
np_filtered = data[data > 500000]
np_time = time.time() - start_time
print(f"Time taken: {np_time:.5f} seconds")

print("\n** 3. Memory Check (View vs Copy) **")
is_sharing = np.shares_memory(data, np_filtered)
print(f"Is it sharing memory (View)? : {is_sharing}")