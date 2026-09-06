import numpy as np

data = np.arange(10)

# Normal Slicing (Fixed Stride) = Memory View
normal_slice = data[1:8:2]
print(f"Normal Slicing Shares Memory? : {np.shares_memory(data, normal_slice)}") # True

# Fancy Indexing (Random Stride) = Memory Copy
fancy_slice = data[[1, 5, 8]]
print(f"Fancy Indexing Shares Memory? : {np.shares_memory(data, fancy_slice)}") # False