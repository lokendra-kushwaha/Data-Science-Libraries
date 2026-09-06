import numpy as np

# A standard 2D C-Order matrix
matrix = np.array([[1, 2, 3],
                   [4, 5, 6]])

print("=== ADVANCED C-LEVEL ITERATION (np.nditer) ===\n")

print("1. Standard Memory-Aligned Traversal:")
# nditer doesn't just blindly loop; it reads directly from RAM in physical order
for x in np.nditer(matrix):
    print(x, end=' ')
print("\n")


print("2. In-Place Memory Modification (Read/Write Mode):")
# Normal loops create copies. op_flags give you direct pointers to modify original RAM.
for x in np.nditer(matrix, op_flags=['readwrite']):
    x[...] = x * 10  # The [...] syntax writes back directly to the physical memory block

print(matrix)

matrix = np.array([[10, 20], 
                   [30, 40]])

print("=== 1. The np.ndenumerate Tool ===\n")
# Returns exactly WHERE the item is, and WHAT the item is
for index, value in np.ndenumerate(matrix):
    print(f"Location {index} has value: {value}")


print("\n=== 2. The np.ndindex Tool ===\n")
# Generates only the coordinate grid for a given shape (e.g., 2x2)
for index in np.ndindex(matrix.shape):
    print(f"Valid Coordinate: {index}")