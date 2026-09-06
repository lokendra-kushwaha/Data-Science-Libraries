import numpy as np

# Case 1: 1D to 2D (Creating a diagonal matrix)
my_1d_array = np.array([10, 20, 30])
diagonal_matrix = np.diag(my_1d_array)

print("Created 2D Matrix:\n", diagonal_matrix)


# Case 2: 2D to 1D (Extracting the diagonal)
my_2d_matrix = np.array([[1, 2, 3],
                         [4, 5, 6],
                         [7, 8, 9]])

extracted_diagonal = np.diag(my_2d_matrix)

print("\nExtracted Diagonal:", extracted_diagonal)

# The Architecture Proof (It's a View, not a Copy!)
print("Shares Memory? :", np.shares_memory(my_2d_matrix, extracted_diagonal)) # Returns True

# A non-square 2x4 matrix
non_square_matrix = np.array([[1, 2, 3, 4],
                              [5, 6, 7, 8]])

# Extracting the diagonal
extracted = np.diag(non_square_matrix)
print("Extracted Diagonal:", extracted) 
# Output: [1, 6]