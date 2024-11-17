import numpy as np

# Example matrix A
np.random.seed(0)
A = 255*np.random.rand(5, 5)

# Step 1: Compute SVD decomposition using NumPy
U, S, Vt = np.linalg.svd(A)

# Print the original matrix A
print("Original Matrix A:")
print(A)

# Step 2: Modify the singular values by setting the smallest to 0
S_modified = S.copy()
S_modified[-1] = 0  # Set the smallest singular value to 0

# Step 3: Reconstruct the matrix with modified singular values
Σ_modified = np.zeros_like(A, dtype=float)  # Create a Σ matrix of the same size as A
np.fill_diagonal(Σ_modified, S_modified)  # Place modified singular values on the diagonal
A_reconstructed = np.dot(U, np.dot(Σ_modified, Vt))  # Reconstruct A

# Print the reconstructed matrix
print("\nReconstructed Matrix A (After Modifying Singular Values):")
print(A_reconstructed)

# Print the difference matrix
print("\nDifference Matrix:")
print(A-A_reconstructed)
