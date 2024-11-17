from svd import svd
import numpy as np
# Generate a random matrix A
np.random.seed(0)
A = np.random.rand(5, 5)

# Compute SVD using custom svd function
results = svd(A)
U_custom, S_custom, Vt_custom = results["Left singular vectors (U)"], np.diag(results["Σ"]), results["Right singular vectors (V)"]

# Compute SVD using numpy's svd function
U_np, S_np, Vt_np = np.linalg.svd(A, full_matrices=False)

# Print the decompositions
print("Custom SVD Decomposition:")
print(U_custom)
print(np.diag(S_custom))
print(Vt_custom)

print("\nNumpy SVD Decomposition:")
print(U_np)
print(np.diag(S_np))
print(Vt_np)