import numpy as np

def svd(A):
    # Step 1: Compute A * A^T to find the singular values (σi)
    AAT = np.dot(A, A.T)
    eigenvalues_AAT, eigenvectors_AAT = np.linalg.eig(AAT)
    singular_values = np.sqrt(eigenvalues_AAT)

    # Step 2: Compute AT * A to find the right singular vectors (columns of V)
    ATA = np.dot(A.T, A)
    eigenvalues_ATA, eigenvectors_ATA = np.linalg.eig(ATA)

    # Normalize eigenvectors of AT * A to form V (right singular vectors)
    V = eigenvectors_ATA / np.linalg.norm(eigenvectors_ATA, axis=0)

    # Step 3: Sort singular values and reorder U and V accordingly
    sorted_indices = np.argsort(singular_values)[::-1]  # Indices to sort in descending order
    singular_values = singular_values[sorted_indices]  # Sort singular values
    V = V[:, sorted_indices]  # Reorder columns of V
    eigenvectors_AAT = eigenvectors_AAT[:, sorted_indices]  # Reorder columns of U (AAT eigenvectors)

    # Step 4: Form Σ (diagonal matrix of singular values, same size as A)
    Σ = np.zeros_like(A, dtype=float)
    np.fill_diagonal(Σ, singular_values)

    # Step 5: Compute U (left singular vectors) using U = (1/σ) * A * v
    U = np.zeros((A.shape[0], A.shape[0]))
    for i in range(len(singular_values)):
        U[:, i] = np.dot(A, V[:, i]) / singular_values[i]

    # Normalize U
    U = U / np.linalg.norm(U, axis=0)

    # Verification: Reconstruct A from U, Σ, and V^T
    A_reconstructed = np.dot(U, np.dot(Σ, V.T))

    # Return results
    results = {
        "AAT": AAT,
        "Eigenvalues of AAT": eigenvalues_AAT,
        "Singular values": singular_values,
        "ATA": ATA,
        "Eigenvalues of ATA": eigenvalues_ATA,
        "Right singular vectors (V)": V,
        "Σ": Σ,
        "Left singular vectors (U)": U,
        "Reconstructed A": A_reconstructed
    }
    
    return results

if __name__ == "__main__":
    # Example usage
    A = np.random.rand(4, 4)  # Replace with any matrix
    results = svd(A)
    for key, value in results.items():
        print(f"{key}:\n{value}\n")