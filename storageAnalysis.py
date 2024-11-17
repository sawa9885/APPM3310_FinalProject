import numpy as np

matrix_size = 360
A = np.random.rand(matrix_size, matrix_size)

# Compute full SVD
U, S, Vt = np.linalg.svd(A, full_matrices=False)

# Function to calculate data storage
def calculate_storage(U, S, Vt, cutoff):
    rank = len(S) - cutoff
    U_reduced = U[:, :rank]
    S_reduced = S[:rank]
    Vt_reduced = Vt[:rank, :]
    total_data = U_reduced.size + S_reduced.size + Vt_reduced.size
    return U_reduced.size, S_reduced.size, Vt_reduced.size, total_data

# Store results for different cutoffs
results = []
for cutoff in range(matrix_size):
    U_size, S_size, Vt_size, total_data = calculate_storage(U, S, Vt, cutoff)
    results.append({
        "Cutoff": cutoff,
        "U_size": U_size,
        "S_size": S_size,
        "Vt_size": Vt_size,
        "Total_data": total_data
    })

# Display results
import pandas as pd
df = pd.DataFrame(results)

# Disable truncation
pd.set_option("display.max_rows", None)  # Show all rows
pd.set_option("display.max_columns", None)  # Show all columns
pd.set_option("display.width", None)  # Don't wrap columns

print(df)
