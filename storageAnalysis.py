import pandas as pd

# Function to calculate data storage for reduced SVD in KB per channel
def calculate_storage(height, width, num_singular_values):
    U_size = height * num_singular_values / 1024  # U: m x k
    S_size = num_singular_values / 1024          # S: k
    Vt_size = num_singular_values * width / 1024  # Vt: k x n
    total_data = U_size + S_size + Vt_size
    return total_data

# Input dimensions
height = 512
width = 768

# Store results for different numbers of singular values used
results = []
max_singular_values = min(height, width)  # Max singular values is the minimum of dimensions
for num_singular_values in range(1, max_singular_values + 1):  # Use at least 1 singular value
    total_data_per_channel = calculate_storage(height, width, num_singular_values)
    total_data_rgb = total_data_per_channel * 3  # Multiply by 3 for RGB channels
    results.append({
        "# Singular Values": num_singular_values,
        "Total Data (KB)": total_data_rgb
    })

# Create a DataFrame to display results
df = pd.DataFrame(results)

# Adjust pandas display settings to show all rows and columns
pd.set_option("display.max_rows", None)  # Show all rows
pd.set_option("display.max_columns", None)  # Show all columns
pd.set_option("display.width", None)  # Don't wrap columns

# Display results
print(df.to_string(index=False))
