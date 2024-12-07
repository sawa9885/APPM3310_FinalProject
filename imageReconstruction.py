import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os

def compress_image_svd(image_path, num_singular_values):
    # Load the image
    img = Image.open(image_path)
    img_array = np.array(img)

    # Ensure the image has RGB channels
    if img_array.ndim != 3 or img_array.shape[2] != 3:
        raise ValueError("Image must have three color channels (RGB).")

    # Extract the R, G, B channels
    R, G, B = img_array[:, :, 0], img_array[:, :, 1], img_array[:, :, 2]

    # Function to perform SVD and calculate size of reduced decomposition
    def svd_reconstruct(channel, num_singular_values):
        U, S, Vt = np.linalg.svd(channel, full_matrices=False)
        total_singular_values = len(S)

        # Reconstruct the channel with reduced SVD
        S_reduced = np.diag(S[:num_singular_values])
        reconstructed_channel = np.dot(U[:, :num_singular_values], np.dot(S_reduced, Vt[:num_singular_values, :]))

        # Quantize values to 8-bit integers (0-255)
        quantized_channel = np.clip(reconstructed_channel, 0, 255).astype(np.uint8)

        # Calculate size of reduced matrices (in bytes) after quantization
        reduced_size = (
            (U[:, :num_singular_values].shape[0] * U[:, :num_singular_values].shape[1])  # U
            + num_singular_values  # S
            + (Vt[:num_singular_values, :].shape[0] * Vt[:num_singular_values, :].shape[1])  # Vt
        )  # All in bytes per element (1 byte after quantization)

        return quantized_channel, total_singular_values, reduced_size

    # Compress each channel
    R_compressed, R_total_sv, R_reduced_size = svd_reconstruct(R, num_singular_values)
    G_compressed, G_total_sv, G_reduced_size = svd_reconstruct(G, num_singular_values)
    B_compressed, B_total_sv, B_reduced_size = svd_reconstruct(B, num_singular_values)

    # Ensure all channels have the same total singular values
    assert R_total_sv == G_total_sv == B_total_sv, "Total singular values mismatch across channels."
    total_singular_values = R_total_sv

    # Combine the channels to form the compressed image
    compressed_image = np.stack([R_compressed, G_compressed, B_compressed], axis=2)

    # Total reduced size for all channels (in bytes)
    total_reduced_size = (R_reduced_size + G_reduced_size + B_reduced_size)

    # Calculate sizes of the original image and the reduced decomposition
    original_size = os.path.getsize(image_path)

    return img, compressed_image, original_size, total_reduced_size, total_singular_values

# Parameters
image_path = "images/1.bmp"
num_singular_values = 512  # Adjust for desired compression level

# Compress the image
original_img, compressed_img, original_size, reduced_size, total_singular_values = compress_image_svd(image_path, num_singular_values)

# Calculate percentage of singular values used
percentage_singular_values = (num_singular_values / total_singular_values) * 100

# Display the images
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title(f"Original Image\nSize: {original_size / 1024:.2f} KB")
plt.imshow(original_img)
plt.axis("off")

plt.subplot(1, 2, 2)
plt.title(f"Compressed Image\nSVD Size: {reduced_size / 1024:.2f} KB\n{num_singular_values} Singular Values Used ({percentage_singular_values:.2f}%)")
plt.imshow(compressed_img)
plt.axis("off")

plt.tight_layout()
plt.show()

# Output the sizes and percentage in the console
print(f"Original image size: {original_size / 1024:.2f} KB")
print(f"Reduced SVD decomposition size: {reduced_size / 1024:.2f} KB")
print(f"Number of singular values used: {num_singular_values} ({percentage_singular_values:.2f}%)")
