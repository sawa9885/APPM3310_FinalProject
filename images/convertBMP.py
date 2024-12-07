import os
from PIL import Image

# Folder containing PNG images
folder_path = "images"  # Replace with your folder path

# Process PNG images in the folder
try:
    for i in range(1, 25):  # Loop through filenames '1.png' to '24.png'
        input_file = os.path.join(folder_path, f"{i}.png")
        output_file = os.path.join(folder_path, f"{i}.bmp")
        
        # Open the image and save as BMP
        with Image.open(input_file) as img:
            img.save(output_file, format="BMP")
            print(f"Converted {input_file} to {output_file}")

    print("All images converted successfully!")
except FileNotFoundError as e:
    print(f"File not found: {e}")
except Exception as e:
    print(f"An error occurred: {e}")
