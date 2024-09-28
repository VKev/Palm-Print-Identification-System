import os
import re

import os
import re

def rename_images_with_suffix_in_path(path):
    # List to store image file paths
    image_files = []
    
    # Traverse through the directory and subdirectories
    for root, _, files in os.walk(path):
        for file in files:
            # Check if the file is an image with the specified naming pattern
            if re.match(r'^\d{5}_s2\.bmp$', file):
                image_files.append(os.path.join(root, file))
    
    # Rename the images based on the specified format
    for index, old_file in enumerate(image_files):
        # Calculate class number based on index
        class_number = index // 10
        # Extract the base name without the suffix
        base_name = os.path.basename(old_file).replace('_s2.bmp', '')
        # Create new file name keeping the s2 suffix
        new_file_name = f"{int(base_name):05d}_s2_{class_number}.bmp"
        # Get the new file path
        new_file_path = os.path.join(os.path.dirname(old_file), new_file_name)
        # Rename the file
        os.rename(old_file, new_file_path)
        print(f'Renamed: {old_file} -> {new_file_path}')

# Example usage:
# rename_images_with_suffix_in_path('C:\\My_Laptop\\Repo\\MambaVision')


def rename_images_in_path(path):
    # List to store image file paths
    image_files = []
    
    # Traverse through the directory and subdirectories
    for root, _, files in os.walk(path):
        for file in files:
            # Check if the file is an image with the specified naming pattern
            if re.match(r'^\d{5}_s1_s1\.bmp$', file):
                image_files.append(os.path.join(root, file))
    
    # Rename the images based on the specified format
    for index, old_file in enumerate(image_files):
        # Calculate class number based on index
        class_number = index // 10
        # Create new file name
        new_file_name = f"{int(re.sub('_s1_s1\.bmp$', '', os.path.basename(old_file))):05d}_s1_{class_number}.bmp"
        # Get the new file path
        new_file_path = os.path.join(os.path.dirname(old_file), new_file_name)
        # Rename the file
        os.rename(old_file, new_file_path)
        print(f'Renamed: {old_file} -> {new_file_path}')

def rename_images_in_groups(input_dir, prefix="", suffix="_s0_", extension=".bmp"):
    # Get all images in the directory and subfolders
    image_files = []
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.endswith(extension):
                image_files.append(os.path.join(root, file))

    # Sort the images to maintain a consistent order
    image_files.sort()

    # Rename each image in groups of 10
    for i, image_path in enumerate(image_files):
        # Calculate group number and within-group index
        group_number = i // 10
        image_number_in_group = i % 10 + 1
        
        # Create new name with padded numbers
        new_name = f"{prefix}{str(i + 1).zfill(5)}{suffix}{group_number}{extension}"
        
        # Construct the new file path
        new_image_path = os.path.join(os.path.dirname(image_path), new_name)
        
        # Rename the image
        os.rename(image_path, new_image_path)
        print(f"Renamed {image_path} to {new_image_path}")

# Example usage
rename_images_in_groups(r"C:\My_Laptop\Repo\MambaVision\AI_server\roiextraction\data\grayscale\001")

# Example usage:
# rename_images_with_suffix_in_path(r'C:\My_Laptop\Repo\MambaVision\AI_server\roiextraction\data\grayscale\003')
