import os
import re

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
            if re.match(r"^\d{5}_s2\.bmp$", file):
                image_files.append(os.path.join(root, file))

    # Rename the images based on the specified format
    for index, old_file in enumerate(image_files):
        # Calculate class number based on index
        class_number = index // 10
        # Extract the base name without the suffix
        base_name = os.path.basename(old_file).replace("_s2.bmp", "")
        # Create new file name keeping the s2 suffix
        new_file_name = f"{int(base_name):05d}_s2_{class_number}.bmp"
        # Get the new file path
        new_file_path = os.path.join(os.path.dirname(old_file), new_file_name)
        # Rename the file
        os.rename(old_file, new_file_path)
        print(f"Renamed: {old_file} -> {new_file_path}")


# Example usage:
# rename_images_with_suffix_in_path('C:\\My_Laptop\\Repo\\MambaVision')


def rename_images_in_path(path):
    # List to store image file paths
    image_files = []

    # Traverse through the directory and subdirectories
    for root, _, files in os.walk(path):
        for file in files:
            # Check if the file is an image with the specified naming pattern
            if re.match(r"^\d{5}_s1_s1\.bmp$", file):
                image_files.append(os.path.join(root, file))

    # Rename the images based on the specified format
    for index, old_file in enumerate(image_files):
        # Calculate class number based on index
        class_number = index // 10
        # Create new file name
        # new_file_name = f"{int(re.sub('_s1_s1\.bmp$', '', os.path.basename(old_file))):05d}_s1_{class_number}.bmp"
        # Get the new file path
        new_file_path = os.path.join(os.path.dirname(old_file), new_file_name)
        # Rename the file
        os.rename(old_file, new_file_path)
        print(f"Renamed: {old_file} -> {new_file_path}")


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


import os
import glob
import shutil


def rename_images_in_batches(root_path):
    # Get all image files from the directory and subdirectories
    image_paths = glob.glob(os.path.join(root_path, "**", "*.*"), recursive=True)

    # Filter only image files (adjust the extensions if needed)
    image_paths = [
        path
        for path in image_paths
        if path.lower().endswith((".png", ".jpg", ".jpeg", ".gif", ".bmp"))
    ]

    # Sort images to maintain consistent order
    image_paths.sort()

    # Initialize variables
    batch_size = 10
    image_index = 0

    # Loop through the images and rename them in batches of 10
    for i, image_path in enumerate(image_paths):
        # Calculate the batch index
        batch_index = i // batch_size

        # Construct the new file name
        new_filename = f"{i+1:05d}_s3_{batch_index}{os.path.splitext(image_path)[1]}"

        # Create the new image path with the new file name
        new_image_path = os.path.join(os.path.dirname(image_path), new_filename)

        # Rename the file
        shutil.move(image_path, new_image_path)
        image_index += 1


import os


def rename_images_in_path(path):
    # Define the allowed image file extensions
    image_extensions = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".webp"}

    # List to store all image paths for global indexing
    all_images = []

    # Walk through all subdirectories and collect all image paths
    for root, _, files in os.walk(path):
        for file in files:
            if os.path.splitext(file)[1].lower() in image_extensions:
                all_images.append(os.path.join(root, file))

    # Sort images for consistent renaming order (by original name)
    all_images.sort()

    # Iterate over images and rename them
    for index, image_path in enumerate(all_images):
        root, filename = os.path.split(image_path)
        name, extension = os.path.splitext(filename)

        # Split the original name by underscores to extract components
        try:
            person_id, _, palm_side, image_num = name.split("_")
        except ValueError:
            print(f"Skipping file with invalid format: {filename}")
            continue

        # Convert person_id to integer and pad to 5 digits
        person_id_int = int(person_id)
        image_index = index + 1  # Start the global index from 1 and increment

        # Determine the palm side: left palm (0) or right palm (1)
        palm_side_num = 0 if palm_side == "L" else 1

        # Generate the new filename
        new_name = f"{image_index:05d}_s4_{(person_id_int - 1) * 2 + palm_side_num}"
        new_filename = new_name + extension
        new_filepath = os.path.join(root, new_filename)

        # Rename the file
        os.rename(image_path, new_filepath)
        print(f"Renamed: {image_path} -> {new_filepath}")


# Example usage
# rename_images_in_path('/path/to/your/folder')


import os
import re


def rename_images(input_path):
    # Initialize an empty list to hold the file paths
    image_files = []

    # Walk through the input directory to collect all image files
    for dirpath, dirnames, filenames in os.walk(input_path):
        for filename in filenames:
            if filename.endswith(".JPG"):  # Adjust this if needed
                image_files.append(os.path.join(dirpath, filename))

    # Sort the image files to maintain order
    image_files.sort()

    # Rename images
    for index, full_path in enumerate(image_files):
        # Extract the current filename
        filename = os.path.basename(full_path)

        # Use regex to extract the parts of the filename
        match = re.match(r"(\d{3})_F_(\d+)\.JPG", filename)
        if match:
            person_index = match.group(1)  # e.g., '001'
            image_index = match.group(2)  # e.g., '0'

            # Construct the new filename
            new_filename = f"{index + 1:05d}_s5_{image_index}.JPG"  # 5-digit index
            new_full_path = os.path.join(os.path.dirname(full_path), new_filename)

            # Rename the file
            os.rename(full_path, new_full_path)
            print(f"Renamed: {full_path} to {new_full_path}")


import os
import re


def rename_images_in_batches(input_path):
    # Initialize an empty list to hold the file paths
    image_files = []

    # Walk through the input directory to collect all image files
    for dirpath, dirnames, filenames in os.walk(input_path):
        for filename in filenames:
            if filename.endswith(".JPG"):  # Adjust this if needed
                image_files.append(os.path.join(dirpath, filename))

    # Sort the image files to maintain order
    image_files.sort()

    # Process images in batches of 10
    for index, full_path in enumerate(image_files):
        # Determine the batch index (0 for first 10 images, 1 for next 10, etc.)
        batch_index = index // 10

        # Construct the new filename
        new_filename = f"{index + 1:05d}_s5_{batch_index}.JPG"  # 5-digit index
        new_full_path = os.path.join(os.path.dirname(full_path), new_filename)

        # Rename the file
        os.rename(full_path, new_full_path)
        print(f"Renamed: {full_path} to {new_full_path}")


# Usage
root_directory = r"C:\My_Laptop\Repo\Palm-Print-Identification-System\AI_server\roiextraction\data\raw\005"
rename_images_in_batches(root_directory)

# Example usage:

# Example usage
# rename_images_in_groups(r"C:\My_Laptop\Repo\MambaVision\AI_server\roiextraction\data\grayscale\001")

# Example usage:
# rename_images_with_suffix_in_path(r'C:\My_Laptop\Repo\MambaVision\AI_server\roiextraction\data\grayscale\003')
