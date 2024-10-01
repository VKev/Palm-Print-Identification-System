import os


def rename_images(base_path, output_path):
    # Store image paths
    image_paths = []

    # Traverse the directory and subdirectories
    for root, _, files in os.walk(base_path):
        for file in files:
            if file.endswith(".bmp"):  # Filter for .bmp files
                image_paths.append(os.path.join(root, file))

    # Sort the image paths (optional)
    image_paths.sort()

    # Rename images based on the required format
    for index, image_path in enumerate(image_paths):
        # Extract parts of the original filename
        original_name = os.path.basename(image_path)

        # Create a new filename based on the index
        new_name = (
            f"{index + 1:05d}_{original_name[6:]}"  # Retain the suffix after first part
        )
        new_path = os.path.join(output_path, new_name)

        # Rename the file
        os.rename(image_path, new_path)
        print(f"Renamed: {original_name} -> {new_name}")


import os
import re


def shift_group_index(base_path, shift, file_extension):
    # Traverse the directory and subdirectories
    for root, _, files in os.walk(base_path):
        for file in files:
            if file.endswith(file_extension):  # Filter for .bmp files
                original_path = os.path.join(root, file)

                # Use regex to match and capture parts of the filename
                match = re.match(r"(\d{5}_s\d)_(\d+)" + file_extension, file)
                if match:
                    prefix = match.group(1)  # '00001_s1'
                    group_index = int(match.group(2))  # Last number
                    new_group_index = group_index + shift  # Shift the group index

                    # Create the new filename
                    new_name = f"{prefix}_{new_group_index}" + file_extension
                    new_path = os.path.join(root, new_name)

                    # Rename the file
                    os.rename(original_path, new_path)
                    print(f"Renamed: {file} -> {new_name}")


# Example usage:
# shift_group_index('path/to/your/images', 10)


def shift_first_number(base_path, shift, file_extension):
    # Traverse the directory and subdirectories
    for root, _, files in os.walk(base_path):
        for file in files:
            if file.endswith(file_extension):  # Filter for .bmp files
                original_path = os.path.join(root, file)

                # Use regex to match and capture parts of the filename
                match = re.match(r"(\d{5})_(s\d)_(\d+)"+file_extension, file)
                if match:
                    first_number = int(match.group(1))  # First number
                    suffix = match.group(2)  # 's1', 's2', etc.
                    group_index = int(match.group(3))  # Last number

                    # Shift the first number
                    new_first_number = first_number + shift  # Shift the first number

                    # Create the new filename
                    new_name = f"{new_first_number:05d}_{suffix}_{group_index}"+file_extension
                    new_path = os.path.join(root, new_name)

                    # Rename the file
                    os.rename(original_path, new_path)
                    print(f"Renamed: {file} -> {new_name}")


# Example usage:
shift_first_number(
    r"C:\My_Laptop\Repo\Palm-Print-Identification-System\AI_server\mambavision\raw\train\002",
    15678,
    ".JPG",
)
