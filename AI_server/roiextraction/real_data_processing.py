import os
import re
import random
import shutil


def rename_images_by_parent_folder(input_path):
    """
    Renames images in each subfolder by using the frame number combined with its parent folder's identifier.

    Parameters:
    - input_path (str): Path to the directory containing subfolders.
    """
    # Regex pattern to match the parent folder (e.g., video_frames_1 -> 1)
    folder_pattern = re.compile(r"video_frames_(\d+)")
    # Regex pattern to match frame number in filenames (e.g., frame_1.jpg -> 1)
    file_pattern = re.compile(r"frame_(\d+)\.jpg", re.IGNORECASE)

    # Iterate through subfolders in the input directory
    for subfolder in os.listdir(input_path):
        folder_match = folder_pattern.match(subfolder)
        if folder_match:
            parent_id = folder_match.group(1)  # Extract the parent folder ID
            subfolder_path = os.path.join(input_path, subfolder)

            # Ensure it's a directory
            if os.path.isdir(subfolder_path):
                # Iterate through image files in the subfolder
                for filename in os.listdir(subfolder_path):
                    file_match = file_pattern.match(filename)
                    if file_match:
                        frame_number = file_match.group(1)  # Extract frame number
                        # Construct new filename
                        new_filename = f"{frame_number}_{parent_id}.jpg"
                        source_file = os.path.join(subfolder_path, filename)
                        destination_file = os.path.join(subfolder_path, new_filename)

                        # Rename the file
                        os.rename(source_file, destination_file)
                        print(f"Renamed {filename} to {new_filename}")


def move_all_files(source_path, destination_path):
    """
    Moves all files from the source path, including subdirectories, to the destination path.

    Parameters:
    - source_path (str): The path from which to move files.
    - destination_path (str): The path to which files should be moved.
    """
    # Ensure the destination path exists
    os.makedirs(destination_path, exist_ok=True)

    # Walk through the source directory
    for dirpath, _, filenames in os.walk(source_path):
        for filename in filenames:
            source_file = os.path.join(dirpath, filename)
            destination_file = os.path.join(destination_path, filename)

            # Move the file
            try:
                shutil.move(source_file, destination_file)
                print(f"Moved: {source_file} to {destination_file}")
            except Exception as e:
                print(f"Error moving file {source_file}: {e}")


# Example usage

import os
import shutil
from collections import defaultdict


def split_images_by_label(source_path, folder1, folder2):
    """
    Splits images by label and copies them to two specified folders.

    Parameters:
    - source_path (str): The path where the images are located.
    - folder1 (str): The path to the first folder where half of the images will be copied.
    - folder2 (str): The path to the second folder where the other half will be copied.
    """
    # Ensure the destination folders exist
    os.makedirs(folder1, exist_ok=True)
    os.makedirs(folder2, exist_ok=True)

    # Dictionary to hold images by label
    label_dict = defaultdict(list)

    # Walk through the source directory to categorize images by label
    for filename in os.listdir(source_path):
        if (
            filename.endswith(".jpg")
            or filename.endswith(".jpeg")
            or filename.endswith(".JPG")
            or filename.endswith(".bmp")
        ):
            # Extract the label from the filename
            label = filename.split("_")[-1].split(".")[
                0
            ]  # Get the last part before the extension
            label_dict[label].append(filename)

    # Split and copy images to the two folders
    for label, images in label_dict.items():
        half = len(images) // 2
        for i, image in enumerate(images):
            source_file = os.path.join(source_path, image)
            if i < half:
                destination_file = os.path.join(folder1, image)
            else:
                destination_file = os.path.join(folder2, image)

            # Copy the file to the appropriate destination
            try:
                shutil.copy(source_file, destination_file)
                print(f"Copied: {source_file} to {destination_file}")
            except Exception as e:
                print(f"Error copying file {source_file}: {e}")


def split_dataset_into_two(source_path, folder1, folder2):
    """
    Splits an entire dataset into two folders such that each half has different labels.

    Parameters:
    - source_path (str): The path where the images are located.
    - folder1 (str): The path to the first folder for one half of the dataset.
    - folder2 (str): The path to the second folder for the other half of the dataset.
    """
    # Ensure the destination folders exist
    os.makedirs(folder1, exist_ok=True)
    os.makedirs(folder2, exist_ok=True)

    # Dictionary to hold images by label
    label_dict = defaultdict(list)

    # Walk through the source directory to categorize images by label
    for filename in os.listdir(source_path):
        if (
            filename.endswith(".jpg")
            or filename.endswith(".jpeg")
            or filename.endswith(".JPG")
            or filename.endswith(".bmp")
        ):
            # Extract the label from the filename
            label = filename.split("_")[-1].split(".")[
                0
            ]  # Get the last part before the extension
            label_dict[label].append(filename)

    # Shuffle labels to ensure randomness
    labels = list(label_dict.keys())
    random.shuffle(labels)

    # Split labels into two halves
    half_size = len(labels) // 2
    labels1 = labels[:half_size]
    labels2 = labels[half_size:]

    # Copy images to the respective folders based on labels
    for label in labels1:
        for image in label_dict[label]:
            source_file = os.path.join(source_path, image)
            destination_file = os.path.join(folder1, image)
            shutil.copy(source_file, destination_file)
            print(f"Copied: {source_file} to {destination_file}")

    for label in labels2:
        for image in label_dict[label]:
            source_file = os.path.join(source_path, image)
            destination_file = os.path.join(folder2, image)
            shutil.copy(source_file, destination_file)
            print(f"Copied: {source_file} to {destination_file}")


# Example usage
# split_dataset_into_two(r"C:\My_Laptop\Repo\Palm-Print-Identification-System\AI_server\mambavision\raw\realistic-test\bg-cut\Roi-bg-cut-enhance-test-set", r"C:\My_Laptop\Repo\Palm-Print-Identification-System\AI_server\mambavision\raw\realistic-test\bg-cut\Register-unregister-half", r"C:\My_Laptop\Repo\Palm-Print-Identification-System\AI_server\mambavision\raw\realistic-test\bg-cut\Query-unregister-half")

# Example usage
# split_images_by_label(r"C:\My_Laptop\Repo\Palm-Print-Identification-System\AI_server\mambavision\raw\realistic-test\bg-cut\Roi-bg-cut-enhance-test-set", r"C:\My_Laptop\Repo\Palm-Print-Identification-System\AI_server\mambavision\raw\realistic-test\bg-cut\Register-half", r"C:\My_Laptop\Repo\Palm-Print-Identification-System\AI_server\mambavision\raw\realistic-test\bg-cut\Query-half")

# move_all_files(r"C:\My_Laptop\Repo\Palm-Print-Identification-System\AI_server\mambavision\raw\realistic-test\bg-cut\Roi-bg-cut-enhance - Copy", r"C:\My_Laptop\Repo\Palm-Print-Identification-System\AI_server\mambavision\raw\realistic-test\bg-cut\Roi-bg-cut-enhance-test-set")

# Example usage
# rename_images_by_parent_folder(r"C:\My_Laptop\Repo\Palm-Print-Identification-System\AI_server\mambavision\raw\realistic-test\bg-cut\Roi-bg-cut-enhance - Copy")


# NON BACKGROUND CUT:
# Example usage
# split_dataset_into_two(r"C:\My_Laptop\Repo\Palm-Print-Identification-System\AI_server\mambavision\raw\realistic-test\bg-cut\Rotate-Shilf\Real-bg-cut-rotate-shilf-roi-enhance", r"C:\My_Laptop\Repo\Palm-Print-Identification-System\AI_server\mambavision\raw\realistic-test\bg-cut\Rotate-Shilf\Register-unregistered-half", r"C:\My_Laptop\Repo\Palm-Print-Identification-System\AI_server\mambavision\raw\realistic-test\bg-cut\Rotate-Shilf\Query-unregister-half")
# split_images_by_label(r"C:\My_Laptop\Repo\Palm-Print-Identification-System\AI_server\mambavision\raw\train", r"C:\My_Laptop\Repo\Palm-Print-Identification-System\AI_server\mambavision\raw\Query-register-half", r"C:\My_Laptop\Repo\Palm-Print-Identification-System\AI_server\mambavision\raw\Register-register-half")

# Example usage
split_dataset_into_two(
    r"C:\My_Laptop\Repo\Palm-Print-Identification-System\AI_server\mambavision\raw\dataset-test\Sapienza University Mobile Palmprint Database(SMPD)\Sapienza-University-test-roi-by-our",
    r"C:\My_Laptop\Repo\Palm-Print-Identification-System\AI_server\mambavision\raw\dataset-test\Sapienza University Mobile Palmprint Database(SMPD)\Register-unregister-half",
    r"C:\My_Laptop\Repo\Palm-Print-Identification-System\AI_server\mambavision\raw\dataset-test\Sapienza University Mobile Palmprint Database(SMPD)\Query-unregister-half",
)

# move_all_files(r"C:\My_Laptop\Repo\Palm-Print-Identification-System\AI_server\mambavision\raw\realistic-test\non-bg-cut\Real-roi-enhance", r"C:\My_Laptop\Repo\Palm-Print-Identification-System\AI_server\mambavision\raw\realistic-test\non-bg-cut\Real-roi-enhance-test-set")

# Example usage
# rename_images_by_parent_folder(r"C:\My_Laptop\Repo\Palm-Print-Identification-System\AI_server\mambavision\raw\realistic-test\non-bg-cut\Real-roi-enhance")


# split_dataset_into_two(r"C:\My_Laptop\Repo\Palm-Print-Identification-System\AI_server\mambavision\raw\realistic-test\non-bg-cut\Rotate-Shilf\Real-raw-rotate-shift-roi-enhance", r"C:\My_Laptop\Repo\Palm-Print-Identification-System\AI_server\mambavision\raw\realistic-test\non-bg-cut\Rotate-Shilf\DQuery-half", r"C:\My_Laptop\Repo\Palm-Print-Identification-System\AI_server\mambavision\raw\realistic-test\non-bg-cut\Rotate-Shilf\DRegister-half")

# Example usage
# split_images_by_label(r"C:\My_Laptop\Repo\Palm-Print-Identification-System\AI_server\mambavision\raw\realistic-test\bg-cut\Rotate-Shilf\Real-bg-cut-rotate-shilf-roi-enhance", r"C:\My_Laptop\Repo\Palm-Print-Identification-System\AI_server\mambavision\raw\realistic-test\bg-cut\Rotate-Shilf\SQuery-half", r"C:\My_Laptop\Repo\Palm-Print-Identification-System\AI_server\mambavision\raw\realistic-test\bg-cut\Rotate-Shilf\SRegister-half")

# move_all_files(r"C:\My_Laptop\Repo\Palm-Print-Identification-System\AI_server\mambavision\raw\realistic-test\bg-cut\Real-bg-cut-rotate-shilf-roi-enhance", r"C:\My_Laptop\Repo\Palm-Print-Identification-System\AI_server\mambavision\raw\realistic-test\bg-cut\Real-bg-cut-rotate-shilf-roi-enhance")

# Example usage
# rename_images_by_parent_folder(r"C:\My_Laptop\Repo\Palm-Print-Identification-System\AI_server\mambavision\raw\realistic-test\bg-cut\Real-bg-cut-rotate-shilf")
