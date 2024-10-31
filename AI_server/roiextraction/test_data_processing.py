import os
import re
import shutil

import os
import re

def rename_files(input_path):
    """
    Renames files in the input path by swapping the two numeric parts.

    Parameters:
    - input_path (str): Path to the directory containing files to rename.
    """
    # Regex pattern to match filenames in the form "number_F_number.JPG"
    pattern = re.compile(r"(\d+)_P_(\d+)\.JPG", re.IGNORECASE)
    
    # Iterate through files in the directory
    for filename in os.listdir(input_path):
        match = pattern.match(filename)
        if match:
            # Extract the two numbers from the filename
            first_number = match.group(1)
            second_number = match.group(2)
            
            # Create the new filename by swapping the two numbers
            new_filename = f"{second_number}_P_{first_number}.JPG"
            
            # Construct full file paths
            source_file = os.path.join(input_path, filename)
            destination_file = os.path.join(input_path, new_filename)
            
            # Rename the file
            os.rename(source_file, destination_file)
            print(f"Renamed {filename} to {new_filename}")

# Example usage


def move_filtered_images(input_path, output_path):
    # Regular expression pattern for matching image names
    pattern = re.compile(r'^\d+_(RF|F)_\d+\.(jpg|jpeg|png)$', re.IGNORECASE)
    
    # Ensure output directory exists
    os.makedirs(output_path, exist_ok=True)
    
    # Walk through all files in the input directory and its subdirectories
    for root, _, files in os.walk(input_path):
        for file in files:
            # Check if file matches the pattern
            if pattern.match(file):
                # Move the file to the output directory
                source_file = os.path.join(root, file)
                shutil.move(source_file, os.path.join(output_path, file))
                print(f"Moved: {source_file} to {output_path}")

# Example usage
input_folder = 'C:\My_Laptop\Repo\Palm-Print-Identification-System\AI_server\mambavision\\raw\dataset-test\Sapienza-University-test - Copy'
output_folder = 'C:\My_Laptop\Repo\\testset'
# move_filtered_images(input_folder, output_folder)
rename_files(input_folder)
