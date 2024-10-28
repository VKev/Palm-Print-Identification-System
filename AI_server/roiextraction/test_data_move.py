import os
import re
import shutil

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
input_folder = 'C:\My_Laptop\Repo\Sapienza University Mobile Palmprint Database(SMPD)'
output_folder = 'C:\My_Laptop\Repo\\testset'
move_filtered_images(input_folder, output_folder)
