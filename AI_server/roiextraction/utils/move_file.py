import os
import re
import shutil


def move_and_rename_images(input_path, output_path, regex_pattern, add_string):
    # Compile the regex pattern
    pattern = re.compile(regex_pattern)

    # Ensure the output directory exists
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Walk through the input directory
    for dirpath, dirnames, filenames in os.walk(input_path):
        for filename in filenames:
            # Check if the filename matches the regex pattern
            if pattern.match(filename):
                # Construct full file path
                source_file = os.path.join(dirpath, filename)

                # Create the new filename
                new_filename = f"{os.path.splitext(filename)[0]}_{add_string}{os.path.splitext(filename)[1]}"
                destination_file = os.path.join(output_path, new_filename)

                # Move and rename the file
                shutil.move(source_file, destination_file)
                print(f"Moved and renamed: {source_file} to {destination_file}")

        # Move subdirectories if needed (optional, but keeping for completeness)
        for dirname in dirnames:
            # Check if the directory matches the regex pattern
            if pattern.match(dirname):
                source_dir = os.path.join(dirpath, dirname)
                destination_dir = os.path.join(output_path, dirname)
                shutil.move(source_dir, destination_dir)
                print(f"Moved directory: {source_dir} to {destination_dir}")


def move_images(input_path, output_path, regex_pattern):
    # Compile the regex pattern
    pattern = re.compile(regex_pattern)

    # Walk through the input directory
    for dirpath, dirnames, filenames in os.walk(input_path):
        for filename in filenames:
            # Check if the filename matches the regex pattern
            if pattern.match(filename):
                # Construct full file path
                source_file = os.path.join(dirpath, filename)

                # Move the file to the output path
                shutil.move(source_file, output_path)
                print(f"Moved: {source_file} to {output_path}")

        # Move subdirectories if needed
        for dirname in dirnames:
            # Check if the directory matches the regex pattern
            if pattern.match(dirname):
                source_dir = os.path.join(dirpath, dirname)
                shutil.move(source_dir, output_path)
                print(f"Moved directory: {source_dir} to {output_path}")


# Example usage
if __name__ == "__main__":
    input_path = r"C:\My_Laptop\Repo\Palm-Vein-Recognition-System\AI_Server\data\raw\ROI\session2"
    output_path = (
        r"C:\My_Laptop\Repo\Palm-Vein-Recognition-System\AI_Server\data\grayscale\003"
    )
    regex_pattern = r"^\d{5}\.bmp$"  # Adjust the regex as needed

    move_and_rename_images(input_path, output_path, regex_pattern, "s2")
