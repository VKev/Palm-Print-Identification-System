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


def move_images_to_root(path):
    # Define the allowed image file extensions
    image_extensions = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".webp"}

    # Walk through all subdirectories and files
    for root, _, files in os.walk(path):
        for file in files:
            # Check if the file has a valid image extension
            if os.path.splitext(file)[1].lower() in image_extensions:
                source = os.path.join(root, file)
                destination = os.path.join(path, file)

                # If a file with the same name exists in the destination, rename it
                if os.path.exists(destination):
                    base_name, extension = os.path.splitext(file)
                    counter = 1
                    new_file_name = f"{base_name}_{counter}{extension}"
                    destination = os.path.join(path, new_file_name)

                    # Keep incrementing the counter until a unique file name is found
                    while os.path.exists(destination):
                        counter += 1
                        new_file_name = f"{base_name}_{counter}{extension}"
                        destination = os.path.join(path, new_file_name)

                # Move the image to the root path
                shutil.move(source, destination)
                print(f"Moved: {source} -> {destination}")


# Example usage
if __name__ == "__main__":
    move_images(
        r"C:\My_Laptop\Repo\Palm-Print-Identification-System\AI_server\mambavision\raw\tett\015",
        r"C:\My_Laptop\Repo\Palm-Print-Identification-System\AI_server\mambavision\raw\tett",
        r".*",
    )
    # move_images(
    #     r"C:\My_Laptop\Repo\Palm-Print-Identification-System\AI_server\mambavision\raw\001",
    #     r"C:\My_Laptop\Repo\Palm-Print-Identification-System\AI_server\mambavision\raw\004",
    #     r".*s3.*",
    # )
