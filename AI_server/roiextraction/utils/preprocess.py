import os
import re
from PIL import Image, ImageEnhance


def convert_images_to_grayscale(input_path, regex_pattern):
    # Compile the regex pattern
    pattern = re.compile(regex_pattern)

    # Walk through the input directory
    for dirpath, dirnames, filenames in os.walk(input_path):
        for filename in filenames:
            # Check if the filename matches the regex pattern
            if pattern.match(filename):
                # Construct full file path
                source_file = os.path.join(dirpath, filename)

                # Open the image, convert to grayscale, and save
                try:
                    with Image.open(source_file) as img:
                        gray_img = img.convert("L")  # Convert to grayscal
                        gray_img.save(source_file)  # Save it back with the same name
                        print(f"Converted and saved: {source_file}")

                except Exception as e:
                    print(f"Error processing file {source_file}: {e}")


def enhance_images(input_path, regex_pattern):
    # Compile the regex pattern
    pattern = re.compile(regex_pattern)

    # Walk through the input directory
    for dirpath, dirnames, filenames in os.walk(input_path):
        for filename in filenames:
            # Check if the filename matches the regex pattern
            if pattern.match(filename):
                # Construct full file path
                source_file = os.path.join(dirpath, filename)
                # Open the image, convert to grayscale, and save
                try:
                    with open(source_file, "rb") as f:
                        img = Image.open(f)
                        gray_img = img.convert("L")  # Convert to grayscale
                        enhancer = ImageEnhance.Contrast(gray_img)
                        enhanced_img = enhancer.enhance(5.0 / 3.0)
                        enhanced_img.save(
                            source_file
                        )  # Save it back with the same name
                        print(f"Converted and saved: {source_file}")

                except Exception as e:
                    pass
                    print(f"Error processing file {source_file}: {e}")


# Example usage
if __name__ == "__main__":
    input_path = (
        r"C:\My_Laptop\Repo\Palm-Print-Identification-System\AI_server\mambavision\test"
    )
    # regex_pattern = r"^\d{3}P_(PR|PL)\d{2}\.bmp$"  # Adjust the regex as needed
    regex_pattern = r".+\.(jpg|jpeg|png|bmp|gif|JPG)$"  # Adjust the regex as needed

    enhance_images(input_path, regex_pattern)
