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

def darken_images(input_path, regex_pattern, darken_amount=0.75):
    """
    Darkens images in the input path that match the regex pattern by a specified amount.

    Parameters:
    - input_path (str): Path to the directory containing images.
    - regex_pattern (str): Regex pattern to match specific filenames.
    - darken_amount (float): The factor by which to darken the image (e.g., 0.8 for slight darkening).
    """
    # Compile the regex pattern
    pattern = re.compile(regex_pattern)

    # Walk through the input directory
    for dirpath, dirnames, filenames in os.walk(input_path):
        for filename in filenames:
            # Check if the filename matches the regex pattern
            if pattern.match(filename):
                # Construct full file path
                source_file = os.path.join(dirpath, filename)
                # Open the image, convert to grayscale, and darken
                try:
                    with open(source_file, "rb") as f:
                        img = Image.open(f)
                        gray_img = img.convert("L")  # Convert to grayscale
                        enhancer = ImageEnhance.Brightness(gray_img)
                        darkened_img = enhancer.enhance(darken_amount)  # Apply darkening
                        darkened_img.save(source_file)  # Save it back with the same name
                        print(f"Darkened and saved: {source_file}")

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
                        enhanced_img = enhancer.enhance(1.3)
                        enhanced_img.save(
                            source_file
                        )  # Save it back with the same name
                        print(f"Converted and saved: {source_file}")

                except Exception as e:
                    pass
                    print(f"Error processing file {source_file}: {e}")


# Example usage
if __name__ == "__main__":
    input_path = r"AI_server\mambavision\raw\realistic-test\bg-cut\Real-bg-cut-rotate-shilf-roi-enhance"
    # regex_pattern = r"^\d{3}P_(PR|PL)\d{2}\.bmp$"  # Adjust the regex as needed
    regex_pattern = r".+\.(jpg|jpeg|png|bmp|gif|JPG)$"  # Adjust the regex as needed

    enhance_images(input_path, regex_pattern)
