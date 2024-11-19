import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import torch
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


def apply_gabor_filter_multi_orientations(image, ksize=31, sigma=2.0, lambd=10.0, gamma=0.5, psi=0, num_orientations=8):
    """Apply Gabor filter at multiple orientations and combine the results"""
    filtered_images = []
    thetas = np.linspace(0, np.pi, num_orientations, endpoint=False)
    for theta in thetas:
        kern = cv2.getGaborKernel((ksize, ksize), sigma, theta, lambd, gamma, psi, ktype=cv2.CV_32F)
        if kern.sum() != 0:
            kern /= kern.sum()
        filtered = cv2.filter2D(image, cv2.CV_32F, kern)
        filtered_images.append(filtered)
    # Combine the filtered images by taking the maximum response at each pixel
    combined_filtered = np.maximum.reduce(filtered_images)
    # Normalize the combined image
    combined_filtered = cv2.normalize(combined_filtered, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return combined_filtered

def process_images_with_gabor(input_path, output_path, gabor_params=None):
    """
    Recursively process all images in input_path and its subfolders with Gabor filters to detect vertical and horizontal edges.
    
    Args:
        input_path (str): Path to input directory containing images
        output_path (str): Path to save processed images
        gabor_params (dict, optional): Parameters for Gabor filter
    """
    if gabor_params is None:
        gabor_params = {
            'ksize': 31,
            'sigma': 4.0,
            'lambda_': 10.0,
            'gamma': 0.5,
            'psi': 0
        }
    
    # Create output directory if it doesn't exist
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Supported image extensions
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    
    # Process all images in directory and subdirectories
    for img_path in Path(input_path).rglob('*'):
        if img_path.suffix.lower() in image_extensions:
            try:
                # Read image
                img = cv2.imread(str(img_path))
                if img is None:
                    print(f"Failed to read image: {img_path}")
                    continue
                
                # Convert to grayscale if image is color
                if len(img.shape) == 3:
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                else:
                    gray = img
                
                # Apply Gabor filter
                filtered = apply_gabor_filter_multi_orientations(
                    gray,
                    ksize=gabor_params['ksize'],
                    sigma=gabor_params['sigma'],
                    lambd=gabor_params['lambda_'],
                    gamma=gabor_params['gamma'],
                    psi=gabor_params['psi']
                )
                
                # Create corresponding output path maintaining directory structure
                relative_path = img_path.relative_to(input_path)
                output_file = output_path / relative_path
                output_file.parent.mkdir(parents=True, exist_ok=True)
                
                # Save processed image
                cv2.imwrite(str(output_file), filtered)
                print(f"Processed: {img_path} -> {output_file}")
                
            except Exception as e:
                print(f"Error processing {img_path}: {str(e)}")

# Example usage
if __name__ == "__main__":
    input_dir = r'C:\My_Laptop\Repo\Palm-Print-Identification-System\AI_server\mambavision\raw\train'
    output_dir = r'C:\My_Laptop\Repo\Palm-Print-Identification-System\AI_server\mambavision\raw\train_gabor_filter'
    
    # Custom Gabor parameters (optional)
    gabor_params = {
        'ksize': 15,       # Smaller kernel size for fine details
        'sigma': 2.0,      # Standard deviation of the Gaussian envelope
        'lambda_': 5.0,    # Wavelength of the sinusoidal factor
        'gamma': 0.5,      # Spatial aspect ratio
        'psi': 0           # Phase offset
    }
    
    process_images_with_gabor(input_dir, output_dir, gabor_params)


# # Example usage
# if __name__ == "__main__":
#     input_path = r"AI_server\mambavision\raw\realistic-test\bg-cut\Real-bg-cut-rotate-shilf-roi-enhance"
#     # regex_pattern = r"^\d{3}P_(PR|PL)\d{2}\.bmp$"  # Adjust the regex as needed
#     regex_pattern = r".+\.(jpg|jpeg|png|bmp|gif|JPG)$"  # Adjust the regex as needed

#     # enhance_images(input_path, regex_pattern)
#     gabor_filter_palmprint(r'C:\My_Laptop\Repo\Palm-Print-Identification-System\AI_server\mambavision\raw\train',r'C:\My_Laptop\Repo\Palm-Print-Identification-System\AI_server\mambavision\raw\train_gabor_filter')
