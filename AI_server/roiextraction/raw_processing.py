import os
from PIL import Image
import numpy as np
import random

def rotate_and_shift_images(root_path, rotation_range=30, shift_range=10):
    """
    Rotates and shifts all images in the specified directory (including subdirectories) 
    and saves them back to their original locations.
    
    :param root_path: Path to the root directory containing images and subdirectories.
    :param rotation_range: Maximum rotation angle (in degrees) for random rotation.
    :param shift_range: Maximum shift in pixels for random shifting in both directions.
    """
    for dirpath, _, filenames in os.walk(root_path):
        for filename in filenames:
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.webp')):
                image_path = os.path.join(dirpath, filename)
                
                # Open the image
                with Image.open(image_path) as img:
                    # Random rotation
                    angle = random.uniform(-rotation_range, rotation_range)
                    img = img.rotate(angle)
                    
                    # Random shift
                    shift_x = random.randint(-shift_range, shift_range)
                    shift_y = random.randint(-shift_range, shift_range)
                    img = img.transform(img.size, Image.AFFINE, (1, 0, shift_x, 0, 1, shift_y))

                    # Save the modified image
                    img.save(image_path)
                    
                print(f"Processed and saved: {image_path}")

# Example usage:
rotate_and_shift_images(r'C:\My_Laptop\Repo\Palm-Print-Identification-System\AI_server\mambavision\raw\realistic-test\bg-cut\Real-bg-cut-rotate-shilf')
