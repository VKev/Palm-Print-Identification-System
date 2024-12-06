import base64
from PIL import Image
import io
from concurrent.futures import ThreadPoolExecutor

def decode_single_image(base64_str):
    """
    Decode a single base64 string into a PIL Image.
    
    Parameters:
    - base64_str (str): A base64 encoded image string.
    
    Returns:
    - PIL Image object or None if decoding fails.
    """
    try:
        # Remove the `data:image/...;base64,` prefix if it exists
        if "," in base64_str:
            base64_str = base64_str.split(",")[1]
        
        # Decode the base64 string
        image_data = base64.b64decode(base64_str)
        
        # Convert the binary data to a PIL image
        image = Image.open(io.BytesIO(image_data))
        return image
    except Exception as e:
        print(f"Error decoding image: {e}")
        return None

def decode_base64_images(base64_str_list):
    """
    Decode a list of base64 strings into a list of PIL Images efficiently using concurrency.

    Parameters:
    - base64_str_list (list): A list of base64 encoded strings.

    Returns:
    - List of PIL Image objects or None for failed decodings.
    """
    # Use ThreadPoolExecutor to process images concurrently
    with ThreadPoolExecutor() as executor:
        images = list(executor.map(decode_single_image, base64_str_list))

    return images

import base64
import io
import numpy as np
import cv2
from PIL import Image

def decode_base64_image(base64_string):
    try:
        # Remove data URL prefix if present
        if base64_string.startswith('data:image'):
            base64_string = base64_string.split(',')[1]
        
        # Ensure proper base64 padding
        base64_string = base64_string.replace(' ', '+')
        if len(base64_string) % 4 != 0:
            base64_string += '=' * (4 - (len(base64_string) % 4))
        
        # Decode base64 to bytes
        img_bytes = base64.b64decode(base64_string)
        
        # Convert to numpy array
        nparr = np.frombuffer(img_bytes, np.uint8)
        
        # Decode image using OpenCV
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        
        return pil_img
    
    except Exception as e:
        print(f"Error decoding base64 image: {e}")
        return None

# Usage example
def decode_base64_images_for_inference(base64_images):
    images = []
    for base64_img in base64_images:
        img = decode_base64_image(base64_img)
        if img is not None:
            images.append(img)
    return images