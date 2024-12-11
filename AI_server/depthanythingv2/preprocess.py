import os
import cv2
import torch
import numpy as np
from .depth_anything_v2.dpt import DepthAnythingV2
from PIL import Image
import concurrent.futures
DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)

model_configs = {
    "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
    "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
    "vitl": {
        "encoder": "vitl",
        "features": 256,
        "out_channels": [256, 512, 1024, 1024],
    },
    "vitg": {
        "encoder": "vitg",
        "features": 384,
        "out_channels": [1536, 1536, 1536, 1536],
    },
}

encoder = "vits"  # or 'vits', 'vitb', 'vitg'

# Load the model
model = DepthAnythingV2(**model_configs[encoder])
model.load_state_dict(
    torch.load(
        f"depthanythingv2/checkpoints/depth_anything_v2_{encoder}.pth",
        map_location="cpu",
    )
)
model = model.to(DEVICE).eval()

def resize_image(raw_img, target_size=(1008, 1344)):
    """
    Efficiently resize the image to the target size.

    Parameters:
    - raw_img (numpy array): The input image in BGR format.
    - target_size (tuple): The target size (width, height). Defaults to (1008, 1344).

    Returns:
    - resized_img (numpy array): The resized image.
    """
    return cv2.resize(raw_img, target_size, interpolation=cv2.INTER_AREA)

def crop_image(raw_img, crop_percentage=0.1, target_size=(1008, 1344)):
    """
    Efficiently crop the image vertically by the specified percentage.

    Parameters:
    - raw_img (numpy array): The input image in BGR format.
    - crop_percentage (float): The percentage of the image to crop from the top and bottom.
    - target_size (tuple): Size to resize image before cropping.

    Returns:
    - cropped_img (numpy array): The cropped image.
    """
    # Resize image first to ensure consistent processing
    # resized_img = resize_image(raw_img, target_size)
    h, w = raw_img.shape[:2]
    
    # Use integer math for faster cropping
    crop_width= int(w * crop_percentage)
    return raw_img[:, crop_width:w - crop_width]

def process_image(image, crop_percentage=0.1):
    """
    Efficiently process a single image (convert and crop).

    Parameters:
    - image: PIL Image object, numpy array, or file path.
    - crop_percentage (float): Percentage to crop from top and bottom.

    Returns:
    - cropped_img (numpy array): The cropped image in BGR format.
    """
    # Optimize image loading
    if isinstance(image, Image.Image):
        # Direct conversion for PIL images
        raw_img = np.array(image)[:, :, :3]  # Ensure 3 channels
    elif isinstance(image, str):
        # Use faster imread with color flag
        raw_img = cv2.imread(image, cv2.IMREAD_COLOR)
        if raw_img is None:
            raise ValueError(f"Could not read image: {image}")
    elif isinstance(image, np.ndarray):
        raw_img = image
    else:
        raise TypeError("Unsupported image type")
    
    # Ensure RGB to BGR conversion if needed
    if raw_img.shape[2] == 3 and raw_img.dtype == np.uint8:
        if raw_img.shape[2] == 3 and raw_img.dtype == np.uint8:
            raw_img = cv2.cvtColor(raw_img, cv2.COLOR_RGB2BGR)
    
    return crop_image(raw_img, crop_percentage)

def crop_images_parallel(images, crop_percentage=0.1, max_workers=None):
    """
    Crop a batch of images in parallel with optimized processing.

    Parameters:
    - images (list): A list of images (PIL, numpy, or file paths).
    - crop_percentage (float): Percentage to crop from top and bottom.
    - max_workers (int, optional): Maximum number of worker processes.

    Returns:
    - cropped_images (list): A list of cropped images.
    """
    # Use None to default to number of CPU cores
    if max_workers is None:
        max_workers = os.cpu_count()
    
    # Use a more efficient parallel processing approach
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        try:
            cropped_images = list(executor.map(process_image, images, 
                                               [crop_percentage]*len(images)))
        except Exception as e:
            print(f"Error in parallel processing: {e}")
            cropped_images = []
    
    return cropped_images

def process_image_task(depth, resized_image, depth_threshold):
    """
    Process a single image and apply the threshold mask.
    
    Parameters:
    - depth (np.ndarray): Depth map for the image.
    - resized_image (np.ndarray): Resized image as a NumPy array.
    - depth_threshold (float): The threshold value to create the mask.
    
    Returns:
    - pil_image (PIL.Image): Processed image after applying the mask.
    """
    mask = (depth < depth_threshold)  # Create mask

    # Apply mask to black out pixels
    blacked_out_img = resized_image.copy()
    blacked_out_img[mask] = 0

    # Convert to grayscale using OpenCV
    gray_blacked_out_img = cv2.cvtColor(blacked_out_img, cv2.COLOR_BGR2GRAY)

    # Convert back to PIL Image
    pil_image = Image.fromarray(gray_blacked_out_img)
    return pil_image

import time

def background_cut_batch(images, depth_threshold = 0.5, crop_percentage = 0.1):
    """
    Process a list of images to apply background cutting using depth maps.

    Parameters:
    - images (list): A list of Pillow Image objects or file paths to images.

    Returns:
    - processed_images (list): A list of processed images (PIL Image objects) with the background cut.
    """
    processed_images = []

    # Step 2: Stack images to create a batch (stack along the first axis)
    batch = np.stack(images, axis=0)  # Shape: (batch_size, 1080, 1980, 3)

    # Step 3: Get the depth map for the entire batch

    depth_batch = model.infer_image(batch)  # HxW depth maps for the entire batch (numpy array)
    depth_batch = (depth_batch - depth_batch.min()) / (depth_batch.max() - depth_batch.min())  # Normalize depth
    images = np.array(images)
    # Step 4: Apply background cutting for each image in the batch
    args = [(depth_batch[i], images[i], depth_threshold) for i in range(len(images))]
 
    # Parallelize the image processing
    with concurrent.futures.ThreadPoolExecutor() as executor:
        processed_images = list(executor.map(lambda p: process_image_task(*p), args))
    
    return processed_images
    

def background_cut(images, depth_threshold = 0.5):
    """
    Process a list of images to apply background cutting using depth maps.

    Parameters:
    - images (list): A list of Pillow Image objects or file paths to images.

    Returns:
    - processed_images (list): A list of processed images (PIL Image objects) with the background cut.
    """
    processed_images = []

    for image in images:
        # Step 1: Load the image (handle both Pillow Image and file paths)
        if isinstance(image, Image.Image):
            raw_img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        else:
            raw_img = cv2.imread(image)

        # Step 2: Get the depth map for the current image
        depth = model.infer_image(raw_img)  # HxW raw depth map in numpy
        depth = (depth - depth.min()) / (depth.max() - depth.min())  # Normalize depth

        # Step 3: Create mask based on depth threshold
        mask = depth < depth_threshold  # Threshold to create mask (0.475, 0.495)
        # Step 4: Black out pixels where the mask is True
        blacked_out_img = raw_img.copy()
        blacked_out_img[mask] = 0
        gray_blacked_out_img = cv2.cvtColor(blacked_out_img, cv2.COLOR_BGR2GRAY)

        pil_image = Image.fromarray(gray_blacked_out_img)

        processed_images.append(pil_image)

    return processed_images
    


def process_and_blackout_image(image_path):
    # Step 1: Load the image
    raw_img = cv2.imread(image_path)

    # Step 2: Get the depth map
    depth = model.infer_image(raw_img)  # HxW raw depth map in numpy
    depth = (depth - depth.min()) / (depth.max() - depth.min())  # Normalize depth

    print(depth.max())
    # Step 3: Create mask based on depth threshold
    mask = depth < 0.5
    # 0.475

    # Step 4: Black out pixels where the mask is True
    blacked_out_img = raw_img.copy()
    blacked_out_img[mask] = 0
    gray_blacked_out_img = cv2.cvtColor(blacked_out_img, cv2.COLOR_BGR2GRAY)
    # Step 5: Save the blacked-out image back to the original path (replacing the old image)
    cv2.imwrite(image_path, gray_blacked_out_img)


def process_images_in_directory(root_directory):
    # Traverse all files and subdirectories in the given root directory
    for dirpath, _, filenames in os.walk(root_directory):
        for filename in filenames:
            # Check if the file is an image (adjust extensions as needed)
            if filename.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
                image_path = os.path.join(dirpath, filename)
                print(f"Processing {image_path}")
                process_and_blackout_image(image_path)


# Example usage:
root_directory = r"C:\My_Laptop\Repo\Palm-Print-Identification-System\AI_server\mambavision\raw\realistic-test\bg-cut\Real-bg-cut-rotate-shilf"
process_images_in_directory(root_directory)
