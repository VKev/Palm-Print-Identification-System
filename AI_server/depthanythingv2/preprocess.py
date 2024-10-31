import os
import cv2
import torch
import numpy as np
from depth_anything_v2.dpt import DepthAnythingV2
from PIL import Image

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

encoder = "vitb"  # or 'vits', 'vitb', 'vitg'

# Load the model
model = DepthAnythingV2(**model_configs[encoder])
model.load_state_dict(
    torch.load(
        f"checkpoints/depth_anything_v2_{encoder}.pth",
        map_location="cpu",
    )
)
model = model.to(DEVICE).eval()


def background_cut(image):
    # Step 1: Load the image
    if isinstance(image, Image.Image):
        raw_img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    else:
        raw_img = cv2.imread(image)

    # Step 2: Get the depth map
    depth = model.infer_image(raw_img)  # HxW raw depth map in numpy
    depth = (depth - depth.min()) / (depth.max() - depth.min())  # Normalize depth

    # Step 3: Create mask based on depth threshold
    mask = depth < 0.495
    # 0.475

    # Step 4: Black out pixels where the mask is True
    blacked_out_img = raw_img.copy()
    blacked_out_img[mask] = 0
    gray_blacked_out_img = cv2.cvtColor(blacked_out_img, cv2.COLOR_BGR2GRAY)
    # Step 5: Save the blacked-out image back to the original path (replacing the old image)
    if isinstance(image, Image.Image):
        # Convert NumPy array back to Pillow Image
        return Image.fromarray(gray_blacked_out_img).convert("RGB")
    else:
        # Return as NumPy array (OpenCV format)
        return gray_blacked_out_img


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
