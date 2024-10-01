import cv2
import torch
import numpy as np
import matplotlib
from depth_anything_v2.dpt import DepthAnythingV2

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

encoder = "vitl"  # or 'vits', 'vitb', 'vitg'

model = DepthAnythingV2(**model_configs[encoder])
model.load_state_dict(
    torch.load(
        f"C:\My_Laptop\Repo\Palm-Print-Identification-System\AI_server\Depth-Anything-V2\checkpoints\depth_anything_v2_{encoder}.pth",
        map_location="cpu",
    )
)
model = model.to(DEVICE).eval()

raw_img = cv2.imread(
    r"C:\My_Laptop\Repo\Palm-Print-Identification-System\AI_server\mambavision\raw\test\test.jpg"
)
depth = model.infer_image(raw_img)  # HxW raw depth map in numpy
depth = (depth - depth.min()) / (depth.max() - depth.min())
# print(depth)

mask = depth < 0.47

# Step 5: Black out pixels in the image where mask is True
blacked_out_img = raw_img.copy()
blacked_out_img[mask] = 0

scale_percent = 15  # Scale down to 50% of original size
width = int(blacked_out_img.shape[1] * scale_percent / 100)
height = int(blacked_out_img.shape[0] * scale_percent / 100)
dim = (width, height)

# Resize the image
resized_img = cv2.resize(blacked_out_img, dim, interpolation=cv2.INTER_AREA)
cv2.imshow("Blacked Out Image", resized_img)
output_path = r"C:\My_Laptop\Repo\Palm-Print-Identification-System\AI_server\mambavision\raw\test\test.jpg"
cv2.imwrite(output_path, blacked_out_img)  # Saves the original-size blacked-out image
cv2.waitKey(0)
cv2.destroyAllWindows()
