import torch
from PIL import Image
from torchvision import transforms
from .models import mamba_vision_L2
import cv2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(checkpoint_path):
    model = mamba_vision_L2()  # Initialize the model without pretrained weights
    model.load_state_dict(torch.load(checkpoint_path))  # Load the fine-tuned weights
    model.to(device)
    model.eval()
    return model


# images is [image1, image2]
# image = Image.open(image_path).convert("RGB")
def preprocess_images(images):
    # Define the preprocessing transformations

    return torch.stack(images).to(device)


def get_embbedding(model, preprocess_images):
    input_tensor = preprocess_images  # Preprocess all images
    with torch.no_grad():  # Disable gradient computation
        outputs = model.forward_features(input_tensor)  # Forward pass
    return outputs
