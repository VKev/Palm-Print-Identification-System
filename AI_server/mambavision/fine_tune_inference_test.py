import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
from models import mamba_vision_L2
from scipy.spatial.distance import cosine

# Load the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = mamba_vision_L2()  # Initialize the model without pretrained weights
model.load_state_dict(
    torch.load(r"checkpoints/batch4/fine_tuned_mamba_vision_L2_e11.pth")
)  # Load the fine-tuned weights
model.to(device)
model.eval()  # Set the model to evaluation mode


# Image preprocessing
def preprocess_images(image_paths):
    preprocess = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),  # Convert to tensor
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),  # Normalize
        ]
    )

    images = []
    for image_path in image_paths:
        image = Image.open(image_path).convert("RGB")
        image_tensor = preprocess(image)
        images.append(image_tensor)

    return torch.stack(images).to(device)


def run_inference(image_paths, batch_size=32):
    all_outputs = []

    total_batches = (len(image_paths) + batch_size - 1) // batch_size
    progress_bar = tqdm(total=total_batches, desc="Processing Batches", unit="batch")

    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i : i + batch_size]
        input_tensor = preprocess_images(batch_paths)

        with torch.no_grad():
            batch_outputs = model.forward_features(input_tensor)

        all_outputs.append(batch_outputs)

        progress_bar.update(1)

    progress_bar.close()

    return torch.cat(all_outputs, dim=0)


import os


def get_image_paths(directory):
    image_paths = []
    # Walk through the directory and its subdirectories
    for root, _, files in os.walk(directory):
        for file in files:
            # Check if the file is an image (you can extend this check based on file extensions)
            if file.lower().endswith((".png", ".jpg", ".jpeg", ".gif", ".bmp")):
                # Construct the full path to the image file
                image_path = os.path.join(root, file)
                # Add the image path to the list
                image_paths.append(image_path)
    return image_paths


def l2_normalize(embeddings):
    # Calculate the L2 norm for each vector
    l2_norm = torch.norm(embeddings, p=2, dim=1, keepdim=True)
    # Normalize each vector by dividing by its L2 norm
    normalized_embeddings = embeddings / l2_norm
    return normalized_embeddings


# Example usage
image_paths = get_image_paths("raw/test")

# Run inference and compute similarity scores
outputs = run_inference(image_paths)
outputs = l2_normalize(outputs)

strange_paths = get_image_paths("raw/strange")

strange_outputs = run_inference(strange_paths)
strange_outputs = l2_normalize(strange_outputs)


def euclidean_distance(tensor1, tensor2):
    return torch.cdist(tensor1.unsqueeze(0), tensor2.unsqueeze(0), p=2).item()


# Assuming `outputs` and `strange_outputs` are tensors already on GPU
# Loop through each embedding in `strange_outputs`
for i, strange_embedding in enumerate(strange_outputs):
    print(f"Distances for strange_output[{strange_paths[i]}]:")

    # Calculate distances between this strange_embedding and all output embeddings
    distances = []
    min_ = 1000000
    name = ""
    for j, output_embedding in enumerate(outputs):
        distance = euclidean_distance(strange_embedding, output_embedding)
        distances.append(distance)
        if distance < min_:
            min_ = distance
            name = image_paths[j]
        # print(f"  Distance to outputs[{image_paths[j]}]: {distance}")
    print(f"name :{name} min : {min_}")
    print("\n")  # Optional: Separate output for better readability
