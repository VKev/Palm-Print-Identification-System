import torch
from PIL import Image
from torchvision import transforms
from models import mamba_vision_L2
from scipy.spatial.distance import cosine
from tqdm import tqdm
from collections import defaultdict


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = mamba_vision_L2()
model.load_state_dict(torch.load(r"checkpoints/fine_tuned_mamba_vision_L2_f2_2.pth"))
model.to(device)
model.eval()


def preprocess_images(image_paths):
    preprocess = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
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


# Example usage
image_paths = get_image_paths("raw/test")

# Run inference and compute similarity scores
outputs = run_inference(image_paths, 50)


def cosine_similarity_list(outputs):
    """Calculate a list of similarities for each image."""
    # Normalize the output tensors
    outputs = outputs / outputs.norm(dim=1, keepdim=True)  # Normalize to unit vectors
    similarities = []

    # Use tqdm for a progress bar over the range of outputs
    for i in tqdm(range(outputs.size(0)), desc="Calculating similarities"):
        # Compute similarity of the i-th image with all others
        similarity_scores = (
            torch.mm(outputs[i].unsqueeze(0), outputs.t()).squeeze().tolist()
        )
        similarities.append(similarity_scores)

    return similarities


similarity_list = cosine_similarity_list(outputs)

image_similarity_dict = {}

for idx, (image_path, similarities) in enumerate(zip(image_paths, similarity_list)):
    similarity_with_images = {
        image_paths[j]: similarity
        for j, similarity in enumerate(similarities)
        if j != idx
    }
    image_similarity_dict[image_path] = similarity_with_images


def get_group_number(image_path):
    # Assuming the group number is the last number before the file extension
    return image_path.split("_")[-1].split(".")[0]


group_image_count = defaultdict(int)
for image_path in image_paths:
    group_number = get_group_number(image_path)
    group_image_count[group_number] += 1

correct_match_count = 0

for image_path, similarity_mapping in image_similarity_dict.items():
    group_number = get_group_number(image_path)

    most_similar_image = max(similarity_mapping, key=similarity_mapping.get)

    most_similar_group = get_group_number(most_similar_image)

    if group_number == most_similar_group:
        correct_match_count += 1

max_group_size = len(image_paths)
score = correct_match_count / max_group_size

print(f"Perfomance score: {score}")
