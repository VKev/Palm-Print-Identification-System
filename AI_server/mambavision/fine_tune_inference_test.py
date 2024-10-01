import torch
from PIL import Image
from torchvision import transforms
from models import mamba_vision_L2
from scipy.spatial.distance import cosine

# Load the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = mamba_vision_L2()  # Initialize the model without pretrained weights
model.load_state_dict(
    torch.load(r"checkpoints/fine_tuned_mamba_vision_L2_f2_2.pth")
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


# Run inference
def run_inference(image_paths):
    input_tensor = preprocess_images(image_paths)  # Preprocess all images
    with torch.no_grad():  # Disable gradient computation
        outputs = model.forward_features(input_tensor)  # Forward pass
    return outputs


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
image_paths = get_image_paths("test/")

# Run inference and compute similarity scores
outputs = run_inference(image_paths)


def cosine_similarity_list(outputs):
    """Calculate a list of similarities for each image."""
    # Normalize the output tensors
    outputs = outputs / outputs.norm(dim=1, keepdim=True)  # Normalize to unit vectors
    similarities = []
    for i in range(outputs.size(0)):
        # Compute similarity of the i-th image with all others
        similarity_scores = (
            torch.mm(outputs[i].unsqueeze(0), outputs.t()).squeeze().tolist()
        )
        similarities.append(similarity_scores)

    return similarities


similarity_list = cosine_similarity_list(outputs)  # Get similarity list for each image

from collections import Counter, defaultdict

grouped_similarities = defaultdict(list)
for i, similarities in enumerate(similarity_list):
    # Extract the base name (e.g., "Cuong1" from "test/cuong1.jpg")
    base_name = image_paths[i].split("/")[-1]  # Get the filename, e.g., "cuong1.jpg"
    prefix = base_name[:-5]  # Remove the ".jpg" to get "cuong1"
    name_prefix = "".join(
        [char for char in prefix if not char.isdigit()]
    )  # Extract the non-numeric prefix, e.g., "Cuong"

    # Find the top similarity for this image
    sorted_similarities = sorted(
        enumerate(similarities), key=lambda x: x[1], reverse=True
    )

    # Exclude self-similarity by skipping the first element if it's the same image
    sorted_similarities = [(idx, sim) for idx, sim in sorted_similarities if idx != i][
        :1
    ]

    # Add the top similarity for the current image to the corresponding group
    for idx, similarity in sorted_similarities:
        similar_image_name = image_paths[idx].split("/")[-1][
            :-4
        ]  # Get the similar image name without the ".jpg"
        grouped_similarities[name_prefix].append(
            f"{prefix}: {similar_image_name} ({similarity:.4f})"
        )


import re


def find_most_frequent(group_values):
    # Use regex to extract the prefix without trailing digits (e.g., "Huy1" -> "Huy")
    prefix_counter = Counter(
        [
            re.sub(r"\d+$", "", value.split(": ")[1].split(" ")[0])
            for value in group_values
        ]
    )

    # Find the prefix with the highest occurrence
    most_frequent_prefix, max_count = prefix_counter.most_common(1)[0]

    # If there is more than one most frequent prefix with the same count, compare their similarity sums
    candidates = [
        prefix for prefix, count in prefix_counter.items() if count == max_count
    ]
    if len(candidates) > 1:
        # There is a tie in the most frequent prefix, sum the similarities for each candidate
        similarity_sums = defaultdict(float)
        for value in group_values:
            similarity_prefix = re.sub(
                r"\d+$", "", value.split(": ")[1].split(" ")[0]
            )  # Remove trailing digits
            similarity_score = float(value.split("(")[1][:-1])
            if similarity_prefix in candidates:
                similarity_sums[similarity_prefix] += similarity_score

        # Choose the prefix with the highest total similarity score
        most_frequent_prefix = max(similarity_sums, key=similarity_sums.get)

    return most_frequent_prefix


# Now, iterate through grouped similarities and apply the function
for idx, (group, values) in enumerate(grouped_similarities.items(), start=1):
    print(f"{group.capitalize()} contains:")

    # Count occurrences and handle tie-breaking by similarity score
    most_frequent = find_most_frequent(values)

    print(f"Most frequent prefix: {most_frequent}")

    # for value_idx, value in enumerate(values, start=1):
    #     print(f"    {value_idx}_{value}")
