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
    torch.load(r"checkpoints/best/fine_tuned_mamba_vision_L2_e15.pth")
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


def run_inference(image_paths, batch_size=64):
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


def extract_class(image_path):
    # Get the filename without path
    filename = os.path.basename(image_path)
    # Remove file extension
    filename_without_ext = os.path.splitext(filename)[0]

    # Handle both naming conventions
    if filename.startswith("cuong"):
        # For format cuong1_555.jpg
        return filename_without_ext.split("_")[-1]
    else:
        # For format 00003_s4_0.JPG
        return filename_without_ext.split("_")[-1]


def calculate_class_matching_score(
    path1_images, path2_images, min_distance_pairs, min_distances
):
    # Create a dictionary to store images by class for path1
    class_images_path1 = {}
    for img_path in path1_images:
        img_class = extract_class(img_path)
        if img_class not in class_images_path1:
            class_images_path1[img_class] = []
        class_images_path1[img_class].append(img_path)

    # Store classes for path2 images
    path2_classes = {img_path: extract_class(img_path) for img_path in path2_images}

    score = 0
    total_classes = len(class_images_path1)

    # For each class in path1
    for class_id, class_images in class_images_path1.items():
        # Dictionary to count occurrences of each path2 class
        path2_class_counts = {}

        # For each image in this class from path1
        class_total_distance = 0
        for img_path in class_images:
            # Get the best matching image from path2
            best_match_path = min_distance_pairs[img_path]
            # Get the class of the best matching image
            best_match_class = path2_classes[best_match_path]

            class_total_distance += min_distances[img_path]
            # Count occurrences of path2 classes
            path2_class_counts[best_match_class] = (
                path2_class_counts.get(best_match_class, 0) + 1
            )

        # Check if any path2 class appears more than 50% of the time
        avg_distance = class_total_distance / len(class_images)

        # Check both conditions:
        # 1. Any path2 class appears more than 50% of the time
        # 2. Average distance is less than 0.3
        total_images_in_class = len(class_images)
        meets_frequency_condition = False
        for count in path2_class_counts.values():
            if count / total_images_in_class > 0.4:
                meets_frequency_condition = True
                break

        # Only increment score if both conditions are met
        if meets_frequency_condition and avg_distance < 0.2:
            score += 1
            print(f"Class {class_id} meets both conditions:")
            print(f"  Average distance: {avg_distance:.4f}")
            print(f"  Class distribution: {path2_class_counts}")

    # Calculate final score
    final_score = 1 - (score / total_classes)

    return final_score


# Create dictionary of minimum distance pairs
min_distance_pairs = {}
min_distances = {}
for i, strange_embedding in enumerate(strange_outputs):
    min_distance = float("inf")
    best_match = None

    for j, output_embedding in enumerate(outputs):
        distance = euclidean_distance(strange_embedding, output_embedding)
        if distance < min_distance:
            min_distance = distance
            best_match = image_paths[j]

    min_distance_pairs[strange_paths[i]] = best_match
    min_distances[strange_paths[i]] = min_distance

# Calculate and print the final score
final_score = calculate_class_matching_score(
    strange_paths, image_paths, min_distance_pairs, min_distances
)
print(f"\nFinal Score: {final_score}")
