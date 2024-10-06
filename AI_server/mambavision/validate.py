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
    torch.load(r"checkpoints/fine_tuned_mamba_vision_L2_e15.pth")
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


# image_paths = get_image_paths("raw/test")
# outputs = run_inference(image_paths)
# outputs = l2_normalize(outputs)

# image_embedding_dict = {}
# for idx, output in enumerate(outputs):
#     image_embedding_dict[image_paths[idx]] = output


def euclidean_distance(tensor1, tensor2):
    return torch.cdist(tensor1.unsqueeze(0), tensor2.unsqueeze(0), p=2).item()


def extract_class(img_name):
    return img_name.split("_")[-1].split(".")[0]


from collections import defaultdict


def calculate_voting_accuracy(image_embedding_dict):
    """
    Calculate accuracy based on class voting mechanism.
    A class is considered correctly classified if more than 50% of its images
    are correctly matched to the same class.
    """
    # Create class to images mapping
    class_to_images = defaultdict(list)
    for img_path in image_embedding_dict.keys():
        img_class = extract_class(img_path)
        class_to_images[img_class].append(img_path)

    # Convert embeddings to tensor for batch computation
    embeddings_tensor = torch.stack(list(image_embedding_dict.values()))
    pairwise_distances = torch.cdist(embeddings_tensor, embeddings_tensor, p=2)

    # Store image paths in a list to map indices to paths
    image_paths_list = list(image_embedding_dict.keys())

    # Calculate accuracy for each class
    correct_classes = 0
    total_classes = len(class_to_images)

    print(f"\nProcessing {total_classes} classes...")

    # Process each class
    for class_name, class_images in tqdm(
        class_to_images.items(), desc="Processing Classes"
    ):
        correct_matches_in_class = 0
        total_images_in_class = len(class_images)
        similarity_scores = []
        # Process each image in the class
        for img_path in class_images:
            # Get the index of the current image
            img_idx = image_paths_list.index(img_path)

            # Get distances for the current image, excluding self
            distances = pairwise_distances[img_idx]
            distances[img_idx] = float("inf")  # Exclude self

            # Find the closest image
            closest_idx = torch.argmin(distances).item()
            closest_image = image_paths_list[closest_idx]
            min_distance = distances[closest_idx].item()

            # Check if the closest image belongs to the same class
            if extract_class(closest_image) == class_name:
                correct_matches_in_class += 1
                similarity_scores.append(min_distance)

        # Calculate accuracy for this class
        class_accuracy = correct_matches_in_class / total_images_in_class
        avg_similarity = (
            sum(similarity_scores) / len(similarity_scores) if similarity_scores else 0
        )
        # If more than 50% of images in this class are correctly matched,
        # count this class as correctly classified
        if class_accuracy > 0.5 and avg_similarity < 0.3:
            correct_classes += 1

    # Calculate overall accuracy
    final_accuracy = (correct_classes / total_classes) * 100

    print(f"\nFinal Results:")
    print(f"Total classes: {total_classes}")
    print(f"Correctly classified classes: {correct_classes}")
    print(f"Overall accuracy: {final_accuracy:.2f}%")

    return final_accuracy


def calculate_top_1(image_embedding_dict, total_images, pairwise_distances):
    correct_matches = 0
    for i, (img_name_1, embedding_1) in tqdm(
        enumerate(image_embedding_dict.items()), desc="Processing Images", unit="image"
    ):
        img_class_1 = extract_class(img_name_1)  # Extract class of img_name_1

        # Get the distances for the current image, excluding self distance
        distances = pairwise_distances[i]
        distances[i] = float("inf")  # Exclude distance to itself

        # Find the index of the closest image
        closest_index = torch.argmin(distances).item()
        closest_image = list(image_embedding_dict.keys())[closest_index]

        # Extract class of the closest image
        img_class_2 = extract_class(closest_image)

        # Check if the closest image belongs to the same class
        if img_class_1 == img_class_2:
            correct_matches += 1

    # Calculate the accuracy
    accuracy = correct_matches / total_images
    accuracy *= 100
    print(f"Top 1: {accuracy:.4f}%")


# Usage in your existing code:
def main():
    # Your existing code for loading model and preprocessing...

    image_paths = get_image_paths("raw/test")
    outputs = run_inference(image_paths)
    outputs = l2_normalize(outputs)

    # Create image embedding dictionary
    image_embedding_dict = {}
    for idx, output in enumerate(outputs):
        image_embedding_dict[image_paths[idx]] = output

    total_images = len(image_embedding_dict)
    embeddings_tensor = torch.stack(list(image_embedding_dict.values()))
    pairwise_distances = torch.cdist(embeddings_tensor, embeddings_tensor, p=2)
    calculate_top_1(image_embedding_dict, total_images, pairwise_distances)
    # Compute the closest image and check class match

    # Calculate voting-based accuracy
    accuracy = calculate_voting_accuracy(image_embedding_dict)


if __name__ == "__main__":
    main()
