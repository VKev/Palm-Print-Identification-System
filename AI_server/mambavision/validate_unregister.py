import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
from models import CustomHead
from models import mamba_vision_T
from scipy.spatial.distance import cosine
from torch import nn, optim
import torch.nn.functional as F
from models import add_lora_to_model
# Load the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = mamba_vision_T(
#     pretrained=True
# )  
model =  mamba_vision_T(pretrained=False).to(device)
add_lora_to_model(model=model, rank=4, alpha=1, target_modules=["head", "levels.3.blocks.3.mlp.fc2", "levels.3.blocks.3.mlp.fc1", "levels.3.blocks.3.mixer.proj", "levels.3.blocks.3.mixer.qkv", "levels.3.blocks.2.mlp.fc2", "levels.3.blocks.2.mlp.fc1", "levels.3.blocks.2.mixer.proj", "levels.3.blocks.2.mixer.qkv","levels.3.blocks.1.mlp.fc2", "levels.3.blocks.1.mlp.fc1", "levels.3.blocks.1.mixer.in_proj", "levels.3.blocks.1.mixer.x_proj", "levels.3.blocks.1.mixer.dt_proj", "levels.3.blocks.1.mixer.out_proj", "levels.3.blocks.0.mlp.fc2", "levels.3.blocks.0.mlp.fc1", "levels.3.blocks.0.mixer.in_proj", "levels.3.blocks.0.mixer.x_proj", "levels.3.blocks.0.mixer.dt_proj", "levels.3.blocks.0.mixer.out_proj" ])
model.load_state_dict(torch.load(r"checkpoints/fine_tuned_mamba_vision_T_latest_12.pth"))
model.to(device)
model.eval()  # Set the model to evaluation mode


# Image preprocessing
def preprocess_images(image_paths):
    preprocess = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4440], std=[0.2104]),
        ]
    )

    images = []
    for image_path in image_paths:
        image = Image.open(image_path).convert("RGB")
        image_tensor = preprocess(image)
        images.append(image_tensor)

    return torch.stack(images).to(device)


def run_inference(image_paths, batch_size=128):
    all_outputs = []

    total_batches = (len(image_paths) + batch_size - 1) // batch_size
    progress_bar = tqdm(total=total_batches, desc="Processing Batches", unit="batch")

    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i : i + batch_size]
        input_tensor = preprocess_images(batch_paths)

        with torch.no_grad():
            batch_outputs = model.forward(input_tensor)

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
# image_paths = get_image_paths(r"raw/realistic-test/bg-cut/Register-half")
# image_paths = get_image_paths(r"raw/realistic-test/bg-cut/Register-unregister-half")

# image_paths = get_image_paths(r"raw/realistic-test/non-bg-cut/Rotate-Shilf/SRegister-half")
image_paths = get_image_paths(r"raw/realistic-test/bg-cut/Rotate-Shilf/SRegister-half")
# image_paths = get_image_paths(r"raw/realistic-test/non-bg-cut/Register-unregister-half")
# image_paths = get_image_paths(r"raw/dataset-test/SRegister-half")
# image_paths = get_image_paths(r"raw/dataset-test/DRegister-half")
# image_paths = get_image_paths(r"raw/dataset-test/register-half")

# Run inference and compute similarity scores
outputs = run_inference(image_paths)
outputs = l2_normalize(outputs)


# strange_paths = get_image_paths(r"raw/realistic-test/bg-cut/Query-half")
# strange_paths = get_image_paths(r"raw/realistic-test/bg-cut/Query-unregister-half")

# strange_paths = get_image_paths(r"raw/realistic-test/non-bg-cut/Rotate-Shilf/SQuery-half")
strange_paths = get_image_paths(r"raw/realistic-test/bg-cut/Rotate-Shilf/SQuery-half")
# strange_paths = get_image_paths(r"raw/realistic-test/non-bg-cut/Query-unregister-half")
# strange_paths = get_image_paths(r"raw/dataset-test/SQuery-half")
# strange_paths = get_image_paths(r"raw/dataset-test/DQuery-half")
# strange_paths = get_image_paths(r"raw/dataset-test/query-half")

strange_outputs = run_inference(strange_paths)
strange_outputs = l2_normalize(strange_outputs)


def euclidean_distance(tensor1, tensor2):
    return torch.cdist(tensor1.unsqueeze(0), tensor2.unsqueeze(0), p=2).item()

def cosine_sim(tensor1, tensor2):
    # Normalize the tensors
    tensor1_normalized = F.normalize(tensor1, p=2, dim=0)
    tensor2_normalized = F.normalize(tensor2, p=2, dim=0)
    
    # Compute cosine similarity
    cos_sim = F.cosine_similarity(tensor1_normalized.unsqueeze(0), 
                                tensor2_normalized.unsqueeze(0), 
                                dim=1)
    
    # Convert similarity to distance (1 - cos_sim)
    # This ensures that:
    # - identical vectors have distance 0 (cos_sim = 1 → distance = 0)
    # - orthogonal vectors have distance 1 (cos_sim = 0 → distance = 1)
    # - opposite vectors have distance 2 (cos_sim = -1 → distance = 2)
    distance = 1 - cos_sim.item()
    
    return distance

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


output_dict = {}
for idx, image_path in enumerate(image_paths):
    output_dict[image_path] = outputs[idx]

strange_output_dict = {}
for idx, image_path in enumerate(strange_paths):
    strange_output_dict[image_path] = strange_outputs[idx]

# print(output_dict)
# print("\n")
# print(strange_output_dict)


def one_vs_all(query_embedding, dict):
    distances = []
    for key, value in dict.items():
        distance = euclidean_distance(query_embedding, value)
        distances.append((key, distance))
    min_distance_element = min(distances, key=lambda x: x[1])
    return min_distance_element


# first_key = next(iter(strange_output_dict))
# result = one_vs_all(strange_output_dict[first_key], output_dict)

score = 0
count_dict = {}
for strange_key, strange_value in strange_output_dict.items():
    result = one_vs_all(strange_output_dict[strange_key], output_dict)
    min_dis_class = extract_class(result[0])
    min_dis_value = result[1]
    strang_key_class = extract_class(strange_key)
    if strang_key_class not in count_dict:
        count_dict[strang_key_class] = {}
    if min_dis_class not in count_dict[strang_key_class]:
        count_dict[strang_key_class][min_dis_class] = (1, min_dis_value)
    else:
        count, dis_value = count_dict[strang_key_class][min_dis_class]
        count_dict[strang_key_class][min_dis_class] = (
            count + 1,
            dis_value + min_dis_value,
        )

    if min_dis_class == strang_key_class:
        score += 1


def get_max_first_value_elements(input_dict):
    max_first_value = max(value[0] for value in input_dict.values())
    result = [
        (key, value) for key, value in input_dict.items() if value[0] == max_first_value
    ]

    return result


def get_min_value_per_count_ratio(input_list):
    ratios = [
        (cls, (count, value), value / count) for cls, (count, value) in input_list
    ]

    min_element = min(ratios, key=lambda x: x[2])

    return min_element[0], min_element[1]


def calculate_occur_ratio(min_distance, value):
    result_count = min_distance[1][0]

    total_count = sum(count for count, _ in value.values())

    occur_ratio = result_count / total_count if total_count > 0 else 0

    return occur_ratio


voting_score = 0
total_avg = 0
total_metric = 0
total_occurence = 0

avg_values = []
occurence_values = []
metric_values = []
for key, value in count_dict.items():
    print(f"class {key}: ")
    print(f"Value {value}")
    max_occur = get_max_first_value_elements(value)
    min_distance = get_min_value_per_count_ratio(max_occur)
    print(f"Result {min_distance}")
    avarage = min_distance[1][1] / min_distance[1][0]
    total_avg += avarage
    avg_values.append(avarage)  # Store individual average
    print(f"Average {avarage}")
    occur_ratio = calculate_occur_ratio(min_distance, value)
    total_occurence += occur_ratio
    occurence_values.append(occur_ratio)  # Store individual occurrence ratio
    print(f"Occurrence Ratio: {occur_ratio}")
    metric = (((1 - avarage) + occur_ratio) / 2)
    total_metric += metric
    metric_values.append(metric)  # Store individual metric
    print(f"test metric: {metric}")
    if key == min_distance[0]:
        voting_score += 1

n = len(count_dict)
print(f"Number of classes: {n}")
print(f"Top 1: {score / (len(strange_output_dict))} ")
print(f"Top voting: {voting_score / n} ")

# Calculate means
avg_mean = total_avg / n
occurence_mean = total_occurence / n
metric_mean = total_metric / n

# Calculate sample variances
avg_variance = sum((x - avg_mean) ** 2 for x in avg_values) / (n - 1)
occurence_variance = sum((x - occurence_mean) ** 2 for x in occurence_values) / (n - 1)
metric_variance = sum((x - metric_mean) ** 2 for x in metric_values) / (n - 1)

print("\nMeans:")
print(f"Total average's mean: {avg_mean}")
print(f"Total occurrence's mean: {occurence_mean}")
print(f"Total test metric's mean: {metric_mean}")

print("\nVariances:")
print(f"Total average's variance: {avg_variance}")
print(f"Total occurrence's variance: {occurence_variance}")
print(f"Total test metric's variance: {metric_variance}")

