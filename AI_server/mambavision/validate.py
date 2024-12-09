import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
from models import CustomHead, mamba_vision_T,mamba_vision_L2, mamba_vision_B, mamba_vision_T2
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import os
import matplotlib.pyplot as plt
from models import add_lora_to_model
import numpy as np
# Load the model
import torch
import timm


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = mamba_vision_T(pretrained=True)
model =  mamba_vision_T(pretrained=False).to(device)
add_lora_to_model(model=model, rank=4, alpha=1, target_modules=["head", "levels.3.blocks.3.mlp.fc2", "levels.3.blocks.3.mlp.fc1", "levels.3.blocks.3.mixer.proj", "levels.3.blocks.3.mixer.qkv", "levels.3.blocks.2.mlp.fc2", "levels.3.blocks.2.mlp.fc1", "levels.3.blocks.2.mixer.proj", "levels.3.blocks.2.mixer.qkv","levels.3.blocks.1.mlp.fc2", "levels.3.blocks.1.mlp.fc1", "levels.3.blocks.1.mixer.in_proj", "levels.3.blocks.1.mixer.x_proj", "levels.3.blocks.1.mixer.dt_proj", "levels.3.blocks.1.mixer.out_proj", "levels.3.blocks.0.mlp.fc2", "levels.3.blocks.0.mlp.fc1", "levels.3.blocks.0.mixer.in_proj", "levels.3.blocks.0.mixer.x_proj", "levels.3.blocks.0.mixer.dt_proj", "levels.3.blocks.0.mixer.out_proj", "levels.2.blocks.7.mlp.fc2", "levels.2.blocks.7.mlp.fc1", "levels.2.blocks.7.mixer.proj", "levels.2.blocks.7.mixer.qkv"])
checkpoint = torch.load(r"checkpoints/checkpoint_epoch_73.pth")
# model.load_state_dict(torch.load(r"checkpoints/checkpoint_epoch_65.pth"))
# model.head = CustomHead(in_channels=640)
# checkpoint = torch.load(r"checkpoints/checkpoint_epoch_21.pth")
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)
model.eval()  # Set the model to evaluation mode
# Image preprocessing
preprocess = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ]
)
# Define Dataset
class ImageDataset(Dataset):
    def __init__(self, image_paths, transform):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB")
        image_tensor = self.transform(image)
        return image_tensor

def run_inference(image_paths, batch_size=64):
    dataset = ImageDataset(image_paths, preprocess)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    all_outputs = []
    progress_bar = tqdm(total=len(dataloader), desc="Processing Batches", unit="batch")

    with torch.no_grad():
        for input_tensor in dataloader:
            input_tensor = input_tensor.to(device)
            batch_outputs = model.forward(input_tensor)
            all_outputs.append(batch_outputs)
            progress_bar.update(1)
    progress_bar.close()

    return torch.cat(all_outputs, dim=0)

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

# Helper functions for metrics
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

def run_inference_and_compute_metrics(image_paths, strange_paths, 
                                      get_max_first_value_elements, get_min_value_per_count_ratio, 
                                      calculate_occur_ratio, metric_threshold, unregister = False):
    # Run inference on both image and strange paths
    outputs = run_inference(image_paths)
    outputs = l2_normalize(outputs)

    strange_outputs = run_inference(strange_paths)
    strange_outputs = l2_normalize(strange_outputs)

    # Prepare class labels
    image_classes = [extract_class(p) for p in image_paths]
    strange_classes = [extract_class(p) for p in strange_paths]

    # # Compute cosine similarities and distances
    cosine_similarities = torch.matmul(strange_outputs, outputs.T)  # [num_strange_images, num_images]
    distances = 1 - cosine_similarities  # [num_strange_images, num_images]
    # Compute Euclidean distances
    # distances = torch.cdist(strange_outputs, outputs, p=2.0)  # [num_strange_images, num_images]

    # Get min distances and indices
    min_distance_values, min_distance_indices = distances.min(dim=1)  # [num_strange_images]

    # Get min distance classes
    min_dis_classes = [image_classes[idx] for idx in min_distance_indices.cpu().numpy()]

    # Compute score
    score = sum(1 for i in range(len(strange_classes)) if strange_classes[i] == min_dis_classes[i])

    # Build count_dict
    count_dict = {}
    for i in range(len(strange_classes)):
        strange_class = strange_classes[i]
        min_dis_class = min_dis_classes[i]
        min_dis_value = min_distance_values[i].item()

        if strange_class not in count_dict:
            count_dict[strange_class] = {}

        if min_dis_class not in count_dict[strange_class]:
            count_dict[strange_class][min_dis_class] = (1, min_dis_value)
        else:
            count, dis_value = count_dict[strange_class][min_dis_class]
            count_dict[strange_class][min_dis_class] = (count + 1, dis_value + min_dis_value)

    # Initialize variables for metrics
    voting_score = 0
    total_avg = 0
    total_metric = 0
    total_occurence = 0
    avg_values = []
    occurence_values = []
    metric_values = []
    acc = 0
    identity_acc = 0
    frr = 0
    far = 0

    # Iterate over classes in the count dictionary and compute metrics
    for key, value in count_dict.items():
        # Get max occurrence and min distance
        max_occur = get_max_first_value_elements(value)
        min_distance = get_min_value_per_count_ratio(max_occur)

        # Compute average and other metrics
        average = min_distance[1][1] / min_distance[1][0]
        total_avg += average
        avg_values.append(average)
        
        occur_ratio = calculate_occur_ratio(min_distance, value)
        total_occurence += occur_ratio
        occurence_values.append(occur_ratio)

        metric = (((1 - average) + occur_ratio) / 2)
        # metric = (1 - average)
        # metric = (((1 - average) + occur_ratio*0.5) / 1.5)
        # metric = (occur_ratio)
        total_metric += metric
        metric_values.append(metric)

        # Voting score
        if key == min_distance[0]:
            voting_score += 1

        # Accuracy and FRR calculation
        if not unregister:
            if metric > metric_threshold:
                acc += 1
                if key == min_distance[0]:
                    identity_acc += 1
            else:
                frr += 1
        else:
            if metric < metric_threshold:
                acc += 1
            else:
                far += 1

    # Number of classes
    n = len(count_dict)

    # Calculate means
    avg_mean = total_avg / n
    occurence_mean = total_occurence / n
    metric_mean = total_metric / n

    # Calculate sample variances
    avg_variance = sum((x - avg_mean) ** 2 for x in avg_values) / (n - 1)
    occurence_variance = sum((x - occurence_mean) ** 2 for x in occurence_values) / (n - 1)
    metric_variance = sum((x - metric_mean) ** 2 for x in metric_values) / (n - 1)

    # Prepare results dictionary
    results = {
        'voting_score': voting_score / n,
        'acc': acc * 100 / n,
        'identity_acc': identity_acc * 100 / acc,
        'frr': frr * 100 / n,
        'far': far * 100 / n,
        'avg_mean': avg_mean,
        'occurence_mean': occurence_mean,
        'metric_mean': metric_mean,
        'avg_variance': avg_variance,
        'occurence_variance': occurence_variance,
        'metric_variance': metric_variance,
        'top_1': score / len(strange_classes),
        'top_voting': voting_score / n,
        'metric_values' : metric_values
    }

    # Return the result
    return results

def print_results_register(results):
    # print("Results Summary:")
    print(f"Voting Score: {results['voting_score']}")
    print(f"Top 1 Accuracy: {results['top_1']}")
    print(f"ACC: {results['acc']}%")
    print(f"Identity ACC: {results['identity_acc']}%")
    print(f"FRR: {results['frr']}%")
def print_results_unregister(results):
    # print("Results Summary:")
    # print(f"Voting Score: {results['voting_score']}")
    # print(f"Top 1 Accuracy: {results['top_1']}")
    print(f"ACC: {results['acc']}%")
    print(f"FAR: {results['far']}%")



metric_threshold = 0.698555148144563
metric_threshold = 0.7396525979042053


#TRAIN SET
image_paths = get_image_paths(r"raw/Register-register-half")
strange_paths = get_image_paths(r"raw/Query-register-half")
#TEST SET
# image_paths = get_image_paths(r"raw/dataset-test/Sapienza University Mobile Palmprint Database(SMPD)/Register-register-half")
# strange_paths = get_image_paths(r"raw/dataset-test/Sapienza University Mobile Palmprint Database(SMPD)/Query-register-half")
#NON BG CUT
# image_paths = get_image_paths(r"raw/realistic-test/non-bg-cut/Register-register-half")
# strange_paths = get_image_paths(r"raw/realistic-test/non-bg-cut/Query-register-half")
#NON BG CUT - ROTATE, SHIFT
# image_paths = get_image_paths(r"raw/realistic-test/non-bg-cut/Rotate-Shilf/Register-register-half")
# strange_paths = get_image_paths(r"raw/realistic-test/non-bg-cut/Rotate-Shilf/Query-register-half")
#BG CUT 
# image_paths = get_image_paths(r"raw/realistic-test/bg-cut/Register-register-half")
# strange_paths = get_image_paths(r"raw/realistic-test/bg-cut/Query-register-half")
#BG CUT - ROTATE, SHIFT
# image_paths = get_image_paths(r"raw/realistic-test/bg-cut/Rotate-Shilf/Register-register-half")
# strange_paths = get_image_paths(r"raw/realistic-test/bg-cut/Rotate-Shilf/Query-register-half")

results_registered = run_inference_and_compute_metrics(
    image_paths=image_paths, 
    strange_paths=strange_paths, 
    get_max_first_value_elements=get_max_first_value_elements, 
    get_min_value_per_count_ratio=get_min_value_per_count_ratio, 
    calculate_occur_ratio=calculate_occur_ratio,
    metric_threshold=metric_threshold
)


#TRAIN SET
image_paths = get_image_paths(r"raw/Register-unregister-half")
strange_paths = get_image_paths(r"raw/Query-unregister-half")
#TEST SET
# image_paths = get_image_paths(r"raw/dataset-test/Sapienza University Mobile Palmprint Database(SMPD)/Register-unregister-half")
# strange_paths = get_image_paths(r"raw/dataset-test/Sapienza University Mobile Palmprint Database(SMPD)/Query-unregister-half")
# #NON BG CUT
# image_paths = get_image_paths(r"raw/realistic-test/non-bg-cut/Register-unregister-half")
# strange_paths = get_image_paths(r"raw/realistic-test/non-bg-cut/Query-unregister-half")
#NON BG CUT - ROTATE, SHIFT
# image_paths = get_image_paths(r"raw/realistic-test/non-bg-cut/Rotate-Shilf/Register-unregister-half")
# strange_paths = get_image_paths(r"raw/realistic-test/non-bg-cut/Rotate-Shilf/Query-unregister-half")
#BG CUT
# image_paths = get_image_paths(r"raw/realistic-test/bg-cut/Register-unregister-half")
# strange_paths = get_image_paths(r"raw/realistic-test/bg-cut/Query-unregister-half")
#BG CUT - ROTATE, SHIFT
# image_paths = get_image_paths(r"raw/realistic-test/bg-cut/Rotate-Shilf/Register-unregister-half")
# strange_paths = get_image_paths(r"raw/realistic-test/bg-cut/Rotate-Shilf/Query-unregister-half")

results_unregistered = run_inference_and_compute_metrics(
    image_paths=image_paths, 
    strange_paths=strange_paths, 
    get_max_first_value_elements=get_max_first_value_elements, 
    get_min_value_per_count_ratio=get_min_value_per_count_ratio, 
    calculate_occur_ratio=calculate_occur_ratio,
    metric_threshold=metric_threshold,
    unregister=True
)


print_results_register(results_registered)
print_results_unregister(results_unregistered)

err = (results_unregistered['far'] + results_registered['frr'])/2.0
print(f"ERR: {err}")
import numpy as np
import matplotlib.pyplot as plt
def freedman_diaconis_bins(data):
    data = np.asarray(data)
    q75, q25 = np.percentile(data, [75 ,25])
    iqr = q75 - q25
    bin_width = 2 * iqr / len(data)**(1/3)
    if bin_width == 0:
        return 1  # Avoid division by zero
    bins = int(np.ceil((data.max() - data.min()) / bin_width))
    return max(bins, 1)  # Ensure at least one bin

# bins_avg = freedman_diaconis_bins(avg_values)
# bins_occur = freedman_diaconis_bins(occurence_values)

# First plot for registered users
bins_metric = freedman_diaconis_bins(results_registered['metric_values'])

plt.figure(figsize=(8, 6))  # Create a new figure
plt.hist(results_registered['metric_values'], bins=30, color='green', edgecolor='black')
plt.title('Distribution of score values for registered users')
plt.xlabel('Score')
plt.ylabel('Frequency')

plt.tight_layout()  # Ensure the plot fits within the figure area
plt.savefig('registered.png')  # Save the figure to a file
plt.close()  # Close the figure

# Second plot for unregistered users
bins_metric = freedman_diaconis_bins(results_unregistered['metric_values'])

plt.figure(figsize=(8, 6))  # Create a new figure
plt.hist(results_unregistered['metric_values'], bins=30, color='salmon', edgecolor='black')
plt.title('Distribution of score values for unregistered users')
plt.xlabel('Score')
plt.ylabel('Frequency')

plt.tight_layout()  # Ensure the plot fits within the figure area
plt.savefig('unregistered.png')  # Save the figure to a file
plt.close()  # Close the figure

# # Create box plots
# plt.figure(figsize=(12, 5))

# plt.subplot(1, 3, 1)
# plt.boxplot(avg_values)
# plt.title('Box Plot of Average Values')
# plt.ylabel('Average Distance')

# plt.subplot(1, 3, 2)
# plt.boxplot(occurence_values)
# plt.title('Box Plot of Occurrence Ratios')
# plt.ylabel('Occurrence Ratio')

# plt.subplot(1, 3, 3)
# plt.boxplot(results_registered['metric_values'])
# plt.title('Box Plot of Metric Values')
# plt.ylabel('Metric')

# plt.tight_layout()
# plt.savefig('boxplots.png')  # Save the figure to a file
# plt.close()

import numpy as np
import matplotlib.pyplot as plt
from skimage.filters import threshold_otsu
from sklearn.metrics import roc_curve, auc

def otsu_threshold(registered_set, unregistered_set):
    # Combine both sets of data
    all_data = np.concatenate([registered_set, unregistered_set])
    
    # Apply Otsu's method to find the optimal threshold
    otsu_threshold_value = threshold_otsu(all_data)
    
    # Classify based on the threshold
    labels = np.concatenate([np.ones(len(registered_set)), np.zeros(len(unregistered_set))])
    y_pred = (all_data >= otsu_threshold_value).astype(int)
    
    # Calculate metrics
    ACC, FRR, FAR, ERR = calculate_metrics(labels, y_pred)
    
    print(f"Otsu's Threshold: {otsu_threshold_value}")
    print(f"ACC: {ACC}")
    print(f"FRR: {FRR}")
    print(f"FAR: {FAR}")
    print(f"ERR: {ERR}")
    
    return otsu_threshold_value, ACC, FRR, FAR, ERR

def calculate_metrics(y_true, y_pred):
    TP = np.sum((y_true == 1) & (y_pred == 1))  # True Positive
    TN = np.sum((y_true == 0) & (y_pred == 0))  # True Negative
    FP = np.sum((y_true == 0) & (y_pred == 1))  # False Positive
    FN = np.sum((y_true == 1) & (y_pred == 0))  # False Negative
    
    ACC = (TP + TN) / len(y_true)
    FRR = FN / (FN + TP)  # False Rejection Rate
    FAR = FP / (FP + TN)  # False Acceptance Rate
    ERR = (FRR + FAR) / 2  # Error Rate
    
    return ACC, FRR, FAR, ERR

# Example usage:
registered_set = results_registered['metric_values']
unregistered_set = results_unregistered['metric_values']  # Simulated unregistered data

# Apply Otsu's method to find the optimal threshold
otsu_threshold_value, acc, frr, far, err = otsu_threshold(registered_set, unregistered_set)

# Plot Histogram of both distributions and threshold
plt.hist(registered_set, bins=30, alpha=0.5, label='Registered', color='blue')
plt.hist(unregistered_set, bins=30, alpha=0.5, label='Unregistered', color='red')
plt.axvline(otsu_threshold_value, color='black', linestyle='dashed', label=f'Optimal Threshold = {otsu_threshold_value}')
plt.legend()
plt.xlabel('Metric Value')
plt.ylabel('Frequency')
plt.title('Distribution of Registered and Unregistered Metrics')
plt.savefig("ostu.png")


import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

def calculate_metrics(y_true, y_pred):
    TP = np.sum((y_true == 1) & (y_pred == 1))  # True Positive
    TN = np.sum((y_true == 0) & (y_pred == 0))  # True Negative
    FP = np.sum((y_true == 0) & (y_pred == 1))  # False Positive
    FN = np.sum((y_true == 1) & (y_pred == 0))  # False Negative
    
    ACC = (TP + TN) / len(y_true)
    FRR = FN / (FN + TP)  # False Rejection Rate
    FAR = FP / (FP + TN)  # False Acceptance Rate
    ERR = (FRR + FAR) / 2  # Error Rate
    
    return ACC, FRR, FAR, ERR

def roc_youden_index(registered_set, unregistered_set):
    # Combine both sets of data
    all_data = np.concatenate([registered_set, unregistered_set])
    labels = np.concatenate([np.ones(len(registered_set)), np.zeros(len(unregistered_set))])

    # Compute ROC curve and AUC
    fpr, tpr, thresholds = roc_curve(labels, all_data)
    roc_auc = auc(fpr, tpr)

    # Youden's Index: TPR - FPR
    youden_index = tpr - fpr
    optimal_threshold_idx = np.argmax(youden_index)
    optimal_threshold = thresholds[optimal_threshold_idx]

    # Calculate metrics
    y_pred = (all_data >= optimal_threshold).astype(int)
    ACC, FRR, FAR, ERR = calculate_metrics(labels, y_pred)

    # Return all necessary values for further use (including the ROC values)
    return optimal_threshold, ACC, FRR, FAR, ERR, fpr, tpr, thresholds, roc_auc, optimal_threshold_idx

# Example usage:
registered_set = results_registered['metric_values']
unregistered_set = results_unregistered['metric_values']  # Simulated unregistered data

# Apply ROC + Youden's Index method
optimal_threshold, acc, frr, far, err, fpr, tpr, thresholds, roc_auc, optimal_threshold_idx = roc_youden_index(registered_set, unregistered_set)

# Plot ROC curve
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.scatter(fpr[optimal_threshold_idx], tpr[optimal_threshold_idx], color='red', marker='o', label="Optimal Threshold")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.savefig("roc_youden_index.png")

# Output the optimal threshold and the metrics
print(f"Optimal Threshold (Youden's Index): {optimal_threshold}")
print(f"ACC: {acc}")
print(f"FRR: {frr}")
print(f"FAR: {far}")
print(f"ERR: {err}")