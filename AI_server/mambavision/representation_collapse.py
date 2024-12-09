import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
from models import CustomHead, mamba_vision_T,mamba_vision_L2, mamba_vision_B, mamba_vision_T2
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import os
from models import add_lora_to_model
import numpy as np
import matplotlib.pyplot as plt
import torch
import seaborn as sns

from scipy.stats import entropy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = mamba_vision_T(pretrained=True)
model =  mamba_vision_T(pretrained=False).to(device)
add_lora_to_model(model=model, rank=4, alpha=1, target_modules=["head", "levels.3.blocks.3.mlp.fc2", "levels.3.blocks.3.mlp.fc1", "levels.3.blocks.3.mixer.proj", "levels.3.blocks.3.mixer.qkv", "levels.3.blocks.2.mlp.fc2", "levels.3.blocks.2.mlp.fc1", "levels.3.blocks.2.mixer.proj", "levels.3.blocks.2.mixer.qkv","levels.3.blocks.1.mlp.fc2", "levels.3.blocks.1.mlp.fc1", "levels.3.blocks.1.mixer.in_proj", "levels.3.blocks.1.mixer.x_proj", "levels.3.blocks.1.mixer.dt_proj", "levels.3.blocks.1.mixer.out_proj", "levels.3.blocks.0.mlp.fc2", "levels.3.blocks.0.mlp.fc1", "levels.3.blocks.0.mixer.in_proj", "levels.3.blocks.0.mixer.x_proj", "levels.3.blocks.0.mixer.dt_proj", "levels.3.blocks.0.mixer.out_proj" ])
model.load_state_dict(torch.load(r"checkpoints/fine_tuned_mamba_vision_T_140_55.pth"))
# model.head = CustomHead(in_channels=640)
# checkpoint = torch.load(r"checkpoints/checkpoint_epoch_21.pth")
# model.load_state_dict(checkpoint['model_state_dict'])
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

def create_collapse_visualization(collapse_metrics):
    try:
        plt.figure(figsize=(15, 5))
        
        # Sample data to prevent memory issues
        def safe_sample(data, max_points=10000):
            if len(data) > max_points:
                return np.random.choice(data, max_points, replace=False)
            return data
        
        # Variance plot
        plt.subplot(131)
        variance_data = safe_sample(collapse_metrics['variance']['variance_distribution'])
        plt.hist(variance_data, bins=50, edgecolor='black')
        plt.title('Representation Variance')
        plt.xlabel('Variance')
        plt.ylabel('Frequency')
        
        # L2 Norm plot
        plt.subplot(132)
        l2_norm_data = safe_sample(collapse_metrics['l2_norm']['l2_norm_distribution'])
        plt.hist(l2_norm_data, bins=50, edgecolor='black')
        plt.title('L2 Norm Distribution')
        plt.xlabel('L2 Norm')
        plt.ylabel('Frequency')
        
        # Cosine Similarity plot
        plt.subplot(133)
        cosine_sim_data = safe_sample(collapse_metrics['cosine_similarity']['similarity_distribution'].flatten())
        plt.hist(cosine_sim_data, bins=50, edgecolor='black')
        plt.title('Cosine Similarity')
        plt.xlabel('Cosine Similarity')
        plt.ylabel('Frequency')
        
        plt.tight_layout()
        plt.savefig('representation_collapse_analysis.png', dpi=300)
        plt.close()
        print("Visualization saved successfully")
    
    except Exception as e:
        print(f"Visualization error: {e}")

def detect_representation_collapse(
    model, 
    image_paths, 
    batch_size=256, 
    device='cuda', 
    metrics=['variance', 'entropy', 'l2_norm', 'cosine_similarity']
):
    # Ensure model is in evaluation mode
    model.eval()
    
    # Create dataset and dataloader
    dataset = ImageDataset(image_paths, preprocess)
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=4, 
        pin_memory=True
    )
    progress_bar = tqdm(total=len(dataloader), desc="Processing Batches", unit="batch")
    # Collect all model outputs
    all_outputs = []
    with torch.no_grad():
        for input_tensor in dataloader:
            input_tensor = input_tensor.to(device)
            batch_outputs = model.forward_features(input_tensor)
            all_outputs.append(batch_outputs)
            progress_bar.update(1)
    progress_bar.close()
    # Concatenate all outputs
    representations = torch.cat(all_outputs, dim=0)
    # Collapse detection metrics
    collapse_metrics = {}
    
    # 1. Representation Variance
    if 'variance' in metrics:
        rep_variance = torch.var(representations, dim=0)
        collapse_metrics['variance'] = {
            'mean_variance': rep_variance.mean().item(),
            'variance_distribution': rep_variance.cpu().numpy()
        }
    # 2. Entropy of Representations
    if 'entropy' in metrics:
        rep_entropy = torch.distributions.Categorical(
            probs=torch.softmax(representations, dim=1)
        ).entropy()
        collapse_metrics['entropy'] = {
            'mean_entropy': rep_entropy.mean().item(),
            'entropy_distribution': rep_entropy.cpu().numpy()
        }
    
    # 3. L2 Norm of Representations
    if 'l2_norm' in metrics:
        rep_l2_norm = torch.norm(representations, dim=1)
        collapse_metrics['l2_norm'] = {
            'mean_l2_norm': rep_l2_norm.mean().item(),
            'l2_norm_std': rep_l2_norm.std().item(),
            'l2_norm_distribution': rep_l2_norm.cpu().numpy()
        }
    
    # 4. Cosine Similarity Between Representations
    if 'cosine_similarity' in metrics:
        # Compute pairwise cosine similarity
        normalized_reps = torch.nn.functional.normalize(representations, p=2, dim=1)
        cosine_sim_matrix = torch.mm(normalized_reps, normalized_reps.t())
        
        # Remove diagonal (self-similarity)
        cosine_sim_matrix.fill_diagonal_(0)
        
        collapse_metrics['cosine_similarity'] = {
            'mean_similarity': cosine_sim_matrix.mean().item(),
            'max_similarity': cosine_sim_matrix.max().item(),
            'similarity_distribution': cosine_sim_matrix.cpu().numpy()
        }
    
    # 5. Dimensionality Analysis
    collapse_metrics['dimensionality'] = {
        'representation_shape': representations.shape,
        'rank_estimate': torch.linalg.matrix_rank(representations).item()
    }
    
    # Visualization (optional but helpful)
    
    # Generate visualization
    create_collapse_visualization(collapse_metrics)
    
    return collapse_metrics

# Example usage
def is_representation_collapsed(metrics, thresholds=None):
    """
    Determine if representations are collapsed based on predefined thresholds.
    
    Args:
    - metrics: Collapse metrics dictionary
    - thresholds: Custom thresholds for collapse detection
    
    Returns:
    - Boolean indicating potential representation collapse
    """
    default_thresholds = {
        'variance_threshold': 0.01,  # Low variance indicates collapse
        'entropy_threshold': 0.1,    # Low entropy suggests limited diversity
        'l2_norm_var_threshold': 0.05,  # Low variation in L2 norms
        'cosine_sim_threshold': 0.9  # High average cosine similarity
    }
    
    thresholds = thresholds or default_thresholds
    
    # Collapse indicators
    variance_collapsed = (
        metrics['variance']['mean_variance'] < thresholds['variance_threshold']
    )
    
    entropy_collapsed = (
        metrics['entropy']['mean_entropy'] < thresholds['entropy_threshold']
    )
    
    l2_norm_collapsed = (
        metrics['l2_norm']['l2_norm_std'] < thresholds['l2_norm_var_threshold']
    )
    
    cosine_sim_collapsed = (
        metrics['cosine_similarity']['mean_similarity'] > thresholds['cosine_sim_threshold']
    )
    
    # Combine collapse indicators
    is_collapsed = (
        variance_collapsed or 
        entropy_collapsed or 
        l2_norm_collapsed or 
        cosine_sim_collapsed
    )
    
    return is_collapsed

# Comprehensive usage example
def analyze_model_representations(model, image_paths):
    # Detect representation metrics
    collapse_metrics = detect_representation_collapse(model, image_paths)
    
    # Check for collapse
    is_collapsed = is_representation_collapsed(collapse_metrics)
    
    if is_collapsed:
        print("Warning: Potential Representation Collapse Detected!")
    else:
        print("None Representation Collapse Detected")
        # Additional logging or intervention strategies
    
    return collapse_metrics, is_collapsed


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

image_paths = get_image_paths(r"raw/Register-unregister-half")
analyze_model_representations(model,image_paths)