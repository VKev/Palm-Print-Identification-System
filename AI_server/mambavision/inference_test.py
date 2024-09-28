import torch
from PIL import Image
from torchvision import transforms
from models import mamba_vision_L2
from scipy.spatial.distance import cosine
# Load the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = mamba_vision_L2(pretrained=True).to(device)
model.eval()  # Set the model to evaluation mode

# Image preprocessing
def preprocess_images(image_paths):
    preprocess = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),  # Convert to tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize
    ])
    
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

# Example usage
image_paths = [
    r"raw/test/Cat1.jpg",
    r"raw/test/Cat2.jpg",
    r"raw/test/Dog1.jpg",
    r"raw/test/Dog2.jpg",
]
outputs = run_inference(image_paths)


def cosine_similarity_list(outputs):
    """Calculate a list of similarities for each image."""
    # Normalize the output tensors
    outputs = outputs / outputs.norm(dim=1, keepdim=True)  # Normalize to unit vectors
    
    similarities = []
    for i in range(outputs.size(0)):
        # Compute similarity of the i-th image with all others
        similarity_scores = torch.mm(outputs[i].unsqueeze(0), outputs.t()).squeeze().tolist()
        similarities.append(similarity_scores)
    
    return similarities

outputs = run_inference(image_paths)  # Get feature vectors for all images
similarity_list = cosine_similarity_list(outputs)  # Get similarity list for each image

# Print out the similarity scores for each image
for i, similarities in enumerate(similarity_list):
    print(f"Similarity scores for output{i + 1}: {similarities}")