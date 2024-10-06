import torch
from torch import nn, optim
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset, random_split
from PIL import Image
from models import mamba_vision_L2
from torch.nn.functional import normalize
import random
import torch.nn.functional as F
import os
from tqdm import tqdm
import re


# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 1
epoch = 30
lr = 1e-5
test = True
n_negatives = 16

if test:
    n_negatives = 5
    batch_size = 5
    train_size = 40


class SemiHardTripletLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(SemiHardTripletLoss, self).__init__()
        self.margin = margin
        self.eps = 1e-6

    def forward(self, anchors, positives, negatives_list):
        """
        Compute the semi-hard triplet loss with multiple negatives per anchor-positive pair.

        Args:
            anchors: tensor of shape (batch_size, embedding_dim)
            positives: tensor of shape (batch_size, embedding_dim)
            negatives_list: list of tensors, each of shape (batch_size, embedding_dim)

        Returns:
            loss: mean of valid triplet losses across all negatives
        """
        batch_size = anchors.size(0)
        num_negatives = len(negatives_list)
        # Stack negatives along a new dimension for easier broadcasting
        negatives = torch.stack(
            negatives_list, dim=1
        )  # (batch_size, num_negatives, embedding_dim)

        # Compute positive distances and expand them for broadcasting
        pos_dist = torch.norm(
            anchors - positives, p=2, dim=1, keepdim=True
        )  # (batch_size, 1)
        pos_dist = pos_dist.expand(
            batch_size, num_negatives
        )  # (batch_size, num_negatives)

        # Compute negative distances for all negatives
        neg_dist = torch.norm(
            anchors.unsqueeze(1) - negatives, p=2, dim=2
        )  # (batch_size, num_negatives)

        # Semi-hard negative: further than positive but within margin
        semi_hard_mask = (neg_dist > pos_dist) & (neg_dist < pos_dist + self.margin)

        # Hard negative: closer than positive
        hard_mask = neg_dist <= pos_dist

        # Combine masks for valid triplets
        valid_mask = semi_hard_mask | hard_mask

        # Compute triplet loss for valid triplets
        triplet_loss = pos_dist - neg_dist + self.margin  # (batch_size, num_negatives)
        triplet_loss = torch.clamp(triplet_loss, min=0.0)  # Ensure non-negative loss

        # Apply the valid triplet mask
        valid_triplet_loss = triplet_loss[valid_mask]

        # If no valid triplets, return zero loss with gradient
        if valid_triplet_loss.numel() == 0:
            return torch.tensor(0.0, device=anchors.device, requires_grad=True)

        # Return mean of valid triplet losses
        return valid_triplet_loss.mean()


class CombinedTripletDataset(Dataset):
    def __init__(self, dataset1, dataset2):
        self.dataset1 = dataset1
        self.dataset2 = dataset2

    def __len__(self):
        return len(self.dataset1) + len(self.dataset2)

    def __getitem__(self, idx):
        if idx < len(self.dataset1):
            return self.dataset1[idx]  # Get item from the first dataset
        else:
            return self.dataset2[
                idx - len(self.dataset1)
            ]  # Get item from the second dataset


# Define the dataset for triplet loss
class TripletDataset(Dataset):
    def __init__(
        self, image_paths, labels, n_negatives, transform=None, augmentation=None
    ):
        self.image_paths = image_paths
        self.labels = labels
        self.n_negatives = (
            n_negatives  # Number of negative samples per anchor-positive pair
        )
        self.transform = transform
        self.augmentation = augmentation

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Anchor image
        anchor_image = Image.open(self.image_paths[idx]).convert("RGB")
        anchor_label = self.labels[idx]

        # Positive image (same label)
        positive_indices = [
            i
            for i, label in enumerate(self.labels)
            if label == anchor_label and i != idx
        ]
        positive_idx = random.choice(positive_indices)
        positive_image = Image.open(self.image_paths[positive_idx]).convert("RGB")

        # Get multiple negative images (different labels)
        negative_indices = [
            i for i, label in enumerate(self.labels) if label != anchor_label
        ]
        negative_images = []
        selected_negative_indices = random.sample(negative_indices, self.n_negatives)
        for neg_idx in selected_negative_indices:
            negative_image = Image.open(self.image_paths[neg_idx]).convert("RGB")

            # Apply augmentations or transformations
            if self.augmentation and random.random() < 0.5:
                negative_image = self.augmentation(negative_image)
            else:
                negative_image = self.transform(negative_image)

            negative_images.append(negative_image)

        # Apply augmentations to anchor and positive images
        if self.augmentation:
            if random.random() < 0.5:
                anchor_image = self.augmentation(anchor_image)
            else:
                anchor_image = self.transform(anchor_image)

            if random.random() < 0.5:
                positive_image = self.augmentation(positive_image)
            else:
                positive_image = self.transform(positive_image)
        else:
            anchor_image = self.transform(anchor_image)
            positive_image = self.transform(positive_image)

        # Return a tuple of (anchor, positive, [negative1, negative2, ..., negativeN])
        return anchor_image, positive_image, negative_images


# Define augmentations
augmentation = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.RandomRotation(60),  # Random rotation by 30 degrees
        transforms.ColorJitter(
            brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2
        ),  # Color jitter
        transforms.RandomHorizontalFlip(),  # Random horizontal flip
        transforms.RandomResizedCrop(size=224, scale=(0.8, 1.0)),
        transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

# Define preprocessing transformations (resize, normalization)
transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


def load_images_from_folders(base_folder):
    image_paths = []
    labels = []

    for root, _, files in os.walk(base_folder):
        files = [
            file for file in files if file.lower().endswith((".bmp", ".jpg", ".jpeg"))
        ]  # Filter only .bmp files

        # Sort files to maintain consistent ordering if needed
        files.sort()

        for file in files:
            img_path = os.path.join(root, file)
            image_paths.append(img_path)

            # Extract the last number from the filename
            label_match = re.search(r"_(\d+)\.(bmp|jpg|jpeg)$", file, re.IGNORECASE)
            if label_match:
                label = int(
                    label_match.group(1)
                )  # Convert the matched number to an integer
                labels.append(label)

    return image_paths, labels


image_folder = r"raw/train"

image_paths, labels = load_images_from_folders(image_folder)
if test:
    image_paths = image_paths[:train_size]
    labels = labels[:train_size]
# Preprocessing
# Load the dataset
triplet_default_dataset = TripletDataset(
    image_paths, labels, transform=transform, n_negatives=n_negatives
)
triplet_argumentation_dataset = TripletDataset(
    image_paths,
    labels,
    augmentation=augmentation,
    transform=transform,
    n_negatives=n_negatives,
)

# Split the dataset into training and validation sets (e.g., 80% training, 20% validation)
combined_dataset = CombinedTripletDataset(
    triplet_default_dataset, triplet_argumentation_dataset
)
train_size = int(0.9 * len(combined_dataset))
val_size = len(combined_dataset) - train_size
train_dataset, val_dataset = random_split(combined_dataset, [train_size, val_size])

# Load data into dataloaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Load the pre-trained model
model = mamba_vision_L2(pretrained=True).to(device)
# Freeze all layers except the last few
for param in model.parameters():
    param.requires_grad = False

last_mamba_layer = model.levels[-1]

for param in last_mamba_layer.parameters():
    param.requires_grad = True

# Loss function and optimizer
triplet_loss = nn.TripletMarginLoss(margin=1.0, p=2)
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

# Ensure checkpoint directory exists
os.makedirs("checkpoints", exist_ok=True)


# Validation function
def validate(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    total_batches = 0

    with torch.no_grad():
        for anchor, positive, negatives in val_loader:
            anchor = anchor.to(device)
            positive = positive.to(device)
            negatives = [neg.to(device) for neg in negatives]

            # Get embeddings
            anchor_features = model.forward_features(anchor)
            positive_features = model.forward_features(positive)
            negative_features = [model.forward_features(neg) for neg in negatives]

            # Normalize features
            anchor_features = F.normalize(anchor_features, p=2, dim=1)
            positive_features = F.normalize(positive_features, p=2, dim=1)
            negative_features = [
                F.normalize(neg_feat, p=2, dim=1) for neg_feat in negative_features
            ]

            # Compute loss
            loss = criterion(anchor_features, positive_features, negative_features)

            running_loss += loss.item()
            total_batches += 1

    model.train()
    return running_loss / total_batches


def load_test_from_folders(base_folder):
    image_paths = []
    labels = []
    supported_extensions = (".bmp", ".jpg", ".jpeg", ".png", ".JPG")
    for root, _, files in os.walk(base_folder):
        files = [
            file for file in files if file.lower().endswith(supported_extensions)
        ]  # Filter only .bmp files

        # Sort files to maintain consistent ordering if needed
        files.sort()
        for file in files:
            img_path = os.path.join(root, file)
            image_paths.append(img_path)
            # Extract the label from the last number in the file name before the extension
            label_str = file.split("_")[-1].split(".")[0]
            try:
                label = int(label_str)  # Convert to integer
            except ValueError:
                print(f"Warning: Could not extract label from {file}")
                continue  # Skip files with incorrect format

            labels.append(label)

    return image_paths, labels


test_image_folder = r"raw/test"
test_image_paths, test_labels = load_test_from_folders(test_image_folder)

if test:
    test_image_paths = test_image_paths[:train_size]
    test_labels = test_labels[:train_size]

# Preprocessing for test dataset
test_dataset = TripletDataset(
    test_image_paths, test_labels, transform=transform, n_negatives=n_negatives
)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


# Validation and test functions (they can be similar)
def test(model, test_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    total_batches = 0

    with torch.no_grad():
        for anchor, positive, negatives in test_loader:
            anchor = anchor.to(device)
            positive = positive.to(device)
            negatives = [neg.to(device) for neg in negatives]

            # Get embeddings
            anchor_features = model.forward_features(anchor)
            positive_features = model.forward_features(positive)
            negative_features = [model.forward_features(neg) for neg in negatives]

            # Normalize features
            anchor_features = F.normalize(anchor_features, p=2, dim=1)
            positive_features = F.normalize(positive_features, p=2, dim=1)
            negative_features = [
                F.normalize(neg_feat, p=2, dim=1) for neg_feat in negative_features
            ]

            # Compute loss
            loss = criterion(anchor_features, positive_features, negative_features)

            running_loss += loss.item()
            total_batches += 1

    model.train()
    return running_loss / total_batches


# Training loop with triplet loss and validation
def train(
    model, train_loader, val_loader, test_loader, optimizer, epochs=10, device="cuda"
):
    # Initialize the semi-hard triplet loss
    criterion = SemiHardTripletLoss(margin=1.0).to(device)
    model.train()
    test_loss = test(model, test_loader, criterion, device)
    print(f"Before fine tune, Test Loss: {test_loss}")
    min_test_loss = test_loss

    for epoch in range(epochs):
        running_loss = 0.0
        total_batches = 0

        epoch_iterator = tqdm(
            train_loader, desc=f"Epoch [{epoch+1}/{epochs}]", unit="batch"
        )

        for i, (anchor, positive, negatives) in enumerate(epoch_iterator):
            anchor = anchor.to(device)
            positive = positive.to(device)
            negatives = [neg.to(device) for neg in negatives]

            # Get embeddings
            anchor_features = model.forward_features(anchor)
            positive_features = model.forward_features(positive)
            negative_features = [model.forward_features(neg) for neg in negatives]

            # Normalize features
            anchor_features = F.normalize(anchor_features, p=2, dim=1)
            positive_features = F.normalize(positive_features, p=2, dim=1)
            negative_features = [
                F.normalize(neg_feat, p=2, dim=1) for neg_feat in negative_features
            ]

            # Compute loss
            loss = criterion(anchor_features, positive_features, negative_features)

            # Backward pass and optimization
            optimizer.zero_grad()

            # Check if the loss is non-zero before proceeding
            if loss.item() > 0:
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                total_batches += 1

            epoch_iterator.set_postfix({"loss": loss.item()})

        avg_loss = running_loss / total_batches
        print(f"Epoch [{epoch+1}/{epochs}], Training Loss: {avg_loss}")

        # Validate the model after each epoch
        val_loss = validate(model, val_loader, criterion, device)
        print(f"Epoch [{epoch+1}/{epochs}], Validation Loss: {val_loss}")

        test_loss = test(model, test_loader, criterion, device)
        print(f"Epoch [{epoch+1}/{epochs}], Test Loss: {test_loss}")

        # Log the epoch, training loss, and validation loss to a file
        with open("checkpoints/training_log.txt", "a") as log_file:
            log_file.write(
                f"Epoch [{epoch+1}/{epochs}], Training Loss: {avg_loss}, "
                f"Validation Loss: {val_loss}, Test Loss: {test_loss}\n"
            )

        # Save model checkpoint
        if test_loss < min_test_loss:
            torch.save(
                model.state_dict(),
                f"checkpoints/fine_tuned_mamba_vision_L2_e{epoch+1}.pth",
            )
            min_test_loss = test_loss


# Run the training process with progress bars and validation
train(
    model,
    train_loader,
    val_loader,
    test_loader,
    optimizer,
    epochs=epoch,
    device=device,
)

# Save the fine-tuned model
