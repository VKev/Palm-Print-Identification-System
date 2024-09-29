import torch
from torch import nn, optim
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset, random_split
from PIL import Image
from models import mamba_vision_L2
from torch.nn.functional import normalize
import random
import os
from tqdm import tqdm

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 220
epoch = 30
lr = 1e-5
test = True

if test:
    batch_size = 10
    train_size = 1000


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
    def __init__(self, image_paths, labels, transform=None, augmentation=None):
        self.image_paths = image_paths
        self.labels = labels
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

        # Negative image (different label)
        negative_indices = [
            i for i, label in enumerate(self.labels) if label != anchor_label
        ]
        negative_idx = random.choice(negative_indices)
        negative_image = Image.open(self.image_paths[negative_idx]).convert("RGB")

        # Apply augmentations (rotation, color jitter, etc.)
        if self.augmentation:
            # Apply augmentation to anchor, positive, negative images randomly
            if random.random() < 0.5:
                anchor_image = self.augmentation(anchor_image)
            else:
                anchor_image = self.transform(anchor_image)

            if random.random() < 0.5:
                positive_image = self.augmentation(positive_image)
            else:
                positive_image = self.transform(positive_image)

            if random.random() < 0.5:
                negative_image = self.augmentation(negative_image)
            else:
                negative_image = self.transform(negative_image)
        else:
            # If no augmentation, apply standard transformation to all images
            anchor_image = self.transform(anchor_image)
            positive_image = self.transform(positive_image)
            negative_image = self.transform(negative_image)

        return anchor_image, positive_image, negative_image


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
    global_label_counter = 0  # Initialize the global label counter
    image_counter = 0  # Counter to keep track of how many images we've processed

    for root, _, files in os.walk(base_folder):
        files = [
            file for file in files if file.lower().endswith(".bmp")
        ]  # Filter only .bmp files

        # Sort files to maintain consistent ordering if needed
        files.sort()

        for file in files:
            img_path = os.path.join(root, file)
            image_paths.append(img_path)

            # Assign a unique label that increases after every 10 images
            label = global_label_counter
            labels.append(label)

            image_counter += 1

            # Increment global label counter every 10 images
            if image_counter % 10 == 0:
                global_label_counter += 1

    return image_paths, labels


image_folder = r"raw/train"

image_paths, labels = load_images_from_folders(image_folder)
if test:
    image_paths = image_paths[:train_size]
    labels = labels[:train_size]
# Preprocessing
# Load the dataset
triplet_default_dataset = TripletDataset(image_paths, labels, transform=transform)
triplet_argumentation_dataset = TripletDataset(
    image_paths, labels, augmentation=augmentation, transform=transform
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
def validate(model, loader, criterion):
    model.eval()  # Set model to evaluation mode
    val_loss = 0.0

    with torch.no_grad():  # Disable gradient calculation
        for anchor, positive, negative in loader:
            anchor, positive, negative = (
                anchor.to(device),
                positive.to(device),
                negative.to(device),
            )

            # Forward pass
            anchor_features = model.forward_features(anchor)
            positive_features = model.forward_features(positive)
            negative_features = model.forward_features(negative)

            # Normalize features
            anchor_features = normalize(anchor_features, p=2, dim=1)
            positive_features = normalize(positive_features, p=2, dim=1)
            negative_features = normalize(negative_features, p=2, dim=1)

            # Compute validation loss
            loss = criterion(anchor_features, positive_features, negative_features)
            val_loss += loss.item()

    return val_loss / len(loader)


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
test_dataset = TripletDataset(test_image_paths, test_labels, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


# Validation and test functions (they can be similar)
def test(model, loader, criterion):
    model.eval()  # Set model to evaluation mode
    test_loss = 0.0

    with torch.no_grad():  # Disable gradient calculation
        for anchor, positive, negative in loader:
            anchor, positive, negative = (
                anchor.to(device),
                positive.to(device),
                negative.to(device),
            )

            # Forward pass
            anchor_features = model.forward_features(anchor)
            positive_features = model.forward_features(positive)
            negative_features = model.forward_features(negative)

            # Normalize features
            anchor_features = normalize(anchor_features, p=2, dim=1)
            positive_features = normalize(positive_features, p=2, dim=1)
            negative_features = normalize(negative_features, p=2, dim=1)

            # Compute test loss
            loss = criterion(anchor_features, positive_features, negative_features)
            test_loss += loss.item()

    return test_loss / len(loader)


# Training loop with triplet loss and validation
def train(model, train_loader, val_loader, criterion, optimizer, epochs=10):
    model.train()  # Set model to training mode
    log_file_path = "checkpoints/training_log.txt"  # Path for log file

    for epoch in range(epochs):
        running_loss = 0.0

        # Add tqdm progress bar for each epoch
        epoch_iterator = tqdm(
            train_loader, desc=f"Epoch [{epoch+1}/{epochs}]", unit="batch"
        )

        for i, (anchor, positive, negative) in enumerate(epoch_iterator):
            anchor, positive, negative = (
                anchor.to(device),
                positive.to(device),
                negative.to(device),
            )

            # Forward pass: extract features
            anchor_features = model.forward_features(anchor)
            positive_features = model.forward_features(positive)
            negative_features = model.forward_features(negative)

            # Normalize features
            anchor_features = normalize(anchor_features, p=2, dim=1)
            positive_features = normalize(positive_features, p=2, dim=1)
            negative_features = normalize(negative_features, p=2, dim=1)

            # Compute loss
            loss = criterion(anchor_features, positive_features, negative_features)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{epochs}], Training Loss: {avg_loss}")

        # Validate the model after each epoch
        val_loss = validate(model, val_loader, criterion)
        print(f"Epoch [{epoch+1}/{epochs}], Validation Loss: {val_loss}")

        test_loss = test(model, test_loader, criterion)
        print(f"Epoch [{epoch+1}/{epochs}], Test Loss: {test_loss}")

        # Log the epoch, training loss, and validation loss to a file
        with open(log_file_path, "a") as log_file:
            log_file.write(
                f"Epoch [{epoch+1}/{epochs}], Training Loss: {avg_loss}, Validation Loss: {val_loss}, Test Loss: {test_loss}\n"
            )

        torch.save(model.state_dict(), "checkpoints/fine_tuned_mamba_vision_L2.pth")


# Run the training process with progress bars and validation
train(model, train_loader, val_loader, triplet_loss, optimizer, epochs=epoch)

# Save the fine-tuned model
