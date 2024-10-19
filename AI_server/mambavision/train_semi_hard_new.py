import torch
from torch import nn, optim
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset, random_split
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from PIL import Image
from models import mamba_vision_T
from torch.nn.functional import normalize
from timm.models import create_model
import random
import torch.nn.functional as F
import os
from tqdm import tqdm
import re
from utils import (
    load_images_from_folders,
    TripletDataset,
    transform,
    augmentation,
    triplet_collate_fn,
    BatchAllTripletLoss,
    CombinedTripletDataset,
    pc_info,
)

pc_info()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 20
epochs = 1000
lr = 5e-4
test = False
n_negatives = 140
num_label_negative = 96
weight_decay = 0.05
min_lr = 1e-5

test_negatives = 100
test_negatives_class = 100
val_batch_size = 16
test_batch = 12
continue_checkpoint = r"checkpoints/best/fine_tuned_mamba_vision_T_latest_19.pth"
# continue_checkpoint = r""

if test:
    val_batch_size = 4
    n_negatives = 4
    test_batch = 1
    batch_size = 4
    train_size = 200
    test_negatives = 16
    test_negatives_class = 8


# Load the pre-trained model
if not continue_checkpoint:
    print("Initializing")
    model = create_model(
        "mamba_vision_T", pretrained=True, drop_rate=0.3, attn_drop_rate=0.2
    ).to(device)
    model.head = nn.Sequential(
        model.head,  # Original head layer that outputs 1000 features
        nn.ReLU(),  # Add an activation function (optional, e.g., ReLU)
        nn.Linear(
            in_features=1000, out_features=256, bias=True
        ),  # Bottleneck layer reducing to 256
    )
else:
    print("Loading")
    model = create_model(
        "mamba_vision_T", pretrained=True, drop_rate=0.3, attn_drop_rate=0.2
    ).to(device)
    model.head = nn.Sequential(
        model.head,  # Original head layer that outputs 1000 features
        nn.ReLU(),  # Add an activation function (optional, e.g., ReLU)
        nn.Linear(
            in_features=1000, out_features=256, bias=True
        ),  # Bottleneck layer reducing to 256
    )  # Initialize the model without pretrained weights
    model.load_state_dict(torch.load(continue_checkpoint))
    # Freeze all layers except the last few

for param in model.parameters():
    param.requires_grad = False

last_mamba_layer = model.levels[-1]

# for param in last_mamba_layer.parameters():
#     param.requires_grad = True
# print(model)
total_params_3 = 0
total_params_2 = 0
for name, param in last_mamba_layer.named_parameters():
    if "blocks.3" in name:
        param.requires_grad = True
        total_params_3 += param.numel()
    # if "blocks.2" in name:
    #     param.requires_grad = True
    #     total_params_2 += param.numel()
    # print(f"Set requires_grad=True for {name}")

for name, param in model.named_parameters():
    if "head" in name:  # If the layer is in the head
        param.requires_grad = True
    if name in ["norm.weight", "norm.bias"]:
        param.requires_grad = True

for name, param in model.named_parameters():
    if param.requires_grad:
        print(f"Module Name: {name}, Requires Grad: {param.requires_grad}")


print(f"Total param in block 3: {total_params_3}")
print(f"Total param in block 2: {total_params_2}")
# last_mamba_layer = model.levels[-2]

# for param in last_mamba_layer.parameters():
#     param.requires_grad = True


def count_parameters(layer):
    return sum(p.numel() for p in layer.parameters())


# Count parameters in the whole model
total_params = count_parameters(model)
print(f"Total number of parameters in the model: {total_params}")

# Count parameters in level 2
level_2_params = count_parameters(model.levels[-2])
print(f"Number of parameters in level 2: {level_2_params}")

# Count parameters in level 1
level_1_params = count_parameters(model.levels[-1])
print(f"Number of parameters in level 1: {level_1_params}")
print(model)

train_image_folder = r"raw/train"
test_image_folder = (
    r"raw/dataset-test/Sapienza-University-Mobile-Palmprint-Database-ROI-by-our"
)
image_paths, labels = load_images_from_folders(train_image_folder)
test_image_paths, test_labels = load_images_from_folders(test_image_folder)

if test:
    image_paths = image_paths[:train_size]
    labels = labels[:train_size]
    test_image_paths = test_image_paths[:train_size]
    test_labels = test_labels[:train_size]

test_set = TripletDataset(
    test_image_paths,
    test_labels,
    transform=transform,
    n_negatives=test_negatives,
    num_classes_for_negative=test_negatives_class,
)

train_set = TripletDataset(
    image_paths,
    labels,
    transform=transform,
    n_negatives=n_negatives,
    num_classes_for_negative=num_label_negative,
)

augmentation_set = TripletDataset(
    image_paths,
    labels,
    transform=transform,
    n_negatives=n_negatives,
    augmentation=augmentation,
    num_classes_for_negative=num_label_negative,
)
combined_set = CombinedTripletDataset(train_set, augmentation_set)
train_size = int(0.9 * len(combined_set))
val_size = len(combined_set) - train_size
train_dataset, val_dataset = random_split(combined_set, [train_size, val_size])

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    collate_fn=triplet_collate_fn,
    num_workers=4,
    persistent_workers=True,
    pin_memory=True,
)
val_loader = DataLoader(
    val_dataset,
    batch_size=val_batch_size,
    shuffle=True,
    collate_fn=triplet_collate_fn,
    num_workers=4,
    persistent_workers=True,
    pin_memory=True,
)
test_loader = DataLoader(
    test_set,
    batch_size=test_batch,
    shuffle=False,
    collate_fn=triplet_collate_fn,
    num_workers=4,
    persistent_workers=True,
    pin_memory=True,
)


triplet_loss = BatchAllTripletLoss(margin=1)

optimizer = optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=lr,
    weight_decay=weight_decay,
    betas=(0.9, 0.999),
    eps=1e-8,
)
scheduler = CosineAnnealingWarmRestarts(optimizer, eta_min=min_lr, T_0=1, T_mult=1)


def evaluate(model, data_loader, device):
    model.eval()
    total_loss = 0.0
    total_batches = 0

    with torch.no_grad():
        for all_images, num_anchors, num_negatives_per_anchor in tqdm(
            data_loader, desc="Evaluating"
        ):
            all_images = all_images.to(device)

            all_features = model.forward_features(all_images)

            anchors_features = all_features[:num_anchors]
            positives_features = all_features[num_anchors : 2 * num_anchors]
            negatives_features = all_features[2 * num_anchors :].view(
                num_anchors, num_negatives_per_anchor, -1
            )

            anchors_features = F.normalize(anchors_features, p=2, dim=1)
            positives_features = F.normalize(positives_features, p=2, dim=1)
            negatives_features = F.normalize(negatives_features, p=2, dim=2)

            loss = triplet_loss(
                anchors_features, positives_features, negatives_features
            )

            if loss.item() > 0:
                total_loss += loss.item()
            total_batches += 1

    avg_loss = total_loss / total_batches if total_batches > 0 else total_loss
    return avg_loss


if __name__ == "__main__":
    try:
        test_loss = evaluate(model, test_loader, device)
        print(f"Test Loss: {test_loss}")
        for epoch in range(epochs):
            model.train()
            running_loss = 0.0
            total_batches = 0
            epoch_iterator = tqdm(
                train_loader, desc=f"Epoch [{epoch+1}/{epochs}]", unit="batch"
            )

            for i, (all_images, num_anchors, num_negatives_per_anchor) in enumerate(
                epoch_iterator
            ):
                optimizer.zero_grad(set_to_none=True)

                # print("TEST")
                all_images = all_images.to(device)

                all_features = model.forward_features(all_images)

                anchors_features = all_features[:num_anchors]
                positives_features = all_features[num_anchors : 2 * num_anchors]
                negatives_features = all_features[2 * num_anchors :].view(
                    num_anchors, num_negatives_per_anchor, -1
                )

                anchors_features = F.normalize(anchors_features, p=2, dim=1)
                positives_features = F.normalize(positives_features, p=2, dim=1)
                negatives_features = F.normalize(negatives_features, p=2, dim=2)
                # print("positive")
                # print(positives_features)
                # print("negatives_list")
                # print(negatives_features)
                loss = triplet_loss(
                    anchors_features, positives_features, negatives_features
                )

                if loss.item() > 0:
                    loss.backward()
                    optimizer.step()

                    running_loss += loss.item()
                total_batches += 1
                epoch_iterator.set_postfix(
                    loss=loss.item(), lr=optimizer.param_groups[0]["lr"]
                )
                # print("TEST")
                # print(positive)
                # print("TEST")
                # print(negatives)
                scheduler.step()
                del anchors_features, positives_features, negatives_features
                torch.cuda.empty_cache()
            train_loss = (
                running_loss / total_batches if total_batches > 0 else running_loss
            )
            torch.cuda.empty_cache()
            val_loss = evaluate(model, val_loader, device)
            torch.cuda.empty_cache()
            test_loss = evaluate(model, test_loader, device)
            with open("checkpoints/loss.txt", "a") as f:
                f.write(f"Epoch [{epoch+1}/{epochs}]: ")
                f.write(f"Training Loss: {train_loss}, ")
                f.write(f"Validation Loss: {val_loss}, ")
                f.write(f"Test Loss: {test_loss}\n")
            print(
                f"Epoch [{epoch+1}/{epochs}]: Training Loss: {train_loss}, Validation Loss: {val_loss}, Test Loss: {test_loss}"
            )
            torch.save(
                model.state_dict(),
                f"checkpoints/fine_tuned_mamba_vision_T_latest_{epoch+1}.pth",
            )
            torch.cuda.empty_cache()
    finally:
        print("Saving checkpoints, don't close!")
        torch.save(
            model.state_dict(),
            f"checkpoints/fine_tuned_mamba_vision_T_latest.pth",
        )
