import torch
from torch import nn, optim
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset, random_split
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from PIL import Image
from models import mamba_vision_L2
from torch.nn.functional import normalize
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
batch_size = 4
epochs = 100
lr = 1e-4
test = True
n_negatives = 4
num_label_negative = 10
weight_decay = 0.001
min_lr = 1e-5

# continue_checkpoint = r"checkpoints/fine_tuned_mamba_vision_L2_latest.pth"
continue_checkpoint = r""

if test:
    n_negatives = 4
    batch_size = 4
    train_size = 40

# Load the pre-trained model
if not continue_checkpoint:
    model = mamba_vision_L2(pretrained=True).to(device)
else:
    model = mamba_vision_L2(pretrained=False).to(
        device
    )  # Initialize the model without pretrained weights
    model.load_state_dict(torch.load(continue_checkpoint))
# Freeze all layers except the last few
for param in model.parameters():
    param.requires_grad = False

last_mamba_layer = model.levels[-1]

for param in last_mamba_layer.parameters():
    param.requires_grad = True


train_image_folder = r"raw/train"
test_image_folder = (
    r"raw/dataset-test/Sapienza-University-Mobile-Palmprint-Database-ROI-by-our"
)
image_paths, labels = load_images_from_folders(train_image_folder)
test_image_paths, test_labels = load_images_from_folders(test_image_folder)

if test:
    image_paths = image_paths[:train_size]
    labels = labels[:train_size]
    # test_image_paths = test_image_paths[:train_size]
    # test_labels = test_labels[:train_size]

test_set = TripletDataset(
    test_image_paths,
    test_labels,
    transform=transform,
    n_negatives=80,
    num_classes_for_negative=100,
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
    batch_size=batch_size,
    shuffle=True,
    collate_fn=triplet_collate_fn,
    num_workers=4,
    persistent_workers=True,
    pin_memory=True,
)
test_loader = DataLoader(
    test_set,
    batch_size=4,
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
scheduler = CosineAnnealingWarmRestarts(
    optimizer, eta_min=min_lr, T_0=20 // batch_size, T_mult=1
)


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
                optimizer.zero_grad(set_to_none=True)

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
            train_loss = (
                running_loss / total_batches if total_batches > 0 else running_loss
            )
            val_loss = evaluate(model, val_loader, device)
            test_loss = evaluate(model, test_loader, device)

            print(
                f"Epoch [{epoch+1}/{epochs}]: Training Loss: {train_loss}, Validation Loss: {val_loss}, Test Loss: {test_loss}"
            )
    finally:
        print("Saving checkpoints, don't close!")
        torch.save(
            model.state_dict(),
            f"checkpoints/fine_tuned_mamba_vision_L2_latest.pth",
        )