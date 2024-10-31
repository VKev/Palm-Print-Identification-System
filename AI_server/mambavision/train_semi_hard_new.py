import argparse
import torch
import numpy as np
from torch import nn, optim
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset, random_split
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from PIL import Image
from models import mamba_vision_T, CustomHead, add_lora_to_model
from torch.nn.functional import normalize
from timm.models import create_model
import random
import torch.nn.functional as F
import os
import wandb
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

# test_custom_head()

# pass

wandb.login(key="b8b74d6af92b4dea7706baeae8b86083dad5c941")
pc_info()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser(description='Training parameters for the model.')
parser.add_argument('--train_image_folder', type=str, default=r"raw/train", help='Path to the training images folder')
parser.add_argument('--test_image_folder', type=str, default=r"raw/dataset-test/Sapienza-University-Mobile-Palmprint-Database-ROI-by-our", help='Path to the test images folder')
parser.add_argument('--batch_size', type=int, default=24, help='Batch size for training')
parser.add_argument('--epochs', type=int, default=1000, help='Number of epochs to train')
parser.add_argument('--lr', type=float, default=5e-4, help='Learning rate')
parser.add_argument('--test', type=bool, default=False, help='Whether to run in test mode')
parser.add_argument('--n_negatives', type=int, default=140, help='Number of negatives for training')
parser.add_argument('--num_label_negative', type=int, default=96, help='Number of negative labels')
parser.add_argument('--weight_decay', type=float, default=0.05, help='Weight decay for optimization')
parser.add_argument('--min_lr', type=float, default=1e-5, help='Minimum learning rate')
parser.add_argument('--num_workers', type=int, default=1, help='Number of workers for data loading')
parser.add_argument('--test_negatives', type=int, default=100, help='Number of test negatives')
parser.add_argument('--test_negatives_class', type=int, default=100, help='Number of test negatives per class')
parser.add_argument('--val_batch_size', type=int, default=24, help='Batch size for validation')
parser.add_argument('--test_batch', type=int, default=32, help='Batch size for testing')
args = parser.parse_args()

train_image_folder = args.train_image_folder
test_image_folder = args.test_image_folder
batch_size = args.batch_size
epochs = args.epochs
lr = args.lr
test = args.test
n_negatives = args.n_negatives
num_label_negative = args.num_label_negative
weight_decay = args.weight_decay
min_lr = args.min_lr
num_workers = args.num_workers
test_negatives = args.test_negatives
test_negatives_class = args.test_negatives_class
val_batch_size = args.val_batch_size
test_batch = args.test_batch

# continue_checkpoint = r"checkpoints/fine_tuned_mamba_vision_T_latest_19.pth"
continue_checkpoint = r""
wandb.init(
    # set the wandb project where this run will be logged
    project="mamba-vision-palm-print",
    # track hyperparameters and run metadata
    config={
        "learning_rate": lr,
        "epochs": epochs,
    },
)

if test:
    val_batch_size = 4
    test_batch = 4
    batch_size = 4

    n_negatives = 16
    num_label_negative = 12

    train_size = 1000

    test_negatives = 16
    test_negatives_class = 12


# Load the pre-trained model
if not continue_checkpoint:
    print("Initializing")
    model = mamba_vision_T(pretrained=True).to(device)
    for param in model.parameters():
        param.requires_grad = False

    add_lora_to_model(model=model, rank=4, alpha=1, target_modules=["head", "levels.3.blocks.3.mlp.fc2", "levels.3.blocks.3.mlp.fc1", "levels.3.blocks.3.mixer.proj", "levels.3.blocks.3.mixer.qkv", "levels.3.blocks.2.mlp.fc2", "levels.3.blocks.2.mlp.fc1", "levels.3.blocks.2.mixer.proj", "levels.3.blocks.2.mixer.qkv" ])
    model.to(device)
else:
    print("Loading")
    model =  mamba_vision_T(pretrained=False).to(device)
    add_lora_to_model(model=model, rank=4, alpha=1, target_modules=["head", "levels.3.blocks.3.mlp.fc2", "levels.3.blocks.3.mlp.fc1", "levels.3.blocks.3.mixer.proj", "levels.3.blocks.3.mixer.qkv", "levels.3.blocks.2.mlp.fc2", "levels.3.blocks.2.mlp.fc1", "levels.3.blocks.2.mixer.proj", "levels.3.blocks.2.mixer.qkv" ])
    model.load_state_dict(torch.load(continue_checkpoint))
    for param in model.parameters():
        param.requires_grad = False
    for name, param in model.named_parameters():
        if "lora_" in name:
            param.requires_grad = True
    model.to(device)            


for name, param in model.named_parameters():
    if param.requires_grad:
        print(f"Module Name: {name}, Requires Grad: {param.requires_grad}")



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

head_params = count_parameters(model.head)
print(f"Number of parameters in head : {head_params}")


print(model)


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
    num_workers=num_workers,
    persistent_workers=True,
    pin_memory=True,
)
val_loader = DataLoader(
    val_dataset,
    batch_size=val_batch_size,
    shuffle=True,
    collate_fn=triplet_collate_fn,
    num_workers=num_workers,
    persistent_workers=True,
    pin_memory=True,
)
test_loader = DataLoader(
    test_set,
    batch_size=test_batch,
    shuffle=False,
    collate_fn=triplet_collate_fn,
    num_workers=num_workers,
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
        # val_loss = evaluate(model, val_loader, device)
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
                wandb.log({"batch_loss": loss})

                if loss.item() > 0:
                    loss.backward()
                    for name, param in model.named_parameters():
                        if param.grad is not None:
                            grad_data = param.grad.cpu().data.numpy()
                            hist, bin_edges = np.histogram(grad_data, bins=100)
                            wandb.log({
                                f"gradients/{name}": wandb.Histogram(np_histogram=(hist, bin_edges))
                            })
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
            wandb.log(
                {
                    "training_loss": train_loss,
                    "val_loss": val_loss,
                    "test_loss": test_loss,
                }
            )
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
