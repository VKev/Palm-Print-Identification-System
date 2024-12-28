import argparse
import os
import random
from typing import Tuple
import numpy as np
import torch
import torch.nn.functional as F
import wandb
from PIL import Image
from timm.models import create_model
from torch import nn, optim
from torch.nn.functional import normalize
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
from tqdm import tqdm

from models import mamba_vision_T, CustomHead, add_lora_to_model
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

def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Training parameters for the model.')
    parser.add_argument('--train_image_folder', type=str, default=r"../../../Dataset/Palm-Print/TrainAndTest/train", 
                       help='Path to the training images folder')
    parser.add_argument('--train_aug_image_folder', type=str, default=r"raw/train-aug", 
                       help='Path to the training images folder')
    parser.add_argument('--test_image_folder', type=str, 
                       default=r"../../../Dataset/Palm-Print/TrainAndTest/test", 
                       help='Path to the test images folder')
    parser.add_argument('--batch_size', type=int, default=24, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=10000, help='Number of epochs to train')
    parser.add_argument('--lr', type=float, default=2e-4, help='Learning rate')
    parser.add_argument('--test', type=bool, default=False, help='Whether to run in test mode')
    parser.add_argument('--n_negatives', type=int, default=140, help='Number of negatives for training')
    parser.add_argument('--num_label_negative', type=int, default=96, help='Number of negative labels')
    parser.add_argument('--weight_decay', type=float, default=2e-5, help='Weight decay for optimization')
    parser.add_argument('--min_lr', type=float, default=0.00005, help='Minimum learning rate')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for data loading')
    parser.add_argument('--test_negatives', type=int, default=100, help='Number of test negatives')
    parser.add_argument('--test_negatives_class', type=int, default=100, help='Number of test negatives per class')
    parser.add_argument('--val_batch_size', type=int, default=24, help='Batch size for validation')
    parser.add_argument('--test_batch', type=int, default=32, help='Batch size for testing')
    parser.add_argument('--checkpoint_path', type=str, default="", help='Path to checkpoint file for continuing training')
    
    return parser.parse_args()

def setup_wandb(lr: float, epochs: int):
    """Initialize Weights & Biases logging."""
    wandb.login(key="b8b74d6af92b4dea7706baeae8b86083dad5c941")
    wandb.init(
        project="mamba-vision-palm-print",
        config={
            "learning_rate": lr,
            "epochs": epochs,
        },
    )

def get_lora_target_modules() -> list:
    """Get the list of target modules for LoRA adaptation."""
    return [
        "head",
        # Level 3 blocks
        "levels.3.blocks.3.mlp.fc2",
        "levels.3.blocks.3.mlp.fc1",
        "levels.3.blocks.3.mixer.proj",
        "levels.3.blocks.3.mixer.qkv",
        "levels.3.blocks.2.mlp.fc2",
        "levels.3.blocks.2.mlp.fc1",
        "levels.3.blocks.2.mixer.proj",
        "levels.3.blocks.2.mixer.qkv",
        "levels.3.blocks.1.mlp.fc2",
        "levels.3.blocks.1.mlp.fc1",
        "levels.3.blocks.1.mixer.in_proj",
        "levels.3.blocks.1.mixer.x_proj",
        "levels.3.blocks.1.mixer.dt_proj",
        "levels.3.blocks.1.mixer.out_proj",
        "levels.3.blocks.0.mlp.fc2",
        "levels.3.blocks.0.mlp.fc1",
        "levels.3.blocks.0.mixer.in_proj",
        "levels.3.blocks.0.mixer.x_proj",
        "levels.3.blocks.0.mixer.dt_proj",
        "levels.3.blocks.0.mixer.out_proj",
        # Level 2 blocks
        "levels.2.blocks.7.mlp.fc2",
        "levels.2.blocks.7.mlp.fc1",
        "levels.2.blocks.7.mixer.proj",
        "levels.2.blocks.7.mixer.qkv"
    ]

def initialize_model(device: torch.device, args: argparse.Namespace,checkpoint_path: str = "") -> Tuple[torch.nn.Module, optim.Optimizer, object, int]:
    """Initialize the model, optimizer, and scheduler."""
    start_epoch = 0
    
    if not checkpoint_path:
        model = mamba_vision_T(pretrained=True).to(device)
        
        # Freeze all parameters initially
        for param in model.parameters():
            param.requires_grad = False
        
        # Add LoRA to specific layers
        # add_lora_to_model(
        #     model=model,
        #     rank=4,
        #     alpha=1,
        #     target_modules=get_lora_target_modules()
        # )
        # model.head = CustomHead()

        for param in model.levels[3].parameters():
            param.requires_grad = True

        for param in model.levels[2].parameters():
            param.requires_grad = True
        
        for param in model.levels[1].parameters():
            param.requires_grad = True

        model.to(device)
        
        optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=args.lr,
            weight_decay=args.weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8,
        )
        
        scheduler = CosineAnnealingWarmRestarts(optimizer, eta_min=args.min_lr, T_0=1, T_mult=1)
    else:
        model = mamba_vision_T(pretrained=False).to(device)
        # add_lora_to_model(
        #     model=model,
        #     rank=4,
        #     alpha=1,
        #     target_modules=get_lora_target_modules()
        # )
        # model.head = CustomHead()
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint['epoch']
        
        # Set parameter gradients
        for param in model.parameters():
            param.requires_grad = False
        for name, param in model.named_parameters():
            if "lora_" in name:
                param.requires_grad = True
        for param in model.levels[3].parameters():
            param.requires_grad = True

        for param in model.levels[2].parameters():
            param.requires_grad = True
        
        for param in model.levels[1].parameters():
            param.requires_grad = True
        # for param in model.head.parameters():
        #     param.requires_grad = True        
        model.to(device)
        
        optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=args.lr,
            weight_decay=args.weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8,
        )
        scheduler = CosineAnnealingWarmRestarts(optimizer, eta_min=args.min_lr, T_0=1, T_mult=1)
        
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    return model, optimizer, scheduler, start_epoch

def save_checkpoint(model: torch.nn.Module, optimizer: optim.Optimizer, scheduler: object, 
                   epoch: int, loss: float, checkpoint_dir: str = "checkpoints"):
    """Save model checkpoint and training state."""
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'loss': loss
    }
    checkpoint_path = f"{checkpoint_dir}/checkpoint_epoch_{epoch}.pth"
    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved at {checkpoint_path}")

def evaluate(model: torch.nn.Module, data_loader: DataLoader, device: torch.device, 
            triplet_loss: object) -> float:
    """Evaluate the model on the given data loader."""
    model.eval()
    total_loss = 0.0
    total_batches = 0

    with torch.no_grad():
        for all_images, num_anchors, num_negatives_per_anchor in tqdm(data_loader, desc="Evaluating"):
            all_images = all_images.to(device)
            all_features = model.forward(all_images)

            # Split features into anchors, positives, and negatives
            anchors_features = all_features[:num_anchors]
            positives_features = all_features[num_anchors:2*num_anchors]
            negatives_features = all_features[2*num_anchors:].view(
                num_anchors, num_negatives_per_anchor, -1
            )

            # Normalize features
            anchors_features = F.normalize(anchors_features, p=2, dim=1)
            positives_features = F.normalize(positives_features, p=2, dim=1)
            negatives_features = F.normalize(negatives_features, p=2, dim=2)

            loss = triplet_loss(anchors_features, positives_features, negatives_features)
            if loss.item() > 0:
                total_loss += loss.item()
            total_batches += 1

    return total_loss / total_batches if total_batches > 0 else total_loss

def train_epoch(model: torch.nn.Module, train_loader: DataLoader, optimizer: optim.Optimizer, 
                scheduler: object, triplet_loss: object, device: torch.device, 
                epoch: int, total_epochs: int) -> float:
    """Train the model for one epoch."""
    model.train()
    running_loss = 0.0
    total_batches = 0
    
    epoch_iterator = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{total_epochs}]", unit="batch")

    for all_images, num_anchors, num_negatives_per_anchor in epoch_iterator:
        optimizer.zero_grad(set_to_none=True)
        
        # Forward pass
        all_images = all_images.to(device)
        all_features = model.forward(all_images)
        
        # Split and normalize features
        anchors_features = F.normalize(all_features[:num_anchors], p=2, dim=1)
        positives_features = F.normalize(all_features[num_anchors:2*num_anchors], p=2, dim=1)
        negatives_features = F.normalize(
            all_features[2*num_anchors:].view(num_anchors, num_negatives_per_anchor, -1),
            p=2, dim=2
        )
        
        # Calculate loss and update
        loss = triplet_loss(anchors_features, positives_features, negatives_features)
        wandb.log({"batch_loss": loss})

        if loss.item() > 0:
            loss.backward()
            log_gradients(model)
            optimizer.step()
            running_loss += loss.item()
            
        total_batches += 1
        epoch_iterator.set_postfix(loss=loss.item(), lr=optimizer.param_groups[0]["lr"])
        
        scheduler.step()
        torch.cuda.empty_cache()
    
    return running_loss / total_batches if total_batches > 0 else running_loss

def log_gradients(model: torch.nn.Module):
    """Log gradient histograms to W&B."""
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_data = param.grad.cpu().data.numpy()
            hist, bin_edges = np.histogram(grad_data, bins=100)
            wandb.log({
                f"gradients/{name}": wandb.Histogram(np_histogram=(hist, bin_edges))
            })

def setup_dataloaders(args: argparse.Namespace) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Set up train, validation, and test data loaders."""
    # Load datasets
    image_paths, labels = load_images_from_folders(args.train_image_folder)
    test_image_paths, test_labels = load_images_from_folders(args.test_image_folder)
    
    # Create datasets
    test_set = TripletDataset(
        test_image_paths, test_labels, transform=transform,
        n_negatives=args.test_negatives, num_classes_for_negative=args.test_negatives_class
    )

    image_paths_aug, labels_aug = load_images_from_folders(args.train_aug_image_folder)
    
    train_set = TripletDataset(
        image_paths, labels, transform=transform,
        n_negatives=args.n_negatives, num_classes_for_negative=args.num_label_negative
    )
    
    augmentation_set = TripletDataset(
        image_paths_aug, labels_aug, transform=transform,
        n_negatives=args.n_negatives, num_classes_for_negative=args.num_label_negative
    )
    # Combine and split datasets
    # combined_set = CombinedTripletDataset(train_set, augmentation_set)
    # train_size = int(0.9 * len(combined_set))
    # val_size = len(combined_set) - train_size
    # train_dataset, val_dataset = random_split(combined_set, [train_size, val_size])
    
    train_size = int(0.9 * len(train_set))
    val_size = len(train_set) - train_size
    train_dataset, val_dataset = random_split(train_set, [train_size, val_size])

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=triplet_collate_fn,
        num_workers=args.num_workers,
        persistent_workers=True,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.val_batch_size,
        shuffle=True,
        collate_fn=triplet_collate_fn,
        num_workers=args.num_workers,
        persistent_workers=True,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_set,
        batch_size=args.test_batch,
        shuffle=False,
        collate_fn=triplet_collate_fn,
        num_workers=args.num_workers,
        persistent_workers=True,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader

def update_test_settings(args: argparse.Namespace) -> argparse.Namespace:
    """Update settings for test mode."""
    if args.test:
        args.val_batch_size = 4
        args.test_batch = 4
        args.batch_size = 2
        args.n_negatives = 1
        args.num_label_negative = 1
        args.num_workers = 4
        args.test_negatives = 1
        args.test_negatives_class = 1
    return args

def main():
    # Set up environment
    args = parse_args()
    args = update_test_settings(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    setup_wandb(args.lr, args.epochs)
    
    # Initialize model and associated components
    model, optimizer, scheduler, start_epoch = initialize_model(device, args,args.checkpoint_path)
    triplet_loss = BatchAllTripletLoss(margin=0.75)
    
    # Set up data loaders
    train_loader, val_loader, test_loader = setup_dataloaders(args)
    
    # Print model statistics
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable_params}")
    print(f"Total parameters: {total_params}")
    
    # Initial evaluation if loading from checkpoint
    if args.checkpoint_path:
        print(f"Resuming from epoch {start_epoch}")
        initial_test_loss = evaluate(model, test_loader, device, triplet_loss)
        print(f"Initial test loss: {initial_test_loss}")
    
    torch.cuda.empty_cache()
    # Training loop
    try:
        for epoch in range(start_epoch, args.epochs):
            # Train
            train_loss = train_epoch(
                model, train_loader, optimizer, scheduler,
                triplet_loss, device, epoch, args.epochs
            )
            
            # Save checkpoint
            save_checkpoint(model, optimizer, scheduler, epoch+1, train_loss)
            
            # Evaluate
            torch.cuda.empty_cache()
            val_loss = evaluate(model, val_loader, device, triplet_loss)
            torch.cuda.empty_cache()
            test_loss = evaluate(model, test_loader, device, triplet_loss)
            
            # Log results
            with open("checkpoints/loss.txt", "a") as f:
                f.write(f"Epoch [{epoch+1}/{args.epochs}]: ")
                f.write(f"Training Loss: {train_loss:.6f}, ")
                f.write(f"Validation Loss: {val_loss:.6f}, ")
                f.write(f"Test Loss: {test_loss:.6f}\n")
                
            wandb.log({
                "training_loss": train_loss,
                "val_loss": val_loss,
                "test_loss": test_loss,
            })
            
            print(f"Epoch [{epoch+1}/{args.epochs}]: "
                  f"Training Loss: {train_loss:.6f}, "
                  f"Validation Loss: {val_loss:.6f}, "
                  f"Test Loss: {test_loss:.6f}")
            
            torch.cuda.empty_cache()
            
    except KeyboardInterrupt:
        print("\nTraining interrupted by user. Saving checkpoint...")
    except Exception as e:
        print(f"\nError occurred during training: {str(e)}")
        raise
    finally:
        # Save final checkpoint
        try:
            final_checkpoint_path = os.path.join("checkpoints", "final_model.pth")
            save_checkpoint(model, optimizer, scheduler, epoch+1, train_loss)
            print(f"Final checkpoint saved at {final_checkpoint_path}")
        except Exception as e:
            print(f"Error saving final checkpoint: {str(e)}")

if __name__ == "__main__":
    # Set random seeds for reproducibility
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    # Run the main training loop
    main()