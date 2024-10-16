import random
from PIL import Image
from torch.utils.data import DataLoader, Dataset, random_split
import torch


class CombinedTripletDataset(Dataset):
    def __init__(self, dataset1, dataset2):
        self.dataset1 = dataset1
        self.dataset2 = dataset2

    def __len__(self):
        return len(self.dataset1) + len(self.dataset2)

    def __getitem__(self, idx):
        if idx < len(self.dataset1):
            return self.dataset1[idx]
        else:
            return self.dataset2[idx - len(self.dataset1)]


class TripletDataset(Dataset):
    def __init__(
        self,
        image_paths,
        labels,
        n_negatives,
        num_classes_for_negative,
        transform=None,
        augmentation=None,
    ):
        self.image_paths = image_paths
        self.labels = labels
        self.n_negatives = (
            n_negatives  # Number of negative samples per anchor-positive pair
        )
        self.transform = transform
        self.augmentation = augmentation

        # Create a dictionary mapping labels to their indices
        self.label_to_indices = {}
        for idx, label in enumerate(self.labels):
            if label not in self.label_to_indices:
                self.label_to_indices[label] = []
            self.label_to_indices[label].append(idx)

        # Calculate the maximum possible different labels for negatives
        self.max_possible_neg_classes = (
            len(self.label_to_indices) - 1
        )  # -1 for anchor's class

        # Adjust num_classes_for_negative if it exceeds maximum possible
        self.num_classes_for_negative = min(
            num_classes_for_negative, self.max_possible_neg_classes
        )

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Anchor image
        anchor_image = Image.open(self.image_paths[idx]).convert("RGB")
        anchor_label = self.labels[idx]

        # Positive image (same label)
        positive_indices = [i for i in self.label_to_indices[anchor_label] if i != idx]
        positive_idx = random.choice(positive_indices)
        positive_image = Image.open(self.image_paths[positive_idx]).convert("RGB")
        # print("ANCHOR AND POSITIVE")
        # print(self.image_paths[idx])
        # print(self.image_paths[positive_idx])
        # Get multiple negative images with controlled label distribution
        available_labels = [
            label for label in self.label_to_indices.keys() if label != anchor_label
        ]

        # Select the required number of different labels for negatives
        selected_labels = random.sample(available_labels, self.num_classes_for_negative)

        # Calculate how many samples we need from each label
        samples_per_label = [
            self.n_negatives // self.num_classes_for_negative
        ] * self.num_classes_for_negative
        # Distribute remaining samples (if any)
        remaining = self.n_negatives % self.num_classes_for_negative
        for i in range(remaining):
            samples_per_label[i] += 1
        # print("NEGATIVE")
        negative_images = []
        for label, num_samples in zip(selected_labels, samples_per_label):
            # Get indices for this label
            label_indices = self.label_to_indices[label]
            # Select required number of samples for this label
            # Ensure we don't try to select more samples than available
            num_samples = min(num_samples, len(label_indices))
            selected_indices = random.sample(label_indices, num_samples)

            for neg_idx in selected_indices:
                # print(self.image_paths[neg_idx])
                negative_image = Image.open(self.image_paths[neg_idx]).convert("RGB")

                # Apply augmentations or transformations
                if self.augmentation and random.random() < 0.7:
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

            if random.random() < 0.7:
                positive_image = self.augmentation(positive_image)
            else:
                positive_image = self.transform(positive_image)
        else:
            anchor_image = self.transform(anchor_image)
            positive_image = self.transform(positive_image)

        return anchor_image, positive_image, negative_images


def triplet_collate_fn(batch):
    """
    Custom collate function to stack all images (anchor, positive, negatives)
    and return as a single tensor to pass through the model.
    """
    anchors = []
    positives = []
    negatives = []

    for anchor, positive, negative_list in batch:
        anchors.append(anchor)
        positives.append(positive)
        negatives.extend(negative_list)

    # Stack images
    anchors = torch.stack(anchors)  # Batch of anchor images
    positives = torch.stack(positives)  # Batch of positive images
    negatives = torch.stack(negatives)  # Batch of all negative images

    # Concatenate all (anchors, positives, negatives) into a single tensor
    all_images = torch.cat([anchors, positives, negatives], dim=0)
    return all_images, len(anchors), len(negatives) // len(anchors)
