from torchvision import transforms
import os
from PIL import Image
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


class ImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []

        # Walk through the directory and store image paths
        for root, _, files in os.walk(root_dir):
            for file in files:
                if file.endswith(
                    (".png", ".jpg", ".jpeg", ".bmp")
                ):  # Modify extensions as needed
                    self.image_paths.append(os.path.join(root, file))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image


transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4440], std=[0.2104]),
    ]
)

augmentation = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.RandomRotation(20),
        transforms.RandomApply(
            [
                transforms.ColorJitter(
                    brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2
                )
            ],
            p=0.5,
        ),
        transforms.RandomApply([transforms.RandomHorizontalFlip()], p=0.5),
        transforms.RandomApply(
            [transforms.RandomResizedCrop(size=224, scale=(0.8, 1.0))], p=0.5
        ),
        transforms.RandomApply(
            [transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5))], p=0.4
        ),
        transforms.RandomApply(
            [transforms.RandomAffine(degrees=0, translate=(0.1, 0.1))], p=0.7
        ),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.4440], std=[0.2104]),
    ]
)


def augment_images(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Loop over all files and directories in the input path
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.endswith(
                (".png", ".jpg", ".jpeg", ".bmp", ".JPG")
            ):  # Modify extensions as needed
                # Full file path
                file_path = os.path.join(root, file)

                # Open the image
                img = Image.open(file_path).convert("RGB")

                # Apply augmentation
                augmented_img = augmentation(img)

                # Convert back to PIL image for saving
                augmented_img_pil = transforms.ToPILImage()(augmented_img)

                # Construct the output file path
                relative_path = os.path.relpath(root, input_dir)
                save_folder = os.path.join(output_dir, relative_path)

                if not os.path.exists(save_folder):
                    os.makedirs(save_folder)

                output_path = os.path.join(save_folder, "a_" + file)

                # Save the augmented image
                augmented_img_pil.save(output_path)
                print(f"Saved {output_path}")


def calculate_mean_std(loader):
    mean = 0.0
    std = 0.0
    total_images_count = 0

    # Add tqdm for the progress bar
    for images in tqdm(loader, desc="Processing Images"):
        batch_images_count = images.size(0)
        total_images_count += batch_images_count

        # Calculate mean and std for this batch
        mean += images.mean([0, 2, 3]) * batch_images_count
        std += images.std([0, 2, 3]) * batch_images_count

    # Calculate overall mean and std
    mean /= total_images_count
    std /= total_images_count

    return mean, std


def process_images(input_dir, batch_size=32):
    # Create dataset and data loader
    dataset = ImageDataset(root_dir=input_dir, transform=augmentation)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # Calculate mean and std
    mean, std = calculate_mean_std(loader)

    return mean, std


if __name__ == "__main__":
    # input_directory = r"C:\My_Laptop\Repo\Palm-Print-Identification-System\AI_server\mambavision\raw\train"
    # mean, std = process_images(input_directory)
    # print(f"Mean: {mean}")
    # print(f"Std: {std}")

    input_directory = r"raw/1"
    output_directory = r"raw/2"

    augment_images(input_directory, output_directory)
