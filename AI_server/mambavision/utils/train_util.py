import os
import re


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
