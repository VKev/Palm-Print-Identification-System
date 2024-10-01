import os
import torchvision.utils as vutils
from .app import *
import re
import matplotlib.pyplot as plt
import torchvision.transforms as T
from .utils import *


def unnormalize_image(tensor, mean, std):
    """
    Unnormalize a tensor image back to [0, 1] range for saving.
    @tensor: The image tensor to unnormalize
    @mean: Mean used during normalization
    @std: Standard deviation used during normalization
    """
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)  # Reverse the normalization: t = t * std + mean
    return tensor


def process_images_in_folder(input_folder, output_folder):
    """
    Processes all images in a folder and subfolders that start with 'numberO' (e.g., '002O', '098O', etc.),
    saving preprocessed images.

    @input_folder: Folder containing the images to process
    @output_folder: Folder where processed images will be saved
    """
    # Regex pattern to match filenames starting with a number followed by 'O'
    pattern = re.compile(r".*")

    # Mean and standard deviation used during normalization
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    # Walk through input_folder and find all image files
    for root, dirs, files in os.walk(input_folder):
        for file in files:
            # Check if filename matches the 'numberO' pattern
            if pattern.match(file) and file.endswith(
                (".png", ".jpg", ".jpeg", ".bmp", ".JPG")
            ):  # Adjust extensions as needed
                img_path = os.path.join(root, file)

                # Create the output folder if it doesn't exist, keeping folder structure
                relative_path = os.path.relpath(root, input_folder)
                save_folder = os.path.join(output_folder, relative_path)
                os.makedirs(save_folder, exist_ok=True)

                # Predict and preprocess the image
                preprocessed_img = predictAndPreprocess(img_path).squeeze(
                    0
                )  # Remove batch dimension

                # Unnormalize the image to return it back to the [0, 1] range for saving
                unnormalized_img = unnormalize_image(preprocessed_img, mean, std)

                # Save the unnormalized preprocessed image
                save_path = os.path.join(
                    save_folder, file
                )  # Save with the same filename
                vutils.save_image(unnormalized_img, save_path)
                print(f"Processed and saved: {save_path}")


if __name__ == "__main__":
    preprocessed_img = process_images_in_folder(
        r"C:\My_Laptop\Repo\Palm-Print-Identification-System\AI_server\depthanythingv2\test",
        r"C:\My_Laptop\Repo\Palm-Print-Identification-System\AI_server\depthanythingv2\test",
    )
    # visualize_roi_data_agumentation(preprocessed_img)

    # img_to_display = preprocessed_img.squeeze().cpu()
    # img_to_display = img_to_display[0]
    # # If you want to unnormalize to display the image in its original scale:
    # unnormalize = T.Normalize(
    #     mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
    #     std=[1 / 0.229, 1 / 0.224, 1 / 0.225],
    # )
    # img_to_display = unnormalize(img_to_display)

    # # Convert the image tensor to a numpy array for displaying.
    # img_to_display = img_to_display.permute(1, 2, 0).numpy()

    # # Display the image using Matplotlib
    # plt.imshow(img_to_display)
    # plt.title("Preprocessed Image for CNN")
    # plt.axis("off")  # Optional: Turn off axis labels for better viewing
    # plt.show()

    # input_folder = r"C:\My_Laptop\Repo\Palm-Vein-Recognition-System\AI_Server\data\raw\multimodal pp and pv\DB_Print"
    # output_folder = (
    #     r"C:\My_Laptop\Repo\Palm-Vein-Recognition-System\AI_Server\data\raw\train"
    # )

    # process_images_in_folder(input_folder, output_folder)
