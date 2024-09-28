import matplotlib.pyplot as plt
import torchvision.transforms as T


def visualize_roi_data_agumentation(imageBatch):
    """
    Visualizes a list of CNN-ready tensors.
    @imageBatch: List of CNN-ready image tensors
    """
    # Denormalization to reverse the normalization applied to the images
    denormalize = T.Normalize(
        mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
        std=[1 / 0.229, 1 / 0.224, 1 / 0.225],
    )

    # Create a 2x3 grid for the images
    fig, axs = plt.subplots(2, 3, figsize=(10, 7))

    for i, img_tensor in enumerate(imageBatch):
        # Apply denormalization
        img_tensor = denormalize(img_tensor.squeeze().cpu())

        # Convert back to numpy image
        img = T.ToPILImage()(img_tensor).convert("L")

        # Plot each image in the grid
        ax = axs[i // 3, i % 3]
        ax.imshow(img, cmap="gray")
        ax.axis("off")  # Hide axis labels

    # Set the titles for each subplot
    titles = [
        "Original Image (No color shift)",
        "Color Shift",
        "Color Shift",
        "rotation",
        "rotation",
        "Cut-out",
    ]

    for i, ax in enumerate(axs.flat):
        ax.set_title(titles[i])

    # Adjust layout and display
    plt.tight_layout()
    plt.show()
