import os
from PIL import Image


def resize_images(input_dir, output_dir, size=(224, 224)):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.endswith(("png", "jpg", "jpeg", "bmp")):
                img_path = os.path.join(root, file)
                img = Image.open(img_path)
                img_resized = img.resize(size)

                # Create corresponding subdirectory in output folder
                rel_path = os.path.relpath(root, input_dir)
                out_subdir = os.path.join(output_dir, rel_path)
                if not os.path.exists(out_subdir):
                    os.makedirs(out_subdir)

                output_path = os.path.join(out_subdir, file)
                img_resized.save(output_path)


# Example usage
input_folder = r"C:\My_Laptop\Repo\Palm-Print-Identification-System\AI_server\mambavision\raw\Palmprint Image Dataset with Gabor Filter Feature Enchancement\raw"  # Change to your input folder path
output_folder = r"C:\My_Laptop\Repo\Palm-Print-Identification-System\AI_server\mambavision\raw\Palmprint Image Dataset with Gabor Filter Feature Enchancement\raw"  # Change to your output folder path
resize_images(input_folder, output_folder)

# Example usage
