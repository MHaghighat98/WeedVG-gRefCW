import os
import zipfile
import kagglehub
from PIL import Image


def download_weed_detection(dest_dir="WeedDetection"):
    """
    Downloads the weed detection dataset using KaggleHub if not already present.
    Unzips if necessary and returns the path to the dataset.
    """
    if os.path.exists(dest_dir):
        print(f"{dest_dir} already exists, skipping download.")
        return dest_dir

    print("Downloading Weed Detection dataset using kagglehub...")
    path = kagglehub.dataset_download("jaidalmotra/weed-detection")

    if os.path.isdir(path):
        # If download is already a directory
        if path != dest_dir:
            os.rename(path, dest_dir)
    else:
        # If the download is a zip file, extract it
        with zipfile.ZipFile(path, "r") as zip_ref:
            zip_ref.extractall(dest_dir)
        os.remove(path)

    print(f"Dataset downloaded and extracted to {dest_dir}")
    return dest_dir


if __name__ == "__main__":
    # Step 1: Download dataset (if not already downloaded)
    dataset_dir = download_weed_detection(dest_dir="WeedDetection")

    # Step 2: Try to find an example image from typical subfolders
    possible_img_dirs = [
        os.path.join(dataset_dir, "images"),
        os.path.join(dataset_dir, "train"),
        os.path.join(dataset_dir, "test"),
        dataset_dir,
    ]

    example_img_path = None
    for d in possible_img_dirs:
        if os.path.exists(d) and os.path.isdir(d):
            files = [f for f in os.listdir(d) if f.lower().endswith((".jpg", ".png"))]
            if files:
                example_img_path = os.path.join(d, files[0])
                break

    if not example_img_path or not os.path.exists(example_img_path):
        raise FileNotFoundError(
            f"No image found in dataset directories: {possible_img_dirs}"
        )

    print(f"Using example image: {example_img_path}")
    image = Image.open(example_img_path).convert("RGB")
