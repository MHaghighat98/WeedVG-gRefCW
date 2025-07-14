import os
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import resnet50
from torchvision.models.feature_extraction import create_feature_extractor
from transformers import BertModel, BertTokenizer
import kagglehub
import zipfile
import matplotlib.pyplot as plt
import matplotlib.patches as patches


# =======================
# Download Dataset
# =======================
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


# =======================
# Model Definition
# =======================
class VisualGroundingModel(nn.Module):
    """
    A simple visual grounding model that aligns image and text features
    and predicts a bounding box for the phrase within the image.
    """

    def __init__(self, image_dim=2048, text_dim=768, hidden_dim=512):
        super(VisualGroundingModel, self).__init__()

        # Load pretrained ResNet-50 and extract features from 'avgpool' layer
        backbone = resnet50(pretrained=True)
        self.image_encoder = create_feature_extractor(backbone, return_nodes={"avgpool": "features"})

        # Load pretrained BERT model and tokenizer for text encoding
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.text_encoder = BertModel.from_pretrained("bert-base-uncased")

        # Projection layers to map image and text embeddings to the same space
        self.fc_img = nn.Linear(image_dim, hidden_dim)
        self.fc_txt = nn.Linear(text_dim, hidden_dim)

        # Cross-attention mechanism between image and text embeddings
        self.cross_attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=8, batch_first=True)

        # Bounding box regression head
        self.bbox_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 4),  # Output: (x_min, y_min, x_max, y_max) - normalized
        )

    def forward(self, image, phrase):
        """
        Forward pass: encodes image and phrase, aligns them with attention,
        and predicts a bounding box for the referred object.
        """
        # Extract image features
        with torch.no_grad():
            img_feat = self.image_encoder(image)["features"].squeeze(-1).squeeze(-1)

        # Tokenize and encode text
        tokens = self.tokenizer(phrase, return_tensors="pt", padding=True, truncation=True)
        tokens = {k: v.to(image.device) for k, v in tokens.items()}
        txt_output = self.text_encoder(**tokens)
        txt_feat = txt_output.last_hidden_state[:, 0, :]  # Use [CLS] token

        # Project both image and text embeddings
        img_proj = self.fc_img(img_feat).unsqueeze(1)  # Shape: [B, 1, H]
        txt_proj = self.fc_txt(txt_feat).unsqueeze(1)  # Shape: [B, 1, H]

        # Cross-attention: text queries image features
        fused_feat, _ = self.cross_attn(query=txt_proj, key=img_proj, value=img_proj)

        # Predict bounding box
        pred_bbox = self.bbox_head(fused_feat.squeeze(1))  # Shape: [B, 4]
        return pred_bbox


# Checking


# =======================
# Visualization Utility
# =======================
def visualize_prediction(image_path, pred_bbox):
    """
    Visualize the predicted bounding box on the image.
    The coordinates are expected to be normalized in [0, 1].
    """
    image = Image.open(image_path).convert("RGB")
    width, height = image.size

    # Convert normalized bbox to pixel coordinates
    bbox = pred_bbox.squeeze().tolist()
    x_min = bbox[0] * width
    y_min = bbox[1] * height
    x_max = bbox[2] * width
    y_max = bbox[3] * height

    # Plot the image and bounding box
    fig, ax = plt.subplots(1)
    ax.imshow(image)

    rect = patches.Rectangle(
        (x_min, y_min),
        x_max - x_min,
        y_max - y_min,
        linewidth=2,
        edgecolor="r",
        facecolor="none",
    )
    ax.add_patch(rect)
    plt.title("Predicted Bounding Box")
    plt.axis("off")
    plt.show()


# =======================
# Main Pipeline
# =======================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
        raise FileNotFoundError(f"No image found in dataset directories: {possible_img_dirs}")

    print(f"Using example image: {example_img_path}")
    image = Image.open(example_img_path).convert("RGB")

    # Step 3: Preprocess image (resize + convert to tensor)
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ]
    )
    img_tensor = transform(image).unsqueeze(0)  # Shape: [1, 3, 224, 224]

    # Step 4: Initialize the model and run prediction
    model = VisualGroundingModel()
    model.eval()

    phrase = ["a weed in the field"]  # Natural language phrase to ground

    with torch.no_grad():
        pred_bbox = model(img_tensor, phrase)
    visualize_prediction(example_img_path, pred_bbox)
