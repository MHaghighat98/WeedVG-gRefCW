import os
import csv
import json
import cv2
from tqdm import tqdm

# Set paths (update these if needed)
BBOXES_DIR = "C:/Users/jd138001/Downloads/data/bboxes/CropOrWeed2Eval"  # Source bounding-box CSVs
IMAGES_DIR = "C:/Users/jd138001/Downloads/data/images"  # Source images
OUTPUT_JSON = "../data/annotations.json"  # Output annotation file

PARAMS_DIR = "C:/Users/jd138001/Downloads/data/params"


def get_image_info(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return None, None, None
    height, width = img.shape[:2]
    return width, height, os.path.basename(image_path)


annotations = {"images": []}

# Get all image filenames (without extension)
all_image_files = sorted([f for f in os.listdir(IMAGES_DIR) if f.lower().endswith(".jpg")])
all_image_ids = [os.path.splitext(f)[0] for f in all_image_files]

csv_files = set(os.path.splitext(f)[0] for f in os.listdir(BBOXES_DIR) if f.lower().endswith(".csv"))

for image_id, file_name in tqdm(list(zip(all_image_ids, all_image_files)), desc="Processing images"):
    # Read params file for this image (CSV)
    params_path = os.path.join(PARAMS_DIR, image_id + ".csv")
    moisture = soil = lighting = separability = None
    if os.path.exists(params_path):
        with open(params_path, "r", encoding="utf-8") as pf:
            reader = csv.DictReader(pf)
            for row in reader:
                moisture = row.get("moisture")
                soil = row.get("soil")
                lighting = row.get("lighting")
                separability = row.get("separability")
                break
    image_path = os.path.join(IMAGES_DIR, file_name)
    width, height, _ = get_image_info(image_path)
    if width is None:
        continue
    instances = []
    csv_path = os.path.join(BBOXES_DIR, image_id + ".csv")
    if image_id in csv_files:
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f, fieldnames=["left", "top", "right", "bottom", "label_id", "stem_x", "stem_y"])
            for idx, row in enumerate(reader, 1):
                label_id = int(row["label_id"])
                x = int(row["left"])
                y = int(row["top"])
                w = int(row["right"]) - x
                h = int(row["bottom"]) - y
                # Map label_id to category name, skip unknown
                if label_id == 0:
                    category = "crop"
                elif label_id == 1:
                    category = "weed"
                else:
                    continue  # Skip unknown category
                instances.append({"instance_id": str(idx), "category_id": category, "bbox": [x, y, w, h]})
    annotations["images"].append(
        {
            "id": image_id,
            "file_name": file_name,
            "width": width,
            "height": height,
            "moisture": moisture,
            "soil": soil,
            "lighting": lighting,
            "separability": separability,
            "instances": instances,
            "expression": "",
            "negative_sentence": "",
        }
    )

with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
    json.dump(annotations, f, indent=2)
