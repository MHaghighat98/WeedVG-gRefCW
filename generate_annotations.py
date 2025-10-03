import os
import csv
import json
import cv2
import numpy as np
from tqdm import tqdm
import random
from collections import defaultdict

# INPUT/OUTPUT paths
IMAGES_DIR = "C:/Users/jd138001/Downloads/data/images"
BBOXES_DIR = "C:/Users/jd138001/Downloads/data/bboxes/CropOrWeed2Eval"
LABELIDS_DIR = "C:/Users/jd138001/Downloads/data/labelIds/CropOrWeed2"
OUTPUT_DIR = "C:/Users/jd138001/Downloads/data/grefcoco_format"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def get_image_info(image_path):
    """Get image dimensions"""
    img = cv2.imread(image_path)
    if img is None:
        return None, None
    height, width = img.shape[:2]
    return width, height


def load_segmentation_mask(mask_path, image_width, image_height):
    """Load segmentation mask and resize if necessary"""
    if not os.path.exists(mask_path):
        return None
    mask_img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask_img is None:
        return None
    if mask_img.shape != (image_height, image_width):
        mask_img = cv2.resize(mask_img, (image_width, image_height), interpolation=cv2.INTER_NEAREST)
    return mask_img


def mask_to_polygon(mask):
    """Convert binary mask to polygon format (COCO style)"""
    try:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        polygons = []
        for contour in contours:
            if len(contour) >= 3:
                polygon = contour.reshape(-1, 2).flatten().tolist()
                if len(polygon) >= 6:  # At least 3 points
                    polygons.append(polygon)
        return polygons if polygons else [[0, 0, 1, 0, 1, 1, 0, 1]]  # Dummy polygon if empty
    except Exception:
        return [[0, 0, 1, 0, 1, 1, 0, 1]]  # Dummy polygon on error


def extract_instance_segmentation(full_mask, bbox, label_id):
    """Extract instance segmentation from full mask using bbox and label_id"""
    if full_mask is None:
        return None

    x, y, w, h = bbox
    # Extract region of interest
    roi = full_mask[y : y + h, x : x + w]

    # Create binary mask for this specific instance
    instance_mask = (roi == label_id).astype(np.uint8)

    # Create full-size mask
    full_instance_mask = np.zeros((full_mask.shape[0], full_mask.shape[1]), dtype=np.uint8)
    full_instance_mask[y : y + h, x : x + w] = instance_mask

    # Convert to polygon format
    polygons = mask_to_polygon(full_instance_mask)

    return polygons


def generate_single_referring_expression(category, bbox, image_width, image_height, crop_count, weed_count, instance_idx=0):
    """Generate a single referring expression with category, location (both pos_x and pos_y), and size (no pixel dimensions)"""
    x, y, w, h = bbox
    center_x = x + w / 2
    center_y = y + h / 2

    # Position descriptions
    if center_x < image_width / 3:
        pos_x = "left"
    elif center_x > 2 * image_width / 3:
        pos_x = "right"
    else:
        pos_x = "center"

    if center_y < image_height / 3:
        pos_y = "top"
    elif center_y > 2 * image_height / 3:
        pos_y = "bottom"
    else:
        pos_y = "middle"

    # Size descriptions based on area
    area = w * h
    if area < 1000:
        size = "small"
    elif area < 5000:
        size = "medium"
    else:
        size = "large"

    # Compose position phrase using both pos_x and pos_y
    position = f"in the {pos_y} {pos_x}"

    # Generate expression with category, location, and size (no pixel dimensions)
    expression = f"{size} {category} {position}"

    return expression
    
def generate_image_level_expression(crop_count, weed_count):
    """Describe which categories are missing (if any) for an image"""
    if crop_count == 0 and weed_count == 0:
        return "no crops or weeds are visible in this image"
    if crop_count > 0 and weed_count == 0:
        return "no weeds are present in this image"
    if crop_count == 0 and weed_count > 0:
        return "no crops are present in this image"
    return None  # Return None when no categories are missing

def main():
    print("Generating gRefCOCO format dataset...")

    # Initialize gRefCOCO structures
    instances_data = {
        "info": {
            "description": "CropAndWeed Dataset in gRefCOCO format",
            "version": "1.0",
            "year": 2025,
            "contributor": "CropAndWeed",
            "date_created": "2025-09-27",
        },
        "images": [],
        "annotations": [],
        "categories": [{"id": 1, "name": "crop", "supercategory": "plant"}, {"id": 2, "name": "weed", "supercategory": "plant"}],
    }

    refs_data = []
    negative_refs_data = []

    # Counters
    image_id = 1
    ann_id = 1
    ref_id = 0  # Start from 0 like gRefCOCO

    # Category mapping
    cat_name_to_id = {"crop": 1, "weed": 2}

    # Get all image files
    all_image_files = sorted([f for f in os.listdir(IMAGES_DIR) if f.lower().endswith((".jpg", ".jpeg", ".png"))])
    print(f"Found {len(all_image_files)} images")

    # Analyze and categorize images for stratified splitting
    print("Analyzing image contents for stratified splitting...")
    image_categories = {"no_target": [], "crop_single": [], "weed_single": [], "crop_multi": [], "weed_multi": [], "mixed": []}

    for file_name in tqdm(all_image_files, desc="Categorizing images"):
        image_id_str = os.path.splitext(file_name)[0]
        bbox_csv_path = os.path.join(BBOXES_DIR, image_id_str + ".csv")

        crop_count = 0
        weed_count = 0

        if os.path.exists(bbox_csv_path):
            try:
                with open(bbox_csv_path, "r", encoding="utf-8") as f:
                    reader = csv.DictReader(f, fieldnames=["left", "top", "right", "bottom", "label_id", "stem_x", "stem_y"])
                    for row in reader:
                        try:
                            label_id = int(row["label_id"])
                            if label_id == 0:  # crop
                                crop_count += 1
                            elif label_id == 1:  # weed
                                weed_count += 1
                        except (ValueError, TypeError):
                            continue
            except Exception:
                pass

        # Categorize
        if crop_count == 0 and weed_count == 0:
            image_categories["no_target"].append(file_name)
        elif crop_count > 0 and weed_count > 0:
            image_categories["mixed"].append(file_name)
        elif crop_count == 1 and weed_count == 0:
            image_categories["crop_single"].append(file_name)
        elif crop_count > 1 and weed_count == 0:
            image_categories["crop_multi"].append(file_name)
        elif crop_count == 0 and weed_count == 1:
            image_categories["weed_single"].append(file_name)
        elif crop_count == 0 and weed_count > 1:
            image_categories["weed_multi"].append(file_name)

    # Stratified splitting
    def stratified_split(image_list, train_ratio=0.7, val_ratio=0.15):
        if not image_list:
            return [], [], []
        random.shuffle(image_list)
        n_train = int(len(image_list) * train_ratio)
        n_val = int(len(image_list) * val_ratio)
        return (image_list[:n_train], image_list[n_train : n_train + n_val], image_list[n_train + n_val :])

    # Create stratified splits
    splits = {"train": [], "val": [], "test": []}
    for category, images in image_categories.items():
        train_imgs, val_imgs, test_imgs = stratified_split(images)
        splits["train"].extend(train_imgs)
        splits["val"].extend(val_imgs)
        splits["test"].extend(test_imgs)

    for split in splits.values():
        random.shuffle(split)

    print(f"Stratified splits - Train: {len(splits['train'])}, Val: {len(splits['val'])}, Test: {len(splits['test'])}")

    # Process images by split
    for split_name, image_files in splits.items():
        for file_name in tqdm(image_files, desc=f"Processing {split_name} images"):
            image_id_str = os.path.splitext(file_name)[0]
            image_path = os.path.join(IMAGES_DIR, file_name)

            # Get image info
            width, height = get_image_info(image_path)
            if width is None:
                continue

            # Load bounding boxes and count categories
            bbox_csv_path = os.path.join(BBOXES_DIR, image_id_str + ".csv")
            bboxes = []
            categories = []
            label_ids = []
            crop_count = 0
            weed_count = 0

            if os.path.exists(bbox_csv_path):
                try:
                    with open(bbox_csv_path, "r", encoding="utf-8") as f:
                        reader = csv.DictReader(f, fieldnames=["left", "top", "right", "bottom", "label_id", "stem_x", "stem_y"])
                        for row in reader:
                            try:
                                label_id = int(row["label_id"])
                                x = int(row["left"])
                                y = int(row["top"])
                                w = int(row["right"]) - x
                                h = int(row["bottom"]) - y

                                if label_id == 0:
                                    category = "crop"
                                    crop_count += 1
                                elif label_id == 1:
                                    category = "weed"
                                    weed_count += 1
                                else:
                                    continue

                                bboxes.append([x, y, w, h])
                                categories.append(category)
                                label_ids.append(label_id)

                            except (ValueError, TypeError):
                                continue
                except Exception:
                    pass

            # Load segmentation mask
            mask_path = os.path.join(LABELIDS_DIR, image_id_str + ".png")
            full_mask = load_segmentation_mask(mask_path, width, height)

            # Add instance-level annotation to instances.json
            for i, (bbox, category, label_id) in enumerate(zip(bboxes, categories, label_ids)):
                category_id = cat_name_to_id[category]
                segmentation = extract_instance_segmentation(full_mask, bbox, label_id)

                instance_ann = {
                    "id": ann_id,
                    "image_id": image_id,
                    "category_id": category_id,
                    "bbox": bbox,
                    "segmentation": segmentation,
                    "area": bbox[2] * bbox[3],
                    "iscrowd": 0,
                }
                instances_data["annotations"].append(instance_ann)
                ann_id += 1

            # Add image-level reference to grefs(unc).json
            image_ref = {
                "file_name": file_name,
                "image_id": image_id,
                "split": split_name,
            }

            # For train images, add negative sentence if applicable
            if split_name == "train":
                negative_expression = generate_image_level_expression(crop_count, weed_count)
                if negative_expression:
                    image_ref["negative_sentence"] = negative_expression

            refs_data.append(image_ref)

            # Add image to instances_data["images"]
            image_entry = {
                "id": image_id,
                "file_name": file_name,
                "width": width,
                "height": height,
                "date_captured": "2025-09-27 00:00:00",
                "split": split_name,  # <-- ADD THIS LINE
            }
            instances_data["images"].append(image_entry)

            image_id += 1

    # Prepare grefs(unc).json structure
    grefs_data = {"info": instances_data["info"], "images": []}

    for img in instances_data["images"]:
        image_entry = img.copy()
        # Find the corresponding image_ref for negative_sentence (if any)
        ref = next((r for r in refs_data if r["image_id"] == img["id"]), None)
        if ref and "negative_sentence" in ref:
            image_entry["negative_sentence"] = ref["negative_sentence"]

        # Add instance-level sentences for this image
        image_id = img["id"]
        instance_sentences = []
        for ann in instances_data["annotations"]:
            if ann["image_id"] == image_id:
                category = "crop" if ann["category_id"] == 1 else "weed"
                bbox = ann["bbox"]
                width = img["width"]
                height = img["height"]
                sentence_text = generate_single_referring_expression(category, bbox, width, height, 0, 0)
                instance_sentences.append({"ann_id": ann["id"], "category_id": ann["category_id"], "sentence": sentence_text})
        if instance_sentences:
            image_entry["instance_sentences"] = instance_sentences

        grefs_data["images"].append(image_entry)

    # Save grefs(unc).json (image-level metadata + negative sentences + instance sentences)
    grefs_path = os.path.join(OUTPUT_DIR, "grefs(unc).json")
    with open(grefs_path, "w", encoding="utf-8") as f:
        json.dump(grefs_data, f, indent=2)

    # Remove image-level info from instances.json
    instances_data.pop("images", None)

    # Save instances.json (COCO format, instance-level only)
    instances_path = os.path.join(OUTPUT_DIR, "instances.json")
    with open(instances_path, "w", encoding="utf-8") as f:
        json.dump(instances_data, f, indent=2)

    print("\nDataset generation complete!")
    print(f"Images: {len(grefs_data['images'])}")
    print(f"Instance Annotations: {len(instances_data['annotations'])}")
    print(f"Files saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

