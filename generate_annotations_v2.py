import os
import csv
import json
import cv2
import numpy as np
from tqdm import tqdm
import random
from collections import Counter
import sys

sys.path.append(os.path.dirname(__file__))
from utilities.datasets import DATASETS

# ------------------ INPUT/OUTPUT paths ------------------
IMAGES_DIR = "C:/Users/jd138001/Downloads/data/images"
BBOXES_DIR = "C:/Users/jd138001/Downloads/data/bboxes/CropAndWeed"
LABELIDS_DIR = "C:/Users/jd138001/Downloads/data/labelIds/CropAndWeed"
OUTPUT_DIR = "C:/Users/jd138001/Downloads/data/grefcoco_format"
os.makedirs(OUTPUT_DIR, exist_ok=True)

random.seed(42)

# Load CropsOrWeed9 dataset mapping
dataset = DATASETS["CropsOrWeed9"]
# also keep the detailed original dataset mapping to recover original label names
original_dataset = DATASETS.get("CropAndWeed")


# ------------------ Helper functions ------------------
def get_image_info(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return None, None
    height, width = img.shape[:2]
    return width, height


def load_segmentation_mask(mask_path, image_width, image_height):
    if not os.path.exists(mask_path):
        return None
    mask_img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask_img is None:
        return None
    if mask_img.shape != (image_height, image_width):
        mask_img = cv2.resize(mask_img, (image_width, image_height), interpolation=cv2.INTER_NEAREST)
    return mask_img


def mask_to_polygon(mask):
    try:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        polygons = []
        for contour in contours:
            if len(contour) >= 3:
                polygon = contour.reshape(-1, 2).flatten().tolist()
                if len(polygon) >= 6:
                    polygons.append(polygon)
        return polygons if polygons else [[0, 0, 1, 0, 1, 1, 0, 1]]
    except Exception:
        return [[0, 0, 1, 0, 1, 1, 0, 1]]


def extract_instance_segmentation(full_mask, bbox, label_id):
    if full_mask is None:
        return None
    x, y, w, h = bbox
    roi = full_mask[y : y + h, x : x + w]
    instance_mask = (roi == label_id).astype(np.uint8)
    full_instance_mask = np.zeros((full_mask.shape[0], full_mask.shape[1]), dtype=np.uint8)
    full_instance_mask[y : y + h, x : x + w] = instance_mask
    polygons = mask_to_polygon(full_instance_mask)
    return polygons


def generate_single_referring_expression(category, crop_name, bbox, image_width, image_height, test_change=None, include_crop_name=True):
    x, y, w, h = bbox
    center_x, center_y = x + w / 2, y + h / 2

    # Position
    pos_x = "left" if center_x < image_width / 3 else "right" if center_x > 2 * image_width / 3 else "center"
    pos_y = "top" if center_y < image_height / 3 else "bottom" if center_y > 2 * image_height / 3 else "middle"
    position_val = f"{pos_y} {pos_x}"

    # Size
    area = w * h
    size = "tiny" if area < 2089 else "small" if area < 20890 else "medium" if area < 208896 else "large"

    # Build base sentence, optionally including the crop name
    if category == "crop":
        if include_crop_name and crop_name:
            sentence = f"{size} {crop_name} {category} in the {position_val}"
        else:
            sentence = f"{size} {category} in the {position_val}"
    else:
        sentence = f"{size} {category} in the {position_val}"

    change_type = None
    change_detail = None

    # Apply test changes
    if test_change:
        if test_change.get("method") == "replace" and test_change.get("attr") == "category":
            # Flip category only
            old_category = category
            new_category = "weed" if category == "crop" else "crop"
            if new_category == "crop":
                if include_crop_name and crop_name:
                    sentence = f"{size} {crop_name} {new_category} in the {position_val}"
                else:
                    sentence = f"{size} {new_category} in the {position_val}"
            else:
                sentence = f"{size} {new_category} in the {position_val}"
            change_type = "replace"
            change_detail = {"attribute": "category", "from": old_category, "to": new_category}
        elif test_change.get("method") == "swap" and test_change.get("swap_type") == "size_swap":
            # Random size swap
            sizes = ["tiny", "small", "medium", "large"]
            old_size = size
            possible_sizes = [s for s in sizes if s != old_size]
            new_size = random.choice(possible_sizes) if possible_sizes else old_size
            if category == "crop":
                if include_crop_name and crop_name:
                    sentence = f"{new_size} {crop_name} {category} in the {position_val}"
                else:
                    sentence = f"{new_size} {category} in the {position_val}"
            else:
                sentence = f"{new_size} {category} in the {position_val}"
            change_type = "swap"
            change_detail = {"attribute": "size", "from": old_size, "to": new_size}
        elif test_change.get("method") == "swap" and test_change.get("swap_type") == "position_swap":
            # Random position swap
            positions = [
                "top left",
                "top center",
                "top right",
                "middle left",
                "middle center",
                "middle right",
                "bottom left",
                "bottom center",
                "bottom right",
            ]
            old_position = position_val
            possible_positions = [p for p in positions if p != old_position]
            new_position = random.choice(possible_positions) if possible_positions else old_position
            if category == "crop":
                if include_crop_name and crop_name:
                    sentence = f"{size} {crop_name} {category} in the {new_position}"
                else:
                    sentence = f"{size} {category} in the {new_position}"
            else:
                sentence = f"{size} {category} in the {new_position}"
            change_type = "swap"
            change_detail = {"attribute": "position", "from": old_position, "to": new_position}

    return sentence, change_type, change_detail


def generate_image_level_expression(crop_count, weed_count):
    if crop_count == 0 and weed_count == 0:
        return "no crops or weeds are visible in this image"
    if crop_count > 0 and weed_count == 0:
        return "no weeds are present in this image"
    if crop_count == 0 and weed_count > 0:
        return "no crops are present in this image"
    return None


# ------------------ Main Function ------------------
def main():
    print("Generating gRefCOCO format dataset...")

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
    image_id = 1
    ann_id = 1
    cat_name_to_id = {"crop": 1, "weed": 2}

    all_image_files = sorted([f for f in os.listdir(IMAGES_DIR) if f.lower().endswith((".jpg", ".jpeg", ".png"))])
    print(f"Found {len(all_image_files)} images")

    # Categorize images
    image_categories = {"no_target": [], "crop_single": [], "weed_single": [], "crop_multi": [], "weed_multi": [], "mixed": []}
    for file_name in tqdm(all_image_files, desc="Categorizing images"):
        image_id_str = os.path.splitext(file_name)[0]
        bbox_csv_path = os.path.join(BBOXES_DIR, image_id_str + ".csv")
        crop_count = 0
        weed_count = 0
        if os.path.exists(bbox_csv_path):
            with open(bbox_csv_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f, fieldnames=["left", "top", "right", "bottom", "label_id", "stem_x", "stem_y"])
                for row in reader:
                    try:
                        label_id = int(row["label_id"])
                        mapped_id = dataset.get_mapped_id(label_id)
                        if mapped_id is not None and mapped_id <= 7:  # Crops
                            crop_count += 1
                        elif mapped_id == 8:  # Weed
                            weed_count += 1
                    except Exception:
                        continue
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

    # Stratified split
    def stratified_split(image_list, train_ratio=0.7, val_ratio=0.15):
        if not image_list:
            return [], [], []
        random.shuffle(image_list)
        n_train = int(len(image_list) * train_ratio)
        n_val = int(len(image_list) * val_ratio)
        return image_list[:n_train], image_list[n_train : n_train + n_val], image_list[n_train + n_val :]

    splits = {"train": [], "val": [], "test": []}
    for cat, imgs in image_categories.items():
        t, v, te = stratified_split(imgs)
        splits["train"].extend(t)
        splits["val"].extend(v)
        splits["test"].extend(te)
    for split in splits.values():
        random.shuffle(split)

    print(f"Stratified splits - Train: {len(splits['train'])}, Val: {len(splits['val'])}, Test: {len(splits['test'])}")

    # Process images and build annotations
    for split_name, image_files in splits.items():
        for file_name in tqdm(image_files, desc=f"Processing {split_name} images"):
            image_id_str = os.path.splitext(file_name)[0]
            image_path = os.path.join(IMAGES_DIR, file_name)
            width, height = get_image_info(image_path)
            if width is None:
                continue

            bbox_csv_path = os.path.join(BBOXES_DIR, image_id_str + ".csv")
            bboxes, categories, label_ids, crop_names = [], [], [], []
            crop_count, weed_count = 0, 0

            if os.path.exists(bbox_csv_path):
                with open(bbox_csv_path, "r", encoding="utf-8") as f:
                    reader = csv.DictReader(f, fieldnames=["left", "top", "right", "bottom", "label_id", "stem_x", "stem_y"])
                    for row in reader:
                        try:
                            original_label_id = int(row["label_id"])
                            mapped_id = dataset.get_mapped_id(original_label_id)
                            if mapped_id is None:
                                continue
                            x, y, w, h = int(row["left"]), int(row["top"]), int(row["right"]) - int(row["left"]), int(row["bottom"]) - int(row["top"])

                            if mapped_id <= 7:  # Crop
                                category = "crop"
                                crop_count += 1
                                crop_name = dataset.get_label_name(mapped_id)
                            elif mapped_id == 8:  # Weed
                                category = "weed"
                                weed_count += 1
                                crop_name = "weed"
                            else:
                                continue

                            bboxes.append([x, y, w, h])
                            categories.append(category)
                            label_ids.append(original_label_id)
                            crop_names.append(crop_name)
                        except Exception:
                            continue

            mask_path = os.path.join(LABELIDS_DIR, image_id_str + ".png")
            full_mask = load_segmentation_mask(mask_path, width, height)

            # Instance annotations
            for i, (bbox, category, label_id, crop_name) in enumerate(zip(bboxes, categories, label_ids, crop_names)):
                segmentation = extract_instance_segmentation(full_mask, bbox, label_id)
                # determine mapped id from the source label id and store it so later sentence generation
                mapped_id = dataset.get_mapped_id(label_id)
                instances_data["annotations"].append(
                    {
                        "id": ann_id,
                        "image_id": image_id,
                        "category_id": cat_name_to_id[category],
                        "bbox": bbox,
                        "segmentation": segmentation,
                        "area": bbox[2] * bbox[3],
                        "iscrowd": 0,
                        "crop_name": crop_name,
                        "original_label_id": label_id,
                        "mapped_id": mapped_id,
                    }
                )
                ann_id += 1

            # Image entry
            instances_data["images"].append(
                {
                    "id": image_id,
                    "file_name": file_name,
                    "width": width,
                    "height": height,
                    "date_captured": "2025-09-27 00:00:00",
                    "split": split_name,
                }
            )

            # Image-level refs
            negative_expression = generate_image_level_expression(crop_count, weed_count)
            ref = {"file_name": file_name, "image_id": image_id, "split": split_name}
            if split_name in ("train", "val") and negative_expression:
                ref["negative_sentence"] = negative_expression
            refs_data.append(ref)

            image_id += 1

    # Save instances.json
    instances_path = os.path.join(OUTPUT_DIR, "instances.json")
    with open(instances_path, "w", encoding="utf-8") as f:
        json.dump(instances_data, f, indent=2)

    # ------------------ Plan perfectly balanced test set changes ------------------
    print("\nPlanning perfectly balanced test set changes...")
    test_ann_ids = [ann["id"] for ann in instances_data["annotations"] if instances_data["images"][ann["image_id"] - 1]["split"] == "test"]
    random.shuffle(test_ann_ids)
    n = len(test_ann_ids)
    n_each = n // 3
    test_changes_by_ann_id = {}

    for idx, ann_id_val in enumerate(test_ann_ids):
        if idx < n_each:
            test_changes_by_ann_id[ann_id_val] = {"method": "replace", "attr": "category"}
        elif idx < 2 * n_each:
            test_changes_by_ann_id[ann_id_val] = {"method": "swap", "swap_type": "size_swap"}
        else:
            test_changes_by_ann_id[ann_id_val] = {"method": "swap", "swap_type": "position_swap"}

    for ann_id_val in test_ann_ids[3 * n_each :]:
        test_changes_by_ann_id[ann_id_val] = {"method": "replace", "attr": "category"}

    dist = Counter([v["attr"] if v["method"] == "replace" else v["swap_type"] for v in test_changes_by_ann_id.values()])
    print(f"Total test instances: {n}")
    print(f"Change distribution: {dict(dist)}")

    # ------------------ Build final grefs(unc).json ------------------
    grefs_path = os.path.join(OUTPUT_DIR, "grefs(unc).json")
    grefs_data = {"info": instances_data["info"], "images": []}

    for img in instances_data["images"]:
        image_entry = img.copy()
        ref = next((r for r in refs_data if r["image_id"] == img["id"] and "negative_sentence" in r), None)
        if ref and img["split"] in ("train", "val"):
            image_entry["negative_sentence"] = ref["negative_sentence"]

        instance_sentences = []
        for ann in instances_data["annotations"]:
            if ann["image_id"] != img["id"]:
                continue

            # Prefer the original (fine-grained) crop name from the original
            # label id when available. Fall back to the CropsOrWeed9 mapping.
            mapped_id = ann.get("mapped_id")
            orig_id = ann.get("original_label_id")
            orig_name = None
            if original_dataset is not None and orig_id is not None:
                orig_name = original_dataset.get_label_name(orig_id)

            if mapped_id is not None:
                # category follows the CropsOrWeed9 mapping
                category = "crop" if mapped_id <= 7 else "weed"
                # use the CropsOrWeed9 label name for crops; for weed use 'weed'
                if category == "crop":
                    crop_name = dataset.get_label_name(mapped_id) or ann.get("crop_name")
                else:
                    crop_name = "weed"
            else:
                category = "crop" if ann["category_id"] == 1 else "weed"
                # if we have an original name prefer it
                crop_name = orig_name or ann.get("crop_name", category)
            bbox = ann["bbox"]

            if img["split"] != "test":
                sentence, _, _ = generate_single_referring_expression(category, crop_name, bbox, img["width"], img["height"])
                instance_sentences.append({"ann_id": ann["id"], "category_id": ann["category_id"], "sentence": sentence})
            else:
                tc = test_changes_by_ann_id.get(ann["id"])
                # Generate original sentence (without changes)
                original_sentence, _, _ = generate_single_referring_expression(
                    category, crop_name, bbox, img["width"], img["height"], test_change=None, include_crop_name=False
                )
                # Generate modified sentence (with changes)
                modified_sentence, change_type, change_detail = generate_single_referring_expression(
                    category, crop_name, bbox, img["width"], img["height"], test_change=tc, include_crop_name=False
                )
                instance_sentences.append(
                    {
                        "ann_id": ann["id"],
                        "category_id": ann["category_id"],
                        "original_sentence": original_sentence,
                        "test_sentence": modified_sentence,
                        "change_type": change_type,
                        "change_detail": change_detail,
                    }
                )

        if instance_sentences:
            image_entry["instance_sentences"] = instance_sentences
        grefs_data["images"].append(image_entry)

    with open(grefs_path, "w", encoding="utf-8") as f:
        json.dump(grefs_data, f, indent=4)

    print("✅ grefs(unc).json generated with crop names and balanced test changes.")
    print("Dataset generation complete!")


if __name__ == "__main__":
    main()
