import os
import csv
import json
import cv2
import numpy as np
from tqdm import tqdm
import random
from collections import Counter, defaultdict

# ------------------ INPUT/OUTPUT paths ------------------
IMAGES_DIR = "C:/Users/jd138001/Downloads/data/images"
BBOXES_DIR = "C:/Users/jd138001/Downloads/data/bboxes/CropOrWeed2Eval"
LABELIDS_DIR = "C:/Users/jd138001/Downloads/data/labelIds/CropOrWeed2"
OUTPUT_DIR = "C:/Users/jd138001/Downloads/data/grefcoco_format"
os.makedirs(OUTPUT_DIR, exist_ok=True)

random.seed(42)


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


def generate_single_referring_expression(category, bbox, image_width, image_height, crop_count, weed_count, instance_idx=0, test_change=None):
    x, y, w, h = bbox
    center_x, center_y = x + w / 2, y + h / 2

    # Position
    pos_x = "left" if center_x < image_width / 3 else "right" if center_x > 2 * image_width / 3 else "center"
    pos_y = "top" if center_y < image_height / 3 else "bottom" if center_y > 2 * image_height / 3 else "middle"
    position_val = f"{pos_y} {pos_x}"

    # Size
    area = w * h
    size = "tiny" if area < 2089 else "small" if area < 20890 else "medium" if area < 208896 else "large"

    # Apply test changes
    change_type = None
    change_detail = None
    if test_change:
        if test_change.get("method") == "replace":
            # Flip category
            old_category = category
            category = "weed" if category == "crop" else "crop"
            change_type = "replace"
            change_detail = {"attribute": "category", "from": old_category, "to": category}
        elif test_change.get("method") == "swap":
            override = test_change.get("override", {})
            old_size, old_position = size, position_val
            size = override.get("size", size)
            position_val = override.get("position", position_val)
            change_type = "swap"
            change_detail = {"attribute": "size+position", "from": (old_size, old_position), "to": (size, position_val)}

    sentence = f"{size} {category} in the {position_val}"
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
                        if label_id == 0:
                            crop_count += 1
                        elif label_id == 1:
                            weed_count += 1
                    except:
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
            bboxes, categories, label_ids = [], [], []
            crop_count, weed_count = 0, 0

            if os.path.exists(bbox_csv_path):
                with open(bbox_csv_path, "r", encoding="utf-8") as f:
                    reader = csv.DictReader(f, fieldnames=["left", "top", "right", "bottom", "label_id", "stem_x", "stem_y"])
                    for row in reader:
                        try:
                            label_id = int(row["label_id"])
                            x, y, w, h = int(row["left"]), int(row["top"]), int(row["right"]) - int(row["left"]), int(row["bottom"]) - int(row["top"])
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
                        except:
                            continue

            mask_path = os.path.join(LABELIDS_DIR, image_id_str + ".png")
            full_mask = load_segmentation_mask(mask_path, width, height)

            # Instance annotations
            for i, (bbox, category, label_id) in enumerate(zip(bboxes, categories, label_ids)):
                segmentation = extract_instance_segmentation(full_mask, bbox, label_id)
                instances_data["annotations"].append(
                    {
                        "id": ann_id,
                        "image_id": image_id,
                        "category_id": cat_name_to_id[category],
                        "bbox": bbox,
                        "segmentation": segmentation,
                        "area": bbox[2] * bbox[3],
                        "iscrowd": 0,
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
            ref = {"file_name": file_name, "image_id": image_id}
            if split_name == "train" and negative_expression:
                ref["negative_sentence"] = negative_expression
            refs_data.append(ref)

            image_id += 1

    # Save grefs(unc).json
    grefs_path = os.path.join(OUTPUT_DIR, "grefs(unc).json")
    grefs_data = {"info": instances_data["info"], "images": []}
    for img in instances_data["images"]:
        image_entry = img.copy()
        instance_sentences = []
        for ann in instances_data["annotations"]:
            if ann["image_id"] == img["id"]:
                category = "crop" if ann["category_id"] == 1 else "weed"
                bbox = ann["bbox"]
                sentence, _, _ = generate_single_referring_expression(category, bbox, img["width"], img["height"], 0, 0)
                instance_sentences.append({"ann_id": ann["id"], "category_id": ann["category_id"], "sentence": sentence})
        if instance_sentences:
            image_entry["instance_sentences"] = instance_sentences
        grefs_data["images"].append(image_entry)
    with open(grefs_path, "w", encoding="utf-8") as f:
        json.dump(grefs_data, f, indent=4)

    # Save instances.json
    instances_data_copy = instances_data.copy()
    instances_data_copy.pop("images", None)
    instances_path = os.path.join(OUTPUT_DIR, "instances.json")
    with open(instances_path, "w", encoding="utf-8") as f:
        json.dump(instances_data_copy, f, indent=2)

    # ------------------ Build test set changes ------------------
    print("\nPlanning and applying test set changes...")
    imageid2info = {img["id"]: img for img in instances_data["images"]}
    test_instances = []
    for ann in instances_data["annotations"]:
        img_split = imageid2info[ann["image_id"]]["split"]
        if img_split == "test":
            x, y, w, h = ann["bbox"]
            cx, cy = x + w / 2, y + h / 2
            pos_x = (
                "left"
                if cx < imageid2info[ann["image_id"]]["width"] / 3
                else "right"
                if cx > 2 * imageid2info[ann["image_id"]]["width"] / 3
                else "center"
            )
            pos_y = (
                "top"
                if cy < imageid2info[ann["image_id"]]["height"] / 3
                else "bottom"
                if cy > 2 * imageid2info[ann["image_id"]]["height"] / 3
                else "middle"
            )
            position_val = f"{pos_y} {pos_x}"
            area = w * h
            size = "tiny" if area < 2089 else "small" if area < 20890 else "medium" if area < 208896 else "large"
            category = "crop" if ann["category_id"] == 1 else "weed"
            test_instances.append({"ann_id": ann["id"], "image_id": ann["image_id"], "category": category, "size": size, "position": position_val})

    total_test_instances = len(test_instances)
    target_swap = total_test_instances // 2
    target_replace = total_test_instances - target_swap

    # Group by image_id for swap candidates (allow cross-category pairing)
    groups = defaultdict(list)
    for inst in test_instances:
        groups[inst["image_id"]].append(inst)

    # Select swap pairs (50%)
    paired = set()
    swap_changes_by_ann = {}
    candidate_pairs = []
    for image_id, instances in groups.items():
        if len(instances) >= 2:
            random.shuffle(instances)
            for i in range(0, len(instances) - 1, 2):
                if len(candidate_pairs) < target_swap:
                    a, b = instances[i], instances[i + 1]
                    candidate_pairs.append((a["ann_id"], b["ann_id"]))
                    paired.update([a["ann_id"], b["ann_id"]])
                else:
                    break
        if len(candidate_pairs) >= target_swap:
            break

    # Assign swap changes
    for a, b in candidate_pairs[:target_swap]:
        a_info = next(x for x in test_instances if x["ann_id"] == a)
        b_info = next(x for x in test_instances if x["ann_id"] == b)
        swap_changes_by_ann[a] = {"method": "swap", "override": {"size": b_info["size"], "position": b_info["position"]}}
        swap_changes_by_ann[b] = {"method": "swap", "override": {"size": a_info["size"], "position": a_info["position"]}}

    # Assign replace changes (50%)
    remaining = [inst["ann_id"] for inst in test_instances if inst["ann_id"] not in paired]
    random.shuffle(remaining)
    replace_by_ann = {ann_id: {"method": "replace"} for ann_id in remaining[:target_replace]}

    # Combine changes
    test_changes_by_ann_id = {**swap_changes_by_ann, **replace_by_ann}

    # Ensure all test instances have a change (handle any shortfall)
    for inst in test_instances:
        if inst["ann_id"] not in test_changes_by_ann_id:
            test_changes_by_ann_id[inst["ann_id"]] = {"method": "replace"}

    # Report statistics
    replace_count = sum(1 for ch in test_changes_by_ann_id.values() if ch["method"] == "replace")
    swap_count = sum(1 for ch in test_changes_by_ann_id.values() if ch["method"] == "swap")
    print(f"Total test instances: {total_test_instances}")
    print(f"Replace changes: {replace_count}")
    print(f"Swap changes: {swap_count}")
    print(f"Changes applied: {len(test_changes_by_ann_id)}")

    # ------------------ Save grefs(unc).json with changes applied ------------------
    grefs_path = os.path.join(OUTPUT_DIR, "grefs(unc).json")
    grefs_data = {"info": instances_data["info"], "images": []}
    for img in instances_data["images"]:
        image_entry = img.copy()
        instance_sentences = []
        for ann in instances_data["annotations"]:
            if ann["image_id"] == img["id"]:
                category = "crop" if ann["category_id"] == 1 else "weed"
                bbox = ann["bbox"]
                if img["split"] == "test":
                    # Compute original sentence
                    original_sentence, _, _ = generate_single_referring_expression(category, bbox, img["width"], img["height"], 0, 0)
                    # Compute changed sentence
                    test_change = test_changes_by_ann_id.get(ann["id"])
                    changed_sentence, change_type, change_detail = generate_single_referring_expression(
                        category, bbox, img["width"], img["height"], 0, 0, test_change=test_change
                    )
                    instance_sentences.append(
                        {
                            "ann_id": ann["id"],
                            "category_id": ann["category_id"],
                            "original_sentence": original_sentence,
                            "sentence": changed_sentence,
                            "change_type": change_type,
                            "change_detail": change_detail,
                        }
                    )
                else:
                    sentence, _, _ = generate_single_referring_expression(category, bbox, img["width"], img["height"], 0, 0)
                    instance_sentences.append({"ann_id": ann["id"], "category_id": ann["category_id"], "sentence": sentence})
        if instance_sentences:
            image_entry["instance_sentences"] = instance_sentences
        grefs_data["images"].append(image_entry)
    with open(grefs_path, "w", encoding="utf-8") as f:
        json.dump(grefs_data, f, indent=4)

    # ------------------ Apply changes (but do not save the file) ------------------
    # Removed: saving test_sentence_comparison.json
    print("✅ Test changes planned and applied (statistics reported above). No file generated.")
    print("Dataset generation complete!")

    # ------------------ Build perfectly balanced test set changes ------------------
    print("\nPlanning perfectly balanced test set changes (category=replace, size/position=swap)...")
    imageid2info = {img["id"]: img for img in instances_data["images"]}
    test_ann_ids = [ann["id"] for ann in instances_data["annotations"] if imageid2info[ann["image_id"]]["split"] == "test"]
    random.shuffle(test_ann_ids)
    n = len(test_ann_ids)
    n_each = n // 3
    test_changes_by_ann_id = {}

    # Assign first third to category replace, second third to size swap, last third to position swap
    for idx, ann_id in enumerate(test_ann_ids):
        if idx < n_each:
            test_changes_by_ann_id[ann_id] = {"method": "replace", "attr": "category"}
        elif idx < 2 * n_each:
            test_changes_by_ann_id[ann_id] = {"method": "swap", "swap_type": "size_swap"}
        else:
            test_changes_by_ann_id[ann_id] = {"method": "swap", "swap_type": "position_swap"}

    # If n is not divisible by 3, assign the remainder to category replace
    for ann_id in test_ann_ids[3 * n_each :]:
        test_changes_by_ann_id[ann_id] = {"method": "replace", "attr": "category"}

    # Report statistics
    from collections import Counter

    attr_counts = Counter([v["attr"] if v["method"] == "replace" else v["swap_type"] for v in test_changes_by_ann_id.values()])
    print(f"Total test instances: {n}")
    print(f"Change distribution: {dict(attr_counts)}")

    # ------------------ Save grefs(unc).json with changes applied ------------------
    grefs_path = os.path.join(OUTPUT_DIR, "grefs(unc).json")
    grefs_data = {"info": instances_data["info"], "images": []}
    for img in instances_data["images"]:
        image_entry = img.copy()
        instance_sentences = []
        for ann in instances_data["annotations"]:
            if ann["image_id"] == img["id"]:
                category = "crop" if ann["category_id"] == 1 else "weed"
                bbox = ann["bbox"]
                if img["split"] == "test":
                    # Compute original sentence
                    original_sentence, _, _ = generate_single_referring_expression(category, bbox, img["width"], img["height"], 0, 0)
                    test_change = test_changes_by_ann_id.get(ann["id"])
                    # Apply the correct change logic for each type
                    if test_change and test_change.get("method") == "replace" and test_change.get("attr") == "category":
                        # Flip category
                        changed_category = "weed" if category == "crop" else "crop"
                        changed_sentence = f"{original_sentence.split()[0]} {changed_category} in the {' '.join(original_sentence.split()[-2:])}"
                        change_type = "replace"
                        change_detail = {"attribute": "category", "from": category, "to": changed_category}
                    elif test_change and test_change.get("method") == "swap" and test_change.get("swap_type") == "size_swap":
                        # Change size (choose a random size different from the original)
                        sizes = ["tiny", "small", "medium", "large"]
                        old_size = original_sentence.split()[0]
                        possible_sizes = [s for s in sizes if s != old_size]
                        new_size = random.choice(possible_sizes)
                        changed_sentence = f"{new_size} {category} in the {' '.join(original_sentence.split()[-2:])}"
                        change_type = "replace"
                        change_detail = {"attribute": "size", "from": old_size, "to": new_size}
                    elif test_change and test_change.get("method") == "swap" and test_change.get("swap_type") == "position_swap":
                        # Change position (choose a random position different from the original)
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
                        old_position = " ".join(original_sentence.split()[-2:])
                        possible_positions = [p for p in positions if p != old_position]
                        new_position = random.choice(possible_positions)
                        changed_sentence = f"{original_sentence.split()[0]} {category} in the {new_position}"
                        change_type = "replace"
                        change_detail = {"attribute": "position", "from": old_position, "to": new_position}
                    else:
                        changed_sentence = original_sentence
                        change_type = None
                        change_detail = None
                    instance_sentences.append(
                        {
                            "ann_id": ann["id"],
                            "category_id": ann["category_id"],
                            "original_sentence": original_sentence,
                            "test sentence": changed_sentence,
                            "change_type": change_type,
                            "change_detail": change_detail,
                        }
                    )
                else:
                    sentence, _, _ = generate_single_referring_expression(category, bbox, img["width"], img["height"], 0, 0)
                    instance_sentences.append({"ann_id": ann["id"], "category_id": ann["category_id"], "sentence": sentence})
        if instance_sentences:
            image_entry["instance_sentences"] = instance_sentences
        grefs_data["images"].append(image_entry)
    with open(grefs_path, "w", encoding="utf-8") as f:
        json.dump(grefs_data, f, indent=4)

    print("✅ Test changes planned and applied (statistics reported above). No file generated.")
    print("Dataset generation complete!")


if __name__ == "__main__":
    main()
