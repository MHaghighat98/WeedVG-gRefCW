import json
import os

# Paths
OUTPUT_DIR = "C:/Users/jd138001/Downloads/data/grefcoco_format"
GREFS_PATH = os.path.join(OUTPUT_DIR, "grefs(unc).json")
INSTANCES_PATH = os.path.join(OUTPUT_DIR, "instances.json")


def get_original_expression(category, bbox, width, height):
    # Use the same logic as generate_single_referring_expression, but without test_change
    x, y, w, h = bbox
    center_x = x + w / 2
    center_y = y + h / 2

    if center_x < width / 3:
        pos_x = "left"
    elif center_x > 2 * width / 3:
        pos_x = "right"
    else:
        pos_x = "center"

    if center_y < height / 3:
        pos_y = "top"
    elif center_y > 2 * height / 3:
        pos_y = "bottom"
    else:
        pos_y = "middle"

    position_val = f"{pos_y} {pos_x}"

    area = w * h
    if area < 2089:
        size = "tiny"
    elif area < 20890:
        size = "small"
    elif area < 208896:
        size = "medium"
    else:
        size = "large"

    position_phrase = f"in the {position_val}"
    expression = f"{size} {category} {position_phrase}"
    return expression


def main():
    # Load data
    with open(GREFS_PATH, "r", encoding="utf-8") as f:
        grefs = json.load(f)
    with open(INSTANCES_PATH, "r", encoding="utf-8") as f:
        instances_data = json.load(f)

    # Build ann_id -> annotation mapping for fast lookup
    annid2ann = {a["id"]: a for a in instances_data["annotations"]}

    comparison = []

    for img in grefs["images"]:
        if img.get("split") != "test":
            continue
        width = img["width"]
        height = img["height"]
        for inst in img.get("instance_sentences", []):
            ann_id = inst["ann_id"]
            changed_sentence = inst["sentence"]
            category_id = inst["category_id"]
            category = "crop" if category_id == 1 else "weed"
            ann = annid2ann.get(ann_id)
            if ann is None:
                continue
            bbox = ann["bbox"]
            original_sentence = get_original_expression(category, bbox, width, height)
            comparison.append({"image_id": img["id"], "ann_id": ann_id, "original_sentence": original_sentence, "changed_sentence": changed_sentence})

    # Save comparison file
    out_path = os.path.join(OUTPUT_DIR, "test_sentence_comparison.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2)
    print(f"Saved comparison for {len(comparison)} test instances to {out_path}")


if __name__ == "__main__":
    main()
