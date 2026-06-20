import json


def load_gref_annotations(json_path, category_mapping_path=None):
    """
    Load gRefCOCO-style annotations with dynamic negative sentence inference.

    Args:
        json_path: Path to grefs(unc).json
        category_mapping_path: Optional path to instances.json or categories dict
                              If None, categories are inferred from data

    Returns:
        image_data: Dict mapping image_id -> {
            "pos_sentences": list of instance sentences,
            "neg_sentence": negative sentence (explicit or inferred),
            "categories": set of category_ids present
        }
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Load category mapping if provided
    category_id_to_name = {}
    if category_mapping_path:
        with open(category_mapping_path, "r", encoding="utf-8") as f:
            cat_data = json.load(f)
            for cat in cat_data.get("categories", []):
                category_id_to_name[cat["id"]] = cat["name"]

    # If no mapping provided, infer from first pass through data
    if not category_id_to_name:
        # Collect all category IDs
        all_category_ids = set()
        for img in data.get("images", []):
            for inst in img.get("instance_sentences", []):
                all_category_ids.add(inst.get("category_id"))

        # Create default mapping (will be refined during processing)
        for cat_id in sorted(all_category_ids):
            category_id_to_name[cat_id] = f"category_{cat_id}"

    image_data = {}
    for img in data.get("images", []):
        img_id = img.get("id", None)
        if img_id is None:
            continue

        # Extract positive sentences from instance_sentences
        pos_sentences = []
        categories_present = set()
        instance_sentences = []  # CRITICAL FIX: Store full structure with category_id and ann_id

        # Primary source: instance_sentences (gRefCOCO format)
        if "instance_sentences" in img:
            for inst in img["instance_sentences"]:
                # Some datasets use alternate keys; prefer a stable order
                sentence = (
                    inst.get("sentence")
                    or inst.get("original_sentence")
                    or inst.get("raw")
                    or inst.get("sent")
                    or inst.get("test sentence")
                    or inst.get("test_sentence")
                )
                category_id = inst.get("category_id")
                ann_id = inst.get("ann_id")

                # Always collect non-empty sentences for pos_sentences
                if isinstance(sentence, str) and sentence.strip():
                    clean_sentence = sentence.strip()
                    pos_sentences.append(clean_sentence)
                else:
                    # Skip entries with empty/invalid sentence entirely
                    continue

                # Only build structured instance_sentences when category_id exists
                if category_id is None:
                    import warnings

                    warnings.warn(
                        f"Missing 'category_id' in instance_sentence for image {img_id}. "
                        f"Instance: {inst}. This field is REQUIRED for instance-level category mapping. "
                        f"Will keep sentence in pos_sentences but skip category mapping.",
                        UserWarning,
                    )
                else:
                    # Preserve ann_id when available so downstream can align with instances.json annotation ids
                    payload = {"sentence": clean_sentence, "category_id": category_id}
                    if ann_id is not None:
                        payload["ann_id"] = ann_id
                    instance_sentences.append(payload)
                    categories_present.add(category_id)

        # Fallback: sentences field (legacy format)
        elif "sentences" in img:
            for s in img["sentences"]:
                if isinstance(s, dict):
                    if "raw" in s:
                        pos_sentences.append(s["raw"])
                    elif "sent" in s:
                        pos_sentences.append(s["sent"])
                elif isinstance(s, str):
                    pos_sentences.append(s)

        # Get explicit negative sentence or infer it
        neg_sentence = img.get("negative_sentence", None)

        # CRITICAL: Infer negative sentence if missing (handles mixed-category images)
        if neg_sentence is None or neg_sentence == "":
            neg_sentence = _infer_negative_sentence(categories_present, category_id_to_name)

        image_data[img_id] = {
            "pos_sentences": pos_sentences,
            "neg_sentence": neg_sentence,
            "categories": categories_present,  # Store for analysis/debugging
            "file_name": img.get("file_name", f"{img_id}.jpg"),  # CRITICAL: Store actual filename
            "split": img.get("split", "train"),  # CRITICAL: Store split for train/val separation
            "instance_sentences": instance_sentences,  # CRITICAL FIX: Store category_id mapping
        }

    return image_data


def _infer_negative_sentence(categories_present, category_id_to_name):
    """
    Dynamically infer negative sentence based on categories present.
    NO HARDCODING - fully generic for any category structure.

    Args:
        categories_present: Set of category IDs present in image
        category_id_to_name: Dict mapping category_id -> category_name

    Returns:
        Inferred negative sentence string
    """
    # Defensive: Handle empty category mapping
    if not category_id_to_name:
        if not categories_present:
            return "no instances are visible in this image"
        else:
            # Generic fallback when we know instances exist but no category names
            return f"{len(categories_present)} category(ies) are visible in this image"

    if not categories_present:
        # No instances - list all categories as absent
        all_category_names = sorted(category_id_to_name.values())
        if len(all_category_names) == 1:
            return f"no {all_category_names[0]}s are visible in this image"
        elif len(all_category_names) == 2:
            return f"no {all_category_names[0]}s or {all_category_names[1]}s are visible in this image"
        else:
            # More than 2 categories
            cat_list = ", ".join(all_category_names[:-1]) + f", or {all_category_names[-1]}s"
            return f"no {cat_list} are visible in this image"

    elif len(categories_present) == 1:
        # Only one category present - state which category is ABSENT
        present_cat_id = list(categories_present)[0]
        absent_categories = [name for cat_id, name in category_id_to_name.items() if cat_id != present_cat_id]

        if len(absent_categories) == 1:
            return f"no {absent_categories[0]}s are present in this image"
        elif len(absent_categories) == 2:
            return f"no {absent_categories[0]}s or {absent_categories[1]}s are present in this image"
        else:
            cat_list = ", ".join(absent_categories[:-1]) + f", or {absent_categories[-1]}s"
            return f"no {cat_list} are present in this image"

    else:
        # Multiple categories present - there are NO absent categories
        # Return empty string - these images should use only positive sentences for contrastive learning
        # The hierarchy builder will handle this case appropriately
        return ""
