"""
Explicit 2-Level Hierarchy for Contrastive Learning (Existence → Instance)

This module defines an explicit hierarchy L={0,1} tailored to the attached
hierarchicalContrastiveLearning project style:
- Level 0 (Existence): Binary image-level existence vs non-existence derived from
    the image-level negative sentence field. If an image has no instances and a
    valid negative sentence (e.g., "no crops or weeds are visible in this image"),
    it is labeled as non-existence; otherwise, it is labeled as existence.
- Level 1 (Instance): Instance-specific labels based on instance sentences.

This replaces any prior use of per-image negative sentence classes at level 0
with a binary existence target and keeps instance discrimination at level 1.
"""

import os
import json
import socket
import time
import random
import torch
import torch.nn as nn

# Track images we've warned about to avoid repeated noisy warnings across batches/epochs
_WARNED_IMAGES_NO_POS = set()
# For empty-negative level-0 samples we use a dedicated special token.
# Keep the placeholder empty to avoid injecting a natural-language sentence
# that can confuse monitoring/logging and dominate token-level expansions.
_EMPTY_NEGATIVE_PLACEHOLDER = ""
_EMPTY_NEGATIVE_TOKEN = "[EMPTY_NEG_LVL0]"


def _normalize_sentence(text):
    """Strip and validate sentence strings."""
    if isinstance(text, str):
        cleaned = text.strip()
        return cleaned if cleaned else None
    return None


def build_hierarchical_labels(
    batch_image_ids,
    imageid2sentences,
    num_queries=206,
    adaptive_penalties=True,
    query_to_gt_mapping=None,
    instances_per_image=None,
    *,
    include_empty_negatives: bool = False,
    empty_negative_placeholder: str = _EMPTY_NEGATIVE_PLACEHOLDER,
    empty_negative_token: str = _EMPTY_NEGATIVE_TOKEN,
):
    """
        Build explicit 2-level hierarchical labels for contrastive learning.

        Implements the structure consistent with the attached project:
        - Level 0 (l=0): Binary existence (1) vs non-existence (0), derived from the
            image-level negative sentence field. Any image with valid instances is
            treated as existence; images with a valid negative sentence and no
            instances are treated as non-existence.
        - Level 1 (l=1): Instance-level positive sentences (one per instance/query).

    Args:
        batch_image_ids: List of image IDs in the batch
        imageid2sentences: Dict mapping image_id to {'pos_sentences': [...], 'neg_sentence': str}
        num_queries: Number of queries per image (N_q = 206)
        adaptive_penalties: If True, use context-aware penalties based on image content
            - Mixed-category positive images (crop+weed): λ_0=1.0, λ_1=2.5
            - Single-category positive images:            λ_0=1.0, λ_1=2.0
            - Negative images (no instances):             λ_0=2.0, λ_1=1.0

    Returns:
        hierarchical_labels: [B*N_q, 2] - Labels at each level
            - Column 0: Image-level label index (same for all queries in an image)
            - Column 1: Instance-level label index (unique per query)
        level_0_embeddings: List of level-0 label texts ["not_exists", "exists"]
        level_1_embeddings: List of level-1 sentence texts (instance-level positives)
        lambda_weights: [2] - Adaptive penalty weights [λ_0, λ_1] for each level

    Raises:
        ValueError: If required fields are missing from annotations
        KeyError: If image_id not found in imageid2sentences
    """
    # FIX #2: Validate inputs
    if not batch_image_ids:
        raise ValueError("batch_image_ids cannot be empty")

    if not imageid2sentences:
        raise ValueError("imageid2sentences cannot be empty")

    B = len(batch_image_ids)
    N = B * num_queries

    # Initialize hierarchical labels
    hierarchical_labels = torch.zeros(N, 2, dtype=torch.long)

    # Collect all unique sentences for embedding
    # Level-0 uses four negative image-level sentence classes:
    # 0: none_present  -> "no crops or weeds are visible in this image"
    # 1: no_crops      -> "no crops are visible in this image"
    # 2: no_weeds      -> "no weeds are visible in this image"
    # 3: both_present  -> placeholder text for empty negatives / concurrent crops+weeds
    none_present_text = "no crops or weeds are visible in this image"
    no_crops_text = "no crops are visible in this image"
    no_weeds_text = "no weeds are visible in this image"
    effective_placeholder = empty_negative_placeholder
    if effective_placeholder is None:
        effective_placeholder = _EMPTY_NEGATIVE_PLACEHOLDER

    if include_empty_negatives:
        # Use ONLY the special token so this label is short and unambiguous.
        # (No trailing natural language phrase.)
        both_present_text = (empty_negative_token or _EMPTY_NEGATIVE_TOKEN).strip()
    else:
        # Backward-compat path: if caller explicitly provides a placeholder,
        # use it; otherwise keep it empty.
        both_present_text = (effective_placeholder or "").strip()
    level_0_sentences = [none_present_text, no_crops_text, no_weeds_text, both_present_text]
    level_1_sentences = []  # Instance-level positives

    # Map to track sentence -> index
    level_0_map = {
        none_present_text: 0,
        no_crops_text: 1,
        no_weeds_text: 2,
        both_present_text: 3,
    }
    level_1_map = {}

    query_idx = 0
    for batch_idx, img_id in enumerate(batch_image_ids):
        # FIX #2: Validate image_id exists
        if img_id not in imageid2sentences:
            raise KeyError(f"Image ID {img_id} not found in imageid2sentences. Available IDs: {list(imageid2sentences.keys())[:5]}...")

        entry = imageid2sentences[img_id]

        # FIX #2: Validate required fields exist
        if "pos_sentences" not in entry:
            raise ValueError(f"Missing 'pos_sentences' field for image {img_id}. Available fields: {list(entry.keys())}")

        pos_sents = entry["pos_sentences"]
        neg_sent = entry.get("neg_sentence", None)

        # Level 0: Four-class negative-sentence-based label with empty representing both-present
        # Strategy: Prefer instance-derived presence; fallback to negative sentence patterns
        has_instances = isinstance(pos_sents, (list, tuple)) and len(pos_sents) > 0
        if "has_instances" in entry and isinstance(entry["has_instances"], bool):
            has_instances = entry["has_instances"]

        # Temporary flags derived from instance sentences to detect crop/weed presence
        has_crop_flag = False
        has_weed_flag = False

        # We'll resolve instance sentences below; but we may need flags earlier for level-0.
        # So we do a light-weight estimate first using available pos_sents.
        try:
            for s in pos_sents or []:
                ls = s.lower() if isinstance(s, str) else ""
                if "crop" in ls:
                    has_crop_flag = True
                if "weed" in ls:
                    has_weed_flag = True
        except Exception:
            pass

        # Negative sentence pattern mapping (fallback)
        neg_norm = (neg_sent or "").strip().lower() if isinstance(neg_sent, str) else ""
        neg_implies_none = ("no crops or weeds" in neg_norm) or ("no crop or weed" in neg_norm)
        neg_implies_no_crops = ("no crops" in neg_norm) and ("weeds" not in neg_norm or "or weeds" not in neg_norm)
        neg_implies_no_weeds = ("no weeds" in neg_norm) and ("crops" not in neg_norm or "or crops" not in neg_norm)
        neg_is_empty = neg_norm == ""

        # Assign provisional level-0 label
        if not has_instances:
            # No instances: none_present
            level_0_idx = level_0_map[none_present_text]
        else:
            # Instances present. If both crop & weed present -> empty negative
            if has_crop_flag and has_weed_flag:
                level_0_idx = level_0_map[both_present_text]
            elif has_crop_flag and not has_weed_flag:
                # crops present only -> negative sentence states no weeds
                level_0_idx = level_0_map[no_weeds_text]
            elif has_weed_flag and not has_crop_flag:
                # weeds present only -> negative sentence states no crops
                level_0_idx = level_0_map[no_crops_text]
            else:
                # If instance text didn't indicate types, use neg sentence patterns
                if neg_implies_none:
                    level_0_idx = level_0_map[none_present_text]
                elif neg_implies_no_crops:
                    level_0_idx = level_0_map[no_crops_text]
                elif neg_implies_no_weeds:
                    level_0_idx = level_0_map[no_weeds_text]
                elif neg_is_empty:
                    level_0_idx = level_0_map[both_present_text]
                else:
                    # Default to both_present when we have instances but no clear type signal
                    level_0_idx = level_0_map[both_present_text]

        # Optional per-instance metadata for aligning matched queries to sentences
        mapping_for_image = None
        if query_to_gt_mapping is not None and batch_idx < len(query_to_gt_mapping):
            mapping_for_image = query_to_gt_mapping[batch_idx]

        instances_for_image = None
        if instances_per_image is not None and batch_idx < len(instances_per_image):
            instances_for_image = instances_per_image[batch_idx]

        # Build lookup tables from annotations JSON
        ann_sentence_map = {}
        cat_sentence_map = {}
        for sent_entry in entry.get("instance_sentences", []):
            cleaned = _normalize_sentence(sent_entry.get("sentence"))
            if not cleaned:
                continue
            ann_id = sent_entry.get("ann_id")
            if ann_id is not None:
                ann_sentence_map[ann_id] = cleaned
            cat_id = sent_entry.get("category_id")
            if cat_id is not None and cat_id not in cat_sentence_map:
                cat_sentence_map[cat_id] = cleaned

        # Resolve each GT instance to a canonical sentence from annotations
        instance_sentences = []
        if instances_for_image is not None:
            for inst_idx, inst in enumerate(instances_for_image):
                resolved = _normalize_sentence(inst.get("sentence"))
                if resolved is None:
                    for key in ("id", "ann_id", "annotation_id"):
                        ann_id = inst.get(key)
                        if ann_id is not None and ann_id in ann_sentence_map:
                            resolved = ann_sentence_map[ann_id]
                            break
                if resolved is None:
                    cat_id = inst.get("category_id")
                    if cat_id is not None and cat_id in cat_sentence_map:
                        resolved = cat_sentence_map[cat_id]
                if resolved is None and pos_sents:
                    # Fallback: reuse any annotated positive sentence for stability
                    resolved = pos_sents[inst_idx % len(pos_sents)]
                instance_sentences.append(resolved)

        # If instance-based flags were inconclusive, refine them using resolved sentences
        try:
            for s in instance_sentences or []:
                ls = s.lower() if isinstance(s, str) else ""
                if "crop" in ls:
                    has_crop_flag = True
                if "weed" in ls:
                    has_weed_flag = True
        except Exception:
            pass

        # Level 1: Instance-level positives (one per query)
        for q in range(num_queries):
            # Assign level 0 label (binary existence)
            hierarchical_labels[query_idx, 0] = level_0_idx

            # Assign level 1 label (instance-level)
            assigned_sentence = None
            if mapping_for_image is not None and q < len(mapping_for_image):
                gt_idx_raw = mapping_for_image[q]
                gt_idx = int(gt_idx_raw) if isinstance(gt_idx_raw, (int, float)) else -1
                if gt_idx >= 0 and gt_idx < len(instance_sentences):
                    assigned_sentence = _normalize_sentence(instance_sentences[gt_idx])

            if assigned_sentence:
                if assigned_sentence not in level_1_map:
                    level_1_map[assigned_sentence] = len(level_1_sentences)
                    level_1_sentences.append(assigned_sentence)
                level_1_idx = level_1_map[assigned_sentence]
            elif pos_sents:
                # Fallback: reuse any available positive annotation to keep supervision dense
                sent_idx = q % len(pos_sents)
                fallback_sentence = pos_sents[sent_idx]
                if fallback_sentence not in level_1_map:
                    level_1_map[fallback_sentence] = len(level_1_sentences)
                    level_1_sentences.append(fallback_sentence)
                level_1_idx = level_1_map[fallback_sentence]
            else:
                # Empty image: Skip level 1 entirely - no instance discrimination needed
                # These images contribute only through level 0 (if applicable)
                level_1_idx = -1  # Special value to skip level 1

            hierarchical_labels[query_idx, 1] = level_1_idx
            query_idx += 1

    # Adaptive lambda weights based on image content
    if adaptive_penalties:
        # Context-aware penalties: Prioritize different levels based on image content.
        lambda_per_image = []

        def _infer_crop_weed_presence(entry_dict) -> tuple[bool, bool]:
            """Heuristic: infer whether an image contains crop and/or weed instances.

            Uses available text fields (instance_sentences and pos_sentences). This is intentionally
            simple and consistent with the level-0 label heuristics above.
            """
            has_crop = False
            has_weed = False

            # Prefer structured instance sentences if present.
            for sent_entry in entry_dict.get("instance_sentences", []) or []:
                s = sent_entry.get("sentence") if isinstance(sent_entry, dict) else None
                if not isinstance(s, str):
                    continue
                ls = s.lower()
                if "crop" in ls:
                    has_crop = True
                if "weed" in ls:
                    has_weed = True
                if has_crop and has_weed:
                    return True, True

            # Fallback to the simple pos_sentences list.
            for s in entry_dict.get("pos_sentences", []) or []:
                if not isinstance(s, str):
                    continue
                ls = s.lower()
                if "crop" in ls:
                    has_crop = True
                if "weed" in ls:
                    has_weed = True
                if has_crop and has_weed:
                    return True, True

            return has_crop, has_weed

        for img_id in batch_image_ids:
            entry = imageid2sentences[img_id]
            pos_sents = entry["pos_sentences"]
            # Allow explicit override if provided
            has_instances = isinstance(pos_sents, (list, tuple)) and len(pos_sents) > 0
            if "has_instances" in entry and isinstance(entry["has_instances"], bool):
                has_instances = entry["has_instances"]

            if has_instances:
                has_crop_flag, has_weed_flag = _infer_crop_weed_presence(entry)
                if has_crop_flag and has_weed_flag:
                    # Mixed-category positive image: stronger instance discrimination
                    lambda_per_image.append([1.0, 2.5])
                else:
                    # Single-category (or unknown) positive image
                    lambda_per_image.append([1.0, 2.0])
            else:
                # Negative image: emphasize existence classification
                lambda_per_image.append([2.0, 1.0])

        # Average penalties across batch (handles mixed positive/negative batches)
        lambda_weights = torch.tensor(lambda_per_image, dtype=torch.float32).mean(dim=0)
    else:
        # Fixed penalties (legacy): λ_l = 2^l
        lambda_weights = torch.tensor([2**0, 2**1], dtype=torch.float32)  # [1.0, 2.0]

    return hierarchical_labels, level_0_sentences, level_1_sentences, lambda_weights


class HierarchicalContrastiveLoss(nn.Module):
    """
    Hierarchical Multi-label Contrastive Loss (HMLC) from CVPR 2022 paper:
    "Use All the Labels: A Hierarchical Multi-Label Contrastive Learning Framework"

    Supports three loss types:
    - "hmc": Basic hierarchical multi-label contrastive loss
    - "hce": Hierarchical constraint enforcing loss
    - "hmce": Hierarchical multi-label constraint enforcing loss (recommended)

    Implements the hierarchical structure where coarser levels constrain finer levels.
    """

    def __init__(
        self,
        temperature=0.07,
        base_temperature=0.07,
        layer_penalty=None,
        loss_type="hmce",
        target_pos_neg_ratio: float = 0.5,
    ):
        super(HierarchicalContrastiveLoss, self).__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature
        if not layer_penalty:
            self.layer_penalty = self.pow_2
        else:
            self.layer_penalty = layer_penalty
        self.sup_con_loss = SupConLoss(temperature, base_temperature=base_temperature)
        self.loss_type = loss_type
        self.target_pos_neg_ratio = float(target_pos_neg_ratio)
        self.balance_negatives = self.target_pos_neg_ratio > 0

    def pow_2(self, value):
        return torch.pow(2, value)

    def forward(self, features, labels, lambda_weights=None, return_info: bool = False):
        """
        Compute HMLC loss.

        Args:
            features: [N, D] - Query features
            labels: [N, num_levels] - Hierarchical integer labels

        Returns:
            loss: Scalar loss value
        """
        device = features.device
        num_levels = labels.shape[1]

        if lambda_weights is None:
            lambda_weights_tensor = torch.ones(num_levels, device=device, dtype=features.dtype)
        else:
            lambda_weights_tensor = torch.as_tensor(lambda_weights, dtype=features.dtype, device=device)
            if lambda_weights_tensor.numel() < num_levels:
                pad = torch.ones(num_levels - lambda_weights_tensor.numel(), device=device, dtype=features.dtype)
                lambda_weights_tensor = torch.cat([lambda_weights_tensor, pad], dim=0)

        # Convert features to [N, 1, D] format expected by SupConLoss
        if features.dim() == 2:
            features = features.unsqueeze(1)

        # Initialize loss accumulation
        cumulative_loss = torch.tensor(0.0, device=device)
        max_loss_lower_layer = torch.tensor(float("-inf"), device=device)
        active_levels = 0
        per_level_info = []

        # Process levels from coarse to fine (level 0 is coarsest)
        for level in range(num_levels):
            level_labels = labels[:, level : level + 1]
            valid_mask = level_labels >= 0
            if not valid_mask.any():
                continue

            mask_labels = torch.eq(level_labels, level_labels.T)
            pair_valid = valid_mask & valid_mask.T
            # Avoid degenerate self-pairs dominating stats
            pair_valid.fill_diagonal_(False)
            mask_labels = (mask_labels & pair_valid).float()
            if mask_labels.sum() == 0:
                continue

            pos_pairs = float(mask_labels.sum().item())
            neg_pairs = float(pair_valid.sum().item() - pos_pairs)

            negative_mask = None
            keep_fraction = 1.0
            # Note: we disable negative subsampling for level 1 (instance-level)
            # to ensure full instance discrimination signal is preserved.
            if level == 1:
                # Force keeping all negatives at level 1
                keep_fraction = 1.0
                negative_mask = None
            elif self.balance_negatives and neg_pairs > 0 and self.target_pos_neg_ratio > 0:
                desired_neg = pos_pairs / max(self.target_pos_neg_ratio, 1e-6) if pos_pairs > 0 else 0.0
                keep_fraction = 1.0 if desired_neg <= 0 else min(1.0, desired_neg / (neg_pairs + 1e-6))
                if keep_fraction < 0.999 and keep_fraction > 0.0:
                    negative_mask = torch.ones_like(mask_labels, device=mask_labels.device)
                    rand_vals = torch.rand_like(mask_labels, device=mask_labels.device)
                    neg_mask = (pair_valid & (~mask_labels.bool())).float()
                    keep_neg = (rand_vals < keep_fraction).float()
                    negative_mask = torch.where(neg_mask > 0, keep_neg, negative_mask)
                else:
                    keep_fraction = 1.0

            # Diagnostics: compute final pos/neg pairs that will be used after negative subsampling
            final_pos_pairs = int(mask_labels.sum().item())
            if negative_mask is not None:
                final_total_pairs = int(negative_mask.sum().item())
                final_neg_pairs = final_total_pairs - final_pos_pairs
            else:
                final_neg_pairs = int(neg_pairs)
                final_total_pairs = final_pos_pairs + final_neg_pairs
            # final_ratio not used in minimal logging

            # Print per-level pair usage when debug enabled
            if os.environ.get("HMLC_DEBUG", "0") == "1":
                try:
                    # Only emit the final positive/negative counts used for HMLC
                    print(f"[HMLC_PAIRS] level={level}: final_pos={final_pos_pairs} final_neg={final_neg_pairs}")
                except Exception:
                    pass

            # Apply hierarchical constraint based on loss type
            layer_loss = self.sup_con_loss(features, mask=mask_labels, negative_mask=negative_mask)

            # Optional per-anchor diagnostics: show how many positives/negatives each anchor uses
            # Gate with environment variable to avoid noisy output during normal runs
            if os.environ.get("HMLC_DEBUG", "0") == "1":
                try:
                    # Reconstruct boolean masks for per-anchor counts
                    pair_valid_bool = pair_valid.bool()
                    pos_bool = mask_labels.bool()
                    if negative_mask is not None:
                        neg_keep_bool = negative_mask.bool() & pair_valid_bool & (~pos_bool)
                    else:
                        neg_keep_bool = pair_valid_bool & (~pos_bool)

                    # Compute counts per anchor (as CPU tensors for printing)
                    pos_counts = pos_bool.sum(dim=1).to(torch.long).detach().cpu()
                    neg_counts = neg_keep_bool.sum(dim=1).to(torch.long).detach().cpu()

                    # Sample up to 50 queries among those valid for this level to print counts.
                    # Only consider queries that participate (valid_mask == True) to avoid noisy zeros.
                    try:
                        valid_vec = valid_mask.view(-1).bool().cpu()
                        if valid_vec.any():
                            participating_idx = torch.nonzero(valid_vec, as_tuple=False).view(-1).tolist()
                        else:
                            participating_idx = []
                    except Exception:
                        # Fallback: treat all anchors as participating
                        participating_idx = list(range(pos_counts.numel()))

                    num_participating = len(participating_idx)
                    sample_n = min(50, num_participating)
                    if sample_n > 0:
                        try:
                            sampled = random.sample(participating_idx, sample_n)
                        except Exception:
                            sampled = participating_idx[:sample_n]

                        per_anchor_strs = []
                        for idx in sampled:
                            p = int(pos_counts[idx].item()) if idx < len(pos_counts) else 0
                            n = int(neg_counts[idx].item()) if idx < len(neg_counts) else 0
                            lbl = int(level_labels[idx].item()) if idx < level_labels.size(0) else -1
                            per_anchor_strs.append(f"{idx}(lbl={lbl}): +{p}/-{n}")

                        # Only print from the main process when distributed training is used
                        try:
                            is_main = True
                            if torch.distributed.is_available() and torch.distributed.is_initialized():
                                is_main = torch.distributed.get_rank() == 0
                        except Exception:
                            is_main = True

                        if is_main:
                            print(
                                f"[HMLC_PAIRS_PER_QUERY] level={level} sampled_queries={sample_n} total_participating={num_participating}\n  "
                                + ", ".join(per_anchor_strs),
                                flush=True,
                            )
                            # Also optionally dump detailed per-query (or sampled) stats to a JSONL file
                            try:
                                jsonl_path = os.environ.get("HMLC_DEBUG_JSONL", "hmlc_debug_pairs.jsonl")
                                dump_full = os.environ.get("HMLC_DEBUG_JSONL_FULL", "0") == "1"
                                n_q_env = int(os.environ.get("HMLC_DEBUG_NUM_QUERIES", "0")) if os.environ.get("HMLC_DEBUG_NUM_QUERIES") else 0
                                # Build per-query entries: either full (all queries) or sampled as above
                                if dump_full:
                                    per_query_list = []
                                    for idx in range(pos_counts.numel()):
                                        per_query_list.append(
                                            {
                                                "query_idx": int(idx),
                                                "label0": int(level_labels[idx, 0].item())
                                                if level_labels.dim() > 1
                                                else int(level_labels[idx].item()),
                                                "label1": int(labels[idx, 1].item()) if labels.shape[1] > 1 else None,
                                                "pos_count": int(pos_counts[idx].item()),
                                                "neg_count": int(neg_counts[idx].item()),
                                            }
                                        )
                                else:
                                    per_query_list = []
                                    for sidx in sampled:
                                        per_query_list.append(
                                            {
                                                "query_idx": int(sidx),
                                                "label0": int(level_labels[sidx, 0].item())
                                                if level_labels.dim() > 1
                                                else int(level_labels[sidx].item()),
                                                "label1": int(labels[sidx, 1].item()) if labels.shape[1] > 1 else None,
                                                "pos_count": int(pos_counts[sidx].item()),
                                                "neg_count": int(neg_counts[sidx].item()),
                                            }
                                        )

                                # If user provided num_queries, aggregate per-image stats
                                per_image_stats = None
                                N_total = pos_counts.numel()
                                if n_q_env > 0 and N_total % n_q_env == 0:
                                    B_images = N_total // n_q_env
                                    per_image_stats = []
                                    for im_idx in range(B_images):
                                        start = im_idx * n_q_env
                                        end = start + n_q_env
                                        sum_pos = int(pos_counts[start:end].sum().item())
                                        sum_neg = int(neg_counts[start:end].sum().item())
                                        per_image_stats.append(
                                            {
                                                "image_index": int(im_idx),
                                                "start_query": int(start),
                                                "end_query": int(end - 1),
                                                "sum_pos": sum_pos,
                                                "sum_neg": sum_neg,
                                                "avg_pos_per_query": sum_pos / max(n_q_env, 1),
                                                "avg_neg_per_query": sum_neg / max(n_q_env, 1),
                                            }
                                        )

                                dump_obj = {
                                    "ts": time.time(),
                                    "host": socket.gethostname(),
                                    "pid": os.getpid(),
                                    "level": int(level),
                                    "sampled_queries": sample_n,
                                    "total_participating": num_participating,
                                    "per_query": per_query_list,
                                    "per_image_stats": per_image_stats,
                                }

                                # Append JSONL
                                try:
                                    with open(jsonl_path, "a", encoding="utf8") as jf:
                                        jf.write(json.dumps(dump_obj) + "\n")
                                except Exception:
                                    # Don't crash training for debug IO failures
                                    pass
                            except Exception:
                                pass
                    else:
                        try:
                            is_main = True
                            if torch.distributed.is_available() and torch.distributed.is_initialized():
                                is_main = torch.distributed.get_rank() == 0
                        except Exception:
                            is_main = True
                        if is_main:
                            print(f"[HMLC_PAIRS_PER_QUERY] level={level} sampled_queries=0 total_participating=0", flush=True)
                except Exception:
                    pass

            level_weight = lambda_weights_tensor[level] if level < lambda_weights_tensor.numel() else 1.0
            if not torch.is_tensor(level_weight):
                level_weight = torch.tensor(float(level_weight), device=device, dtype=features.dtype)

            if self.loss_type == "hmc":
                # Basic HMLC: equal weighting
                weighted_loss = self.layer_penalty(torch.tensor(1.0 / (level + 1), device=device)) * layer_loss
            elif self.loss_type == "hce":
                # HCE: enforce constraint that finer levels can't have lower loss than coarser
                layer_loss = torch.max(max_loss_lower_layer, layer_loss)
                weighted_loss = layer_loss
            elif self.loss_type == "hmce":
                # HMCE: combine penalty weighting with constraint enforcing
                layer_loss = torch.max(max_loss_lower_layer, layer_loss)
                weighted_loss = self.layer_penalty(torch.tensor(1.0 / (level + 1), device=device)) * layer_loss
            else:
                raise ValueError(f"Unknown loss_type: {self.loss_type}. Must be 'hmc', 'hce', or 'hmce'")

            cumulative_loss += level_weight * weighted_loss
            max_loss_lower_layer = torch.max(max_loss_lower_layer, layer_loss)
            active_levels += 1
            per_level_info.append(
                (
                    int(level),
                    float(layer_loss.detach().cpu().item()),
                    float((level_weight * weighted_loss).detach().cpu().item()),
                    bool(valid_mask.any()),
                    float(pos_pairs),
                    float(neg_pairs),
                    float(keep_fraction),
                    float(final_pos_pairs),
                    float(final_neg_pairs),
                    float(final_pos_pairs / max(final_neg_pairs, 1e-6)),
                )
            )

            # Optional debug print when environment variable set
            if os.environ.get("HMLC_DEBUG", "0") == "1":
                # Minimal per-level summary: raw loss, weighted loss, final pos/neg used
                print(f"[HMLC_DEBUG] num_levels={num_levels}, active_levels={active_levels}")
                for (
                    lvl,
                    raw_loss,
                    weighted_loss_val,
                    _participated,
                    _pos_pairs,
                    _neg_pairs,
                    _keep_frac,
                    final_pos_pairs,
                    final_neg_pairs,
                    _final_ratio,
                ) in per_level_info:
                    print(
                        f"  level={lvl} raw_loss={raw_loss:.6f} weighted={weighted_loss_val:.6f} "
                        f"final_pos={final_pos_pairs:.0f} final_neg={final_neg_pairs:.0f}"
                    )

        if active_levels == 0:
            out_loss = torch.tensor(0.0, device=device)
        else:
            out_loss = cumulative_loss / active_levels

        if return_info:
            # Return both scalar loss and per-level diagnostic info
            return out_loss, per_level_info

        return out_loss


class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf"""

    def __init__(self, temperature=0.07, contrast_mode="all", base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None, negative_mask=None):
        """
        Compute supervised contrastive loss.

        Args:
            features: [bsz, n_views, ...] - hidden vectors
            labels: [bsz] - ground truth labels (optional)
            mask: [bsz, bsz] - contrastive mask (optional)

        Returns:
            loss: Scalar loss value
        """
        device = features.device

        if len(features.shape) < 3:
            raise ValueError("`features` needs to be [bsz, n_views, ...], at least 3 dimensions required")
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError("Cannot define both `labels` and `mask`")
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError("Num of labels does not match num of features")
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)

        if self.contrast_mode == "one":
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == "all":
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError(f"Unknown contrast_mode: {self.contrast_mode}")

        # Compute logits
        anchor_dot_contrast = torch.div(torch.matmul(anchor_feature, contrast_feature.T), self.temperature)

        # For numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # Tile mask
        base_mask_shape = mask.shape
        mask = mask.repeat(anchor_count, contrast_count)

        # CRITICAL FIX: Do not mask self-contrast in positive mask for hierarchical loss
        # Self should be considered a positive for the same class
        # Only mask self in denominator
        logits_mask = torch.scatter(torch.ones_like(mask), 1, torch.arange(batch_size * anchor_count).view(-1, 1).to(device), 0)
        if negative_mask is not None:
            neg_mask = negative_mask
            if neg_mask.shape == base_mask_shape:
                neg_mask = neg_mask.repeat(anchor_count, contrast_count)
            elif neg_mask.shape != mask.shape:
                raise ValueError("negative_mask must have the same shape as mask")
            logits_mask = logits_mask * neg_mask.to(device=device, dtype=logits_mask.dtype)
        # mask remains unchanged (self is positive)
        # Only apply logits_mask to denominator
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-8)

        # Compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-8)

        # Loss
        loss = -(self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss
