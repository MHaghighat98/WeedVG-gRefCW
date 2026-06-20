#!/usr/bin/env python3
"""gRef-CW evaluator for MDETR and SAM3."""

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_ROOT = REPO_ROOT / "Weed-VG"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(MODEL_ROOT) not in sys.path:
    sys.path.insert(0, str(MODEL_ROOT))

import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)

SIZE_KEYWORDS = {"tiny": "Tiny", "small": "Small", "medium": "Med", "large": "Large"}
CATEGORY_MAP = {1: "crop", 2: "weed"}


# ── Utility ──────────────────────────────────────────────────────────────────

def _extract_size_from_sentence(text: str) -> Optional[str]:
    tl = text.lower()
    for kw, label in SIZE_KEYWORDS.items():
        if kw in tl:
            return label
    return None


def _ensure_cxcywh_norm(bbox_xywh: List[float], width: int, height: int) -> List[float]:
    x, y, w, h = [float(v) for v in bbox_xywh]
    cx, cy = x + w / 2.0, y + h / 2.0
    if width <= 0 or height <= 0:
        return [0.5, 0.5, 0.01, 0.01]
    return [cx / width, cy / height, w / width, h / height]


def box_cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    cx, cy, w, h = boxes.unbind(-1)
    return torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dim=-1)


def box_xyxy_to_cxcywh(boxes: torch.Tensor) -> torch.Tensor:
    x0, y0, x1, y1 = boxes.unbind(-1)
    return torch.stack([(x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0], dim=-1)


def compute_iou_matrix(boxes1_xyxy: torch.Tensor, boxes2_xyxy: torch.Tensor) -> torch.Tensor:
    """IoU matrix [N, M]."""
    a1 = (boxes1_xyxy[:, 2] - boxes1_xyxy[:, 0]).clamp(0) * \
         (boxes1_xyxy[:, 3] - boxes1_xyxy[:, 1]).clamp(0)
    a2 = (boxes2_xyxy[:, 2] - boxes2_xyxy[:, 0]).clamp(0) * \
         (boxes2_xyxy[:, 3] - boxes2_xyxy[:, 1]).clamp(0)
    lt = torch.max(boxes1_xyxy[:, None, :2], boxes2_xyxy[None, :, :2])
    rb = torch.min(boxes1_xyxy[:, None, 2:], boxes2_xyxy[None, :, 2:])
    wh = (rb - lt).clamp(0)
    inter = wh[..., 0] * wh[..., 1]
    union = a1[:, None] + a2[None, :] - inter
    return inter / union.clamp(min=1e-8)


def compute_giou(box1_xyxy: torch.Tensor, box2_xyxy: torch.Tensor) -> float:
    """Compute GIoU between two single boxes (each [1,4] in xyxy format)."""
    x1 = max(float(box1_xyxy[0, 0]), float(box2_xyxy[0, 0]))
    y1 = max(float(box1_xyxy[0, 1]), float(box2_xyxy[0, 1]))
    x2 = min(float(box1_xyxy[0, 2]), float(box2_xyxy[0, 2]))
    y2 = min(float(box1_xyxy[0, 3]), float(box2_xyxy[0, 3]))
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    a1 = max(0.0, float(box1_xyxy[0, 2] - box1_xyxy[0, 0])) * \
         max(0.0, float(box1_xyxy[0, 3] - box1_xyxy[0, 1]))
    a2 = max(0.0, float(box2_xyxy[0, 2] - box2_xyxy[0, 0])) * \
         max(0.0, float(box2_xyxy[0, 3] - box2_xyxy[0, 1]))
    union = a1 + a2 - inter
    iou = inter / max(union, 1e-8)
    ex1 = min(float(box1_xyxy[0, 0]), float(box2_xyxy[0, 0]))
    ey1 = min(float(box1_xyxy[0, 1]), float(box2_xyxy[0, 1]))
    ex2 = max(float(box1_xyxy[0, 2]), float(box2_xyxy[0, 2]))
    ey2 = max(float(box1_xyxy[0, 3]), float(box2_xyxy[0, 3]))
    enclose_area = max(0.0, ex2 - ex1) * max(0.0, ey2 - ey1)
    giou = iou - (enclose_area - union) / max(enclose_area, 1e-8)
    return float(giou)


# ── Data Structures ──────────────────────────────────────────────────────────

@dataclass
class SentenceRecord:
    ann_id: int
    category_id: Optional[int]
    sentence: str
    bbox_xywh: List[float]
    gt_box_cxcywh: Optional[List[float]] = None
    change_type: Optional[str] = None
    change_detail: Optional[Dict[str, Any]] = None
    is_manipulated: bool = False
    size_label: Optional[str] = None


# ── Dataset ──────────────────────────────────────────────────────────────────

class GRefCWDataset(Dataset):
    def __init__(self, grefs_json: str, instances_json: str, images_root: str, split: str):
        self.images_root = images_root
        self.split = split

        with open(grefs_json, "r") as f:
            grefs = json.load(f)
        with open(instances_json, "r") as f:
            coco = json.load(f)

        self.catid_to_name = {}
        for cat in coco.get("categories", []):
            cid = cat.get("id")
            nm = cat.get("name", "")
            if cid is not None and nm:
                self.catid_to_name[int(cid)] = nm.strip()

        annid_to_bbox = {}
        for ann in coco.get("annotations", []):
            aid = ann.get("id")
            bbox = ann.get("bbox")
            if aid is not None and isinstance(bbox, (list, tuple)) and len(bbox) == 4:
                annid_to_bbox[int(aid)] = [float(v) for v in bbox]

        self.items = []
        for img in grefs.get("images", []):
            if img.get("split") != split:
                continue
            image_id = img.get("id")
            file_name = img.get("file_name")
            if image_id is None or not file_name:
                continue

            w = int(img.get("width", 0) or 0)
            h = int(img.get("height", 0) or 0)

            sent_recs, manipulated_recs = [], []

            for rec in img.get("instance_sentences", []):
                ann_id = rec.get("ann_id")
                category_id = rec.get("category_id")
                change_type = rec.get("change_type")
                change_detail = rec.get("change_detail")
                test_sentence = rec.get("test_sentence")

                if split == "test":
                    sentence = rec.get("original_sentence", rec.get("sentence", ""))
                else:
                    sentence = rec.get("sentence", "")

                if not sentence or ann_id is None:
                    continue
                bbox = annid_to_bbox.get(int(ann_id))
                if bbox is None:
                    continue

                size_label = _extract_size_from_sentence(sentence)
                sent_recs.append(SentenceRecord(
                    ann_id=int(ann_id),
                    category_id=int(category_id) if category_id is not None else None,
                    sentence=sentence.strip(), bbox_xywh=bbox, size_label=size_label,
                ))

                if change_type and isinstance(test_sentence, str) and test_sentence.strip():
                    manipulated_recs.append(SentenceRecord(
                        ann_id=int(ann_id),
                        category_id=int(category_id) if category_id is not None else None,
                        sentence=test_sentence.strip(), bbox_xywh=bbox,
                        change_type=change_type,
                        change_detail=change_detail if isinstance(change_detail, dict) else {},
                        is_manipulated=True,
                        size_label=_extract_size_from_sentence(test_sentence),
                    ))

            self.items.append({
                "image_id": int(image_id), "file_name": file_name,
                "width": w, "height": h,
                "sentences": sent_recs, "manipulated_sentences": manipulated_recs,
            })

        print(f"[dataset] Loaded {len(self.items)} images for split='{split}'")
        ts = sum(len(it["sentences"]) for it in self.items)
        tm = sum(len(it["manipulated_sentences"]) for it in self.items)
        print(f"[dataset] Total sentences: {ts}, manipulated: {tm}")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        path = os.path.join(self.images_root, item["file_name"])
        img = Image.open(path).convert("RGB")
        w_img, h_img = img.size
        for rec in item["sentences"]:
            rec.gt_box_cxcywh = _ensure_cxcywh_norm(rec.bbox_xywh, w_img, h_img)
        for rec in item["manipulated_sentences"]:
            rec.gt_box_cxcywh = _ensure_cxcywh_norm(rec.bbox_xywh, w_img, h_img)
        return {
            "image": img, "image_id": item["image_id"], "file_name": item["file_name"],
            "width": w_img, "height": h_img,
            "sentences": item["sentences"], "manipulated_sentences": item["manipulated_sentences"],
        }


def collate_fn(batch):
    return batch


# ── Model Wrappers ───────────────────────────────────────────────────────────

class MDETRWrapper:
    def __init__(self, device: str = "cuda", variant: str = "refcoco",
                 mdetr_root: str = "mdetr"):
        self.device = torch.device(device)
        sys.path.insert(0, os.path.abspath(mdetr_root))
        from hubconf import _make_detr
        print(f"[MDETR] Loading MDETR-ResNet101 variant='{variant}'...")
        model = _make_detr("resnet101")
        variant_urls = {
            "pretrained": "https://zenodo.org/record/4721981/files/pretrained_resnet101_checkpoint.pth",
            "refcoco": "https://zenodo.org/record/4721981/files/refcoco_resnet101_checkpoint.pth",
            "refcocoplus": "https://zenodo.org/record/4721981/files/refcoco%2B_resnet101_checkpoint.pth",
            "refcocog": "https://zenodo.org/record/4721981/files/refcocog_resnet101_checkpoint.pth",
        }
        url = variant_urls.get(variant, variant_urls["refcoco"])
        ckpt = torch.hub.load_state_dict_from_url(url=url, map_location="cpu")
        model.load_state_dict(ckpt["model"], strict=False)
        self.model = model
        self.model.eval()
        self.model.to(self.device)
        import torchvision.transforms.functional as TF
        self._TF = TF
        self._cached_image_id = None
        self._cached_img_tensor = None
        print(f"[MDETR] Model loaded on {self.device}")

    def _preprocess(self, image: Image.Image) -> torch.Tensor:
        TF = self._TF
        w, h = image.size
        scale = 800.0 / min(w, h)
        nw, nh = int(w * scale), int(h * scale)
        if max(nw, nh) > 1333:
            scale = 1333.0 / max(w, h)
            nw, nh = int(w * scale), int(h * scale)
        image = image.resize((nw, nh), Image.BILINEAR)
        return TF.normalize(TF.to_tensor(image), MEAN, STD)

    def set_image(self, image: Image.Image, image_id: int):
        self._cached_image_id = image_id
        self._cached_img_tensor = self._preprocess(image).unsqueeze(0).to(self.device)

    @torch.no_grad()
    def predict(self, image: Image.Image, text: str,
                image_id: int = -1) -> Tuple[torch.Tensor, torch.Tensor]:
        from util.misc import NestedTensor
        if (image_id >= 0 and self._cached_image_id == image_id
                and self._cached_img_tensor is not None):
            it = self._cached_img_tensor
        else:
            it = self._preprocess(image).unsqueeze(0).to(self.device)
        mask = torch.zeros((1, it.shape[2], it.shape[3]),
                           dtype=torch.bool, device=self.device)
        samples = NestedTensor(it, mask)
        mc = self.model(samples, [text], encode_and_save=True)
        out = self.model(samples, [text], encode_and_save=False, memory_cache=mc)
        pred_boxes = out["pred_boxes"][0]  # [100,4] cxcywh normalised
        prob = F.softmax(out["pred_logits"][0], dim=-1)
        scores = 1 - prob[:, -1]  # objectness
        return pred_boxes, scores


class SAM3Wrapper:
    def __init__(self, device: str = "cuda", bpe_path: Optional[str] = None):
        self.device = torch.device(device)
        from sam3.model_builder import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor
        print("[SAM3] Building SAM3 image model...")
        kwargs = {"device": device, "eval_mode": True}
        if bpe_path:
            kwargs["bpe_path"] = bpe_path
        self.model = build_sam3_image_model(**kwargs)
        self.processor = Sam3Processor(self.model, confidence_threshold=0)
        self._cached_state = None
        self._cached_image_id = None
        print(f"[SAM3] Model loaded on {self.device}")

    def set_image(self, image: Image.Image, image_id: int):
        self._cached_image_id = image_id
        self._cached_state = self.processor.set_image(image)

    @torch.no_grad()
    def predict(self, image: Image.Image, text: str,
                image_id: int = -1) -> Tuple[torch.Tensor, torch.Tensor]:
        w, h = image.size
        if (image_id >= 0 and self._cached_image_id == image_id
                and self._cached_state is not None):
            inf_st = self._cached_state
        else:
            inf_st = self.processor.set_image(image)
            self._cached_image_id = image_id
            self._cached_state = inf_st
        self.processor.reset_all_prompts(inf_st)
        inf_st = self.processor.set_text_prompt(state=inf_st, prompt=text)
        bx = inf_st.get("boxes")
        sc = inf_st.get("scores")
        if bx is None or sc is None or len(bx) == 0:
            return torch.zeros((0, 4), device=self.device), \
                   torch.zeros((0,), device=self.device)
        bx = bx.clone().to(self.device).float()
        sc = sc.clone().to(self.device).float()
        bx[:, 0] /= w; bx[:, 2] /= w
        bx[:, 1] /= h; bx[:, 3] /= h
        return box_xyxy_to_cxcywh(bx), sc


# ── Evaluation Engine ────────────────────────────────────────────────────────

def evaluate_model(model_wrapper, dataset: GRefCWDataset, args):
    device = args.device
    iou_thresh = args.iou_thresh
    top_k = args.top_k
    split = args.split

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=0, collate_fn=collate_fn)

    total_sent = 0
    det_match_ok = 0
    det_match_miss = 0
    hit_r1 = 0
    hit_rk = 0
    miou_sum = 0.0

    inst_total = 0
    inst_hit_rk = 0

    n_acc_total = 0
    n_acc_correct = 0
    seen_manip_keys = set()
    # Paper's Table 5 "Avg" uses 3616 (Weed-VG model-eligible count) as
    # a fixed denominator for cross-model comparison.  Per-type N-Acc uses
    # each model's own model-eligible denominator.
    NACC_PAPER_DENOM = 3616

    cat_size_stats = {}
    density_stats = {}
    manip_stats = {}

    def density_bin(n):
        if n <= 10: return "1-10"
        if n <= 20: return "11-20"
        if n <= 30: return "21-30"
        return ">30"

    img_count = 0

    for batch_idx, batch in enumerate(loader):
        batch_start = time.time()
        for item in batch:
            img_count += 1
            image = item["image"]
            image_id = item["image_id"]
            sent_recs = item["sentences"]
            manipulated_recs = item["manipulated_sentences"]
            detected_ann_ids = set()

            if hasattr(model_wrapper, "set_image"):
                model_wrapper.set_image(image, image_id)

            sent_to_recs: Dict[str, List[SentenceRecord]] = {}
            sent_texts: List[str] = []
            for rec in sent_recs:
                s = rec.sentence.strip()
                if not s:
                    continue
                if s not in sent_to_recs:
                    sent_to_recs[s] = []
                    sent_texts.append(s)
                sent_to_recs[s].append(rec)

            image_gt_count = sum(len(rs) for rs in sent_to_recs.values())
            dbin = density_bin(image_gt_count)
            if dbin not in density_stats:
                density_stats[dbin] = {
                    k: 0 if "iou" not in k else 0.0
                    for k in [
                        "n_sent", "inst_total", "inst_hit_rk", "miou_sum",
                        "crop_n_sent", "crop_inst_total", "crop_inst_hit_rk",
                        "crop_miou_sum", "weed_n_sent", "weed_inst_total",
                        "weed_inst_hit_rk", "weed_miou_sum",
                    ]
                }

            for sent_text in sent_texts:
                recs = sent_to_recs[sent_text]
                total_sent += 1

                gt_boxes = [r.gt_box_cxcywh for r in recs if r.gt_box_cxcywh is not None]
                cat_id = recs[0].category_id
                cat_name = CATEGORY_MAP.get(cat_id, "unknown") if cat_id else "unknown"
                size_label = recs[0].size_label or "unknown"
                n_gt = len(gt_boxes)

                if n_gt == 0:
                    det_match_miss += 1
                    density_stats[dbin]["n_sent"] += 1
                    continue

                gt_t = torch.tensor(gt_boxes, device=device, dtype=torch.float32)
                gt_xyxy = box_cxcywh_to_xyxy(gt_t)

                inst_total += n_gt
                density_stats[dbin]["inst_total"] += n_gt
                density_stats[dbin]["n_sent"] += 1
                if cat_name in ("crop", "weed"):
                    density_stats[dbin][f"{cat_name}_n_sent"] += 1
                    density_stats[dbin][f"{cat_name}_inst_total"] += n_gt

                try:
                    pred_boxes, scores = model_wrapper.predict(
                        image, sent_text, image_id=image_id)
                except Exception as e:
                    if img_count <= 3:
                        print(f"[warn] predict failed: {e}")
                    det_match_miss += 1
                    continue

                if pred_boxes.numel() == 0:
                    det_match_miss += 1
                    continue

                pred_boxes = pred_boxes.to(device)
                scores = scores.to(device)

                order = torch.argsort(scores, descending=True)
                pred_sorted = pred_boxes[order]
                pred_xyxy = box_cxcywh_to_xyxy(pred_sorted.clamp(0, 1))
                iou_mat = compute_iou_matrix(pred_xyxy, gt_xyxy)
                if iou_mat.numel() == 0:
                    det_match_miss += 1
                    continue

                top1_best_iou = float(iou_mat[0].max().item())
                best_per_prop = iou_mat.max(dim=1)[0]
                has_match = bool((best_per_prop >= iou_thresh).any().item())

                if top1_best_iou >= iou_thresh:
                    for g_idx, rec in enumerate(recs):
                        if iou_mat[0, g_idx] >= iou_thresh:
                            detected_ann_ids.add(rec.ann_id)

                if not has_match:
                    det_match_miss += 1
                    continue

                det_match_ok += 1
                miou_sum += top1_best_iou
                density_stats[dbin]["miou_sum"] += top1_best_iou
                density_stats[dbin]["n_matched"] = \
                    density_stats[dbin].get("n_matched", 0) + 1
                if cat_name in ("crop", "weed"):
                    density_stats[dbin][f"{cat_name}_miou_sum"] += top1_best_iou
                    density_stats[dbin][f"{cat_name}_n_matched"] = \
                        density_stats[dbin].get(f"{cat_name}_n_matched", 0) + 1

                if top1_best_iou >= iou_thresh:
                    hit_r1 += 1

                k = min(top_k, pred_sorted.shape[0])
                topk_ious = iou_mat[:k]
                topk_max_per_gt = topk_ious.max(dim=0)[0]
                if (topk_max_per_gt >= iou_thresh).any().item():
                    hit_rk += 1

                inst_retrieved = int((topk_max_per_gt >= iou_thresh).sum().item())
                inst_hit_rk += inst_retrieved
                density_stats[dbin]["inst_hit_rk"] += inst_retrieved
                if cat_name in ("crop", "weed"):
                    density_stats[dbin][f"{cat_name}_inst_hit_rk"] += inst_retrieved

                kcs = (cat_name, size_label)
                if kcs not in cat_size_stats:
                    cat_size_stats[kcs] = {"total": 0, "r1": 0, "rk": 0, "iou_sum": 0.0}
                cs = cat_size_stats[kcs]
                cs["total"] += 1
                cs["iou_sum"] += top1_best_iou
                if top1_best_iou >= iou_thresh:
                    cs["r1"] += 1
                if (topk_max_per_gt >= iou_thresh).any().item():
                    cs["rk"] += 1

            # ── N-Acc (test split only) ──
            if split == "test" and manipulated_recs:
                orig_map: Dict[str, List[SentenceRecord]] = {}
                for rec in sent_recs:
                    s = rec.sentence.strip().lower()
                    if s:
                        orig_map.setdefault(s, []).append(rec)

                manip_for_nacc: List[SentenceRecord] = []
                for mr in manipulated_recs:
                    if not isinstance(mr, SentenceRecord):
                        continue
                    if mr.ann_id not in detected_ann_ids:
                        continue
                    mt_l = mr.sentence.strip().lower()
                    if not mt_l:
                        continue
                    if mt_l in orig_map and any(
                            r.ann_id != mr.ann_id for r in orig_map[mt_l]):
                        continue
                    manip_for_nacc.append(mr)

                for mr in manip_for_nacc:
                    mt = mr.sentence.strip()
                    if not mt:
                        continue
                    try:
                        cd = mr.change_detail \
                            if isinstance(mr.change_detail, dict) else {}
                        cd_key = json.dumps(cd, sort_keys=True)
                    except Exception:
                        cd_key = "{}"
                    key = (int(image_id), int(mr.ann_id), mt.lower(),
                           str(mr.change_type or ""), cd_key)
                    if key in seen_manip_keys:
                        continue
                    seen_manip_keys.add(key)

                    ct = mr.change_type
                    cd = mr.change_detail or {}
                    attr = (cd.get("attribute", "") if isinstance(cd, dict) else "")
                    if ct == "replace" and attr == "category":
                        group = "Replace_Category"
                    elif ct == "swap" and attr == "size":
                        group = "Swap_Size"
                    elif ct == "swap" and attr == "position":
                        group = "Swap_Position"
                    else:
                        group = f"{ct}_{attr}" if ct and attr else "other"

                    n_acc_total += 1
                    manip_stats.setdefault(group, {"total": 0, "correct": 0})
                    manip_stats[group]["total"] += 1

                    original_gt = mr.gt_box_cxcywh
                    if original_gt is None:
                        n_acc_total -= 1
                        manip_stats[group]["total"] -= 1
                        continue

                    # GIoU-based rejection: correct if top-1 prediction on
                    # manipulated text has GIoU <= 0 with the original GT.
                    gt_single = torch.tensor([original_gt], device=device,
                                             dtype=torch.float32)
                    gt_single_xyxy = box_cxcywh_to_xyxy(gt_single)

                    incorrect = False
                    try:
                        pred_m, scores_m = model_wrapper.predict(
                            image, mt, image_id=image_id)
                        if pred_m is not None and pred_m.numel() > 0:
                            scores_m = scores_m.to(device)
                            top1_idx = torch.argmax(scores_m)
                            top1_box = pred_m[top1_idx:top1_idx+1].to(device)
                            top1_xyxy = box_cxcywh_to_xyxy(top1_box.clamp(0, 1))
                            manip_giou = compute_giou(top1_xyxy, gt_single_xyxy)
                            incorrect = (manip_giou > 0)
                    except Exception:
                        incorrect = False

                    if not incorrect:
                        n_acc_correct += 1
                        manip_stats[group]["correct"] += 1

        elapsed = time.time() - batch_start
        dm = max(det_match_ok, 1)
        dt = max(total_sent, 1)
        msg = (
            f"[eval][batch {batch_idx}] {elapsed:.1f}s | "
            f"n_sent={total_sent} matched={det_match_ok} missed={det_match_miss} | "
            f"Cov={det_match_ok/dt*100:.1f}% R@1={hit_r1/dm*100:.1f}% "
            f"R@{top_k}={hit_rk/dm*100:.1f}% IR={inst_hit_rk/max(inst_total,1)*100:.1f}% "
            f"mIoU={miou_sum/dm*100:.1f}%"
        )
        if n_acc_total > 0:
            msg += f" | N-Acc={n_acc_correct/n_acc_total*100:.1f}% " \
                   f"({n_acc_correct}/{n_acc_total})"
        print(msg)

    # ── Build results dict ──
    dm = max(det_match_ok, 1)
    dt = max(total_sent, 1)

    top1_raw = hit_r1 / dm
    topk_raw = hit_rk / dm
    inst_ret = inst_hit_rk / max(inst_total, 1)
    top1_uncond = hit_r1 / dt
    topk_uncond = hit_rk / dt
    results = {
        "split": split, "model": args.model,
        "iou_thresh": iou_thresh, "top_k": top_k,
        "total_sentences": total_sent, "matched": det_match_ok,
        "missed": det_match_miss,
        "Coverage": det_match_ok / dt,
        "Top1": top1_raw,
        f"Top{top_k}": topk_raw,
        "Top1_uncond": top1_uncond,
        f"Top{top_k}_uncond": topk_uncond,
        "InstRet": inst_ret,
        "mIoU": miou_sum / dm,
    }

    if n_acc_total > 0:
        results["N_Acc"] = n_acc_correct / n_acc_total
        results["N_Acc_n"] = n_acc_total
        results["N_Acc_correct"] = n_acc_correct
        results["N_Acc_method"] = "GIoU<=0"
        results["N_Acc_avg_paper"] = n_acc_correct / NACC_PAPER_DENOM
        results["N_Acc_paper_denom"] = NACC_PAPER_DENOM

    # Table 3: Category × Size
    t3 = {}
    for (cat, sz), s in cat_size_stats.items():
        n = s["total"]
        if n > 0:
            t3[f"{cat}_{sz}"] = {
                "n": n, "Top1": s["r1"] / n, f"Top{top_k}": s["rk"] / n,
                "mIoU": s["iou_sum"] / n,
            }
    for cat in ["crop", "weed"]:
        ct = cr1 = crk = 0
        ciou = 0.0
        for (c, _), s in cat_size_stats.items():
            if c == cat and s["total"] > 0:
                ct += s["total"]; cr1 += s["r1"]; crk += s["rk"]
                ciou += s["iou_sum"]
        if ct > 0:
            t3[f"{cat}_Avg"] = {
                "n": ct, "Top1": cr1 / ct, f"Top{top_k}": crk / ct,
                "mIoU": ciou / ct,
            }
    results["table3"] = t3

    # Table 4: Scene Density
    t4 = {}
    for db, ds in density_stats.items():
        e = {"n_sent": ds["n_sent"]}
        e["InstRet"] = ds["inst_hit_rk"] / max(ds["inst_total"], 1)
        e["mIoU"] = ds["miou_sum"] / max(ds.get("n_matched", 0), 1)
        for pf in ["crop", "weed"]:
            e[f"{pf}_InstRet"] = ds.get(f"{pf}_inst_hit_rk", 0) / \
                                 max(ds.get(f"{pf}_inst_total", 0), 1)
            e[f"{pf}_mIoU"] = ds.get(f"{pf}_miou_sum", 0) / \
                              max(ds.get(f"{pf}_n_matched", 0), 1)
        t4[db] = e
    t4["Overall"] = {
        "InstRet": inst_hit_rk / max(inst_total, 1),
        "mIoU": miou_sum / dm,
    }
    results["table4"] = t4

    # Table 5: Manipulation Type N-Acc
    t5 = {}
    for g, ms in manip_stats.items():
        n = ms["total"]
        if n > 0:
            t5[g] = {"n": n, "N_Acc": ms["correct"] / n}
    if n_acc_total > 0:
        t5["Avg"] = {"n": n_acc_total, "N_Acc": n_acc_correct / n_acc_total}
        t5["Avg_paper"] = {"n": NACC_PAPER_DENOM,
                           "N_Acc": n_acc_correct / NACC_PAPER_DENOM}
    results["table5"] = t5

    return results


def print_summary(results: Dict[str, Any]):
    m = results.get("model", "").upper()
    sp = results["split"]
    tk = results.get("top_k", 5)
    print(f"\n{'='*60}")
    print(f"  {m} — {sp} set results")
    print(f"{'='*60}")

    print(f"\n--- Table 2 (paper format: unconditional metrics) ---")
    t1u  = results.get('Top1_uncond', results['Top1'] * results['Coverage']) * 100
    tku  = results.get(f'Top{tk}_uncond', results.get(f'Top{tk}', 0) * results['Coverage']) * 100
    print(f"  Top-1     (uncond): {t1u:.2f}%   [paper's Top-1]")
    print(f"  Top-{tk}     (uncond): {tku:.2f}%   [paper's Top-{tk}]")
    print(f"  R@0.5 / InstRet:    {results['InstRet']*100:.2f}%   [paper's R@0.5]")
    print(f"  mIoU:               {results['mIoU']*100:.2f}%   [paper's mIoU]")
    if "N_Acc_avg_paper" in results:
        nacc_paper = results["N_Acc_avg_paper"]
        denom = results.get("N_Acc_paper_denom", 3616)
        print(f"  Neg-Acc (paper Avg):{nacc_paper*100:.2f}%  "
              f"[correct={results['N_Acc_correct']}/{denom}]")
    print(f"  --- (internals) ---")
    print(f"  Coverage:           {results['Coverage']*100:.2f}%")
    print(f"  Top-1  (cond):      {results['Top1']*100:.2f}%  [over matched only]")
    print(f"  Top-{tk}  (cond):      {results[f'Top{tk}']*100:.2f}%  [over matched only]")
    if "N_Acc" in results:
        print(f"  N-Acc (per-elig):   {results['N_Acc']*100:.2f}%  "
              f"[correct={results['N_Acc_correct']}/model_elig={results['N_Acc_n']}]")

    print(f"\n--- Table 3: Category x Size ---")
    t3 = results.get("table3", {})
    for cat in ["crop", "weed"]:
        print(f"\n  {cat.capitalize()}:")
        for sz in ["Tiny", "Small", "Med", "Large", "Avg"]:
            k = f"{cat}_{sz}"
            if k in t3:
                s = t3[k]
                print(f"    {sz:6s}: Top-1={s['Top1']*100:5.1f}%  "
                      f"mIoU={s['mIoU']*100:5.1f}%  (n={s['n']})")

    print(f"\n--- Table 4: Scene Density ---")
    t4 = results.get("table4", {})
    for db in ["1-10", "11-20", "21-30", ">30", "Overall"]:
        if db in t4:
            d = t4[db]
            if db != "Overall":
                print(f"  {db:>10s}: Crop R@0.5={d.get('crop_InstRet',0)*100:5.1f}% "
                      f"mIoU={d.get('crop_mIoU',0)*100:5.1f}%  |  "
                      f"Weed R@0.5={d.get('weed_InstRet',0)*100:5.1f}% "
                      f"mIoU={d.get('weed_mIoU',0)*100:5.1f}%")
            else:
                print(f"  {'Overall':>10s}: R@0.5={d['InstRet']*100:5.1f}%  "
                      f"mIoU={d['mIoU']*100:5.1f}%")

    if results.get("table5"):
        print(f"\n--- Table 5: N-Acc by Manipulation Type ---")
        for g in ["Replace_Category", "Swap_Size", "Swap_Position", "Avg"]:
            if g in results["table5"]:
                s = results["table5"][g]
                print(f"  {g:20s}: N-Acc={s['N_Acc']*100:5.1f}% (n={s['n']})")
        if "Avg_paper" in results["table5"]:
            s = results["table5"]["Avg_paper"]
            print(f"  {'Avg_paper':20s}: N-Acc={s['N_Acc']*100:5.1f}% (n={s['n']})  [paper denom=3616]")


# ── Main ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Evaluate MDETR/SAM3 baselines on gRef-CW")
    p.add_argument("--model", required=True, choices=["mdetr", "sam3"])
    p.add_argument("--split", required=True, choices=["val", "test"])
    p.add_argument("--grefs-json", required=True,
                   help="Path to grefs(unc).json")
    p.add_argument("--instances-json", required=True,
                   help="Path to instances.json")
    p.add_argument("--images-root", required=True,
                   help="Path to image directory")
    p.add_argument("--output-dir", default="eval_baselines")
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch-size", default=4, type=int)
    p.add_argument("--limit", default=None, type=int,
                   help="Limit number of images (for debugging)")
    p.add_argument("--mdetr-root", default="mdetr",
                   help="Path to cloned MDETR repository")
    p.add_argument("--mdetr-variant", default="refcoco",
                   choices=["pretrained", "refcoco", "refcocoplus", "refcocog"])
    p.add_argument("--sam3-bpe-path", default=None,
                   help="Path to SAM3 BPE vocab file (auto-detected if omitted)")
    p.add_argument("--iou-thresh", default=0.5, type=float)
    p.add_argument("--top-k", default=5, type=int)
    return p.parse_args()


def main():
    args = parse_args()
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
        print("[warn] CUDA not available, using CPU")
    args.device = device
    os.makedirs(args.output_dir, exist_ok=True)
    if device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    print(f"\n[main] Configuration:")
    print(f"  model={args.model}, split={args.split}, device={device}")
    print(f"  iou_thresh={args.iou_thresh}, top_k={args.top_k}")

    dataset = GRefCWDataset(
        args.grefs_json, args.instances_json, args.images_root, args.split)
    if args.limit:
        dataset.items = dataset.items[:args.limit]
        print(f"[main] Limited to {len(dataset)} images")

    print(f"\n[main] Loading model '{args.model}'...")
    if args.model == "mdetr":
        wrapper = MDETRWrapper(
            device=device, variant=args.mdetr_variant,
            mdetr_root=args.mdetr_root)
    else:
        wrapper = SAM3Wrapper(device=device, bpe_path=args.sam3_bpe_path)

    print(f"\n[main] Starting evaluation...")
    t0 = time.time()
    with torch.no_grad():
        if device == "cuda":
            with torch.amp.autocast("cuda", dtype=torch.float16):
                results = evaluate_model(wrapper, dataset, args)
        else:
            results = evaluate_model(wrapper, dataset, args)
    elapsed = time.time() - t0
    results["elapsed_seconds"] = elapsed
    print(f"\n[main] Evaluation completed in {elapsed:.1f}s")

    out = os.path.join(args.output_dir,
                       f"{args.model}_{args.split}_metrics.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[main] Results saved to {out}")
    print_summary(results)


if __name__ == "__main__":
    main()
