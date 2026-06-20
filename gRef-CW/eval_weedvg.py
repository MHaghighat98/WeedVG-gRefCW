"""gRef-CW evaluator for Weed-VG and GroundingDINO."""

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

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

from transformers import AutoTokenizer, AutoModel

try:
    from groundingdino.datasets.transforms import Compose, Normalize, RandomResize, ToTensor
    from groundingdino.hierarchical.grounding_dino_wrapper import (
        wrap_grounding_dino,
        extract_visual_features_at_boxes,
    )
    from groundingdino.models import build_model
    from groundingdino.models.GroundingDINO.weedvg import (
        TextProjectionHead,
        VisualProjectionHead,
        build_weedvg,
    )
    from groundingdino.util.box_ops import box_cxcywh_to_xyxy, box_iou
    from groundingdino.util.misc import clean_state_dict
    from groundingdino.util.slconfig import SLConfig
    from groundingdino.util.inference import preprocess_caption
except ImportError as e:
    raise RuntimeError(f"GroundingDINO imports failed: {e}") from e

from groundingdino.hierarchical.relevance_scorer import (
    ContrastiveRelevanceScorer,
    sort_and_trim_proposals,
)


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


def density_bin(n: int) -> str:
    if n <= 10:
        return "1-10"
    if n <= 20:
        return "11-20"
    if n <= 30:
        return "21-30"
    return ">30"


def compute_giou(box1_xyxy: torch.Tensor, box2_xyxy: torch.Tensor) -> float:
    """Compute GIoU between two single boxes (each [1,4] in xyxy format).
    Returns a float in [-1, 1].  GIoU <= 0 means negligible or no overlap."""
    x1 = max(float(box1_xyxy[0, 0]), float(box2_xyxy[0, 0]))
    y1 = max(float(box1_xyxy[0, 1]), float(box2_xyxy[0, 1]))
    x2 = min(float(box1_xyxy[0, 2]), float(box2_xyxy[0, 2]))
    y2 = min(float(box1_xyxy[0, 3]), float(box2_xyxy[0, 3]))
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    a1 = max(0.0, float(box1_xyxy[0, 2] - box1_xyxy[0, 0])) * max(0.0, float(box1_xyxy[0, 3] - box1_xyxy[0, 1]))
    a2 = max(0.0, float(box2_xyxy[0, 2] - box2_xyxy[0, 0])) * max(0.0, float(box2_xyxy[0, 3] - box2_xyxy[0, 1]))
    union = a1 + a2 - inter
    iou = inter / max(union, 1e-8)
    # Enclosing box
    ex1 = min(float(box1_xyxy[0, 0]), float(box2_xyxy[0, 0]))
    ey1 = min(float(box1_xyxy[0, 1]), float(box2_xyxy[0, 1]))
    ex2 = max(float(box1_xyxy[0, 2]), float(box2_xyxy[0, 2]))
    ey2 = max(float(box1_xyxy[0, 3]), float(box2_xyxy[0, 3]))
    enclose_area = max(0.0, ex2 - ex1) * max(0.0, ey2 - ey1)
    giou = iou - (enclose_area - union) / max(enclose_area, 1e-8)
    return float(giou)



class InferenceTextProjector(torch.nn.Module):
    """Tokenizer + encoder + projection head wrapper for inference."""

    def __init__(self, tokenizer, bert_model, proj_head, device: torch.device):
        super().__init__()
        self.tokenizer = tokenizer
        self.bert = bert_model.to(device)
        self.proj = proj_head.to(device)
        self.device = device

    @torch.no_grad()
    def encode_many(self, texts: List[str]) -> torch.Tensor:
        if not texts:
            raise ValueError("encode_many() got empty texts")
        tok = self.tokenizer(list(texts), padding=True, truncation=True, max_length=64, return_tensors="pt")
        tok = {k: v.to(self.device) for k, v in tok.items()}
        out = self.bert(**tok)
        pooled = getattr(out, "pooler_output", None)
        if pooled is None:
            last = getattr(out, "last_hidden_state", None)
            if last is None:
                raise RuntimeError("BERT returned no usable embeddings")
            attn = tok.get("attention_mask", None)
            if attn is None:
                pooled = last.mean(dim=1)
            else:
                attn = attn.to(dtype=last.dtype)
                special_mask = torch.zeros_like(attn)
                special_mask[:, 0] = 1.0
                lengths = attn.sum(dim=1).to(dtype=torch.long)
                for i in range(attn.shape[0]):
                    li = int(lengths[i].item())
                    if li > 1:
                        special_mask[i, li - 1] = 1.0
                token_mask = (attn * (1.0 - special_mask)).unsqueeze(-1)
                denom = token_mask.sum(dim=1).clamp(min=1.0)
                pooled = (last * token_mask).sum(dim=1) / denom
        proj = self.proj(pooled)
        proj = F.normalize(proj, p=2, dim=-1)
        return proj

    @torch.no_grad()
    def encode_tokens(self, texts: List[str]):
        tok = self.tokenizer(list(texts), padding=True, truncation=True, max_length=64, return_tensors="pt")
        tok = {k: v.to(self.device) for k, v in tok.items()}
        out = self.bert(**tok)
        last_hidden = out.last_hidden_state
        token_proj = self.proj(last_hidden)
        pad_mask = tok["attention_mask"] == 0
        return token_proj, pad_mask


# ── Data Helpers ─────────────────────────────────────────────────────────────

def _ensure_cxcywh_norm(bbox_xywh: List[float], width: int, height: int) -> List[float]:
    x, y, w, h = [float(v) for v in bbox_xywh]
    cx = x + w / 2.0
    cy = y + h / 2.0
    if width <= 0 or height <= 0:
        return [0.5, 0.5, 0.01, 0.01]
    return [cx / float(width), cy / float(height), w / float(width), h / float(height)]




# ── Model Loading ────────────────────────────────────────────────────────────

def load_grounding_dino(config_file, checkpoint_path, base_checkpoint_path, device, *, decoder_layers=None):
    """Load GroundingDINO, optionally layering finetuned weights on base."""
    args = SLConfig.fromfile(config_file)
    if decoder_layers is not None:
        decoder_layers = int(decoder_layers)
        if decoder_layers <= 0:
            raise ValueError("decoder_layers must be > 0")
        setattr(args, "dec_layers", decoder_layers)
        print(f"=> Overriding GroundingDINO dec_layers={decoder_layers}")
    args.device = device
    model = build_model(args)

    def _extract_best_matching_substate(full_state):
        model_keys = set(model.state_dict().keys())
        if not isinstance(full_state, dict) or not full_state:
            return {}
        prefixes = [
            "", "model.", "module.", "module.model.",
            "grounding_dino.", "grounding_dino.model.",
            "module.grounding_dino.", "module.grounding_dino.model.",
        ]
        best, best_matches, best_prefix = {}, -1, ""
        for pref in prefixes:
            out, matches = {}, 0
            for k, v in full_state.items():
                if not isinstance(k, str) or not k.startswith(pref):
                    continue
                k2 = k[len(pref):]
                if k2 in model_keys:
                    out[k2] = v
                    matches += 1
            if matches > best_matches:
                best_matches, best, best_prefix = matches, out, pref
        if best_matches > 0:
            print(f"=> Resolved detector weights using prefix '{best_prefix}' (matched {best_matches} keys)")
        return best

    def _load_one(path, label):
        print(f"=> Loading {label} checkpoint from '{path}'")
        ckpt = torch.load(path, map_location="cpu")
        try:
            ep = ckpt.get("epoch")
            it = ckpt.get("iter")
            if ep is not None or it is not None:
                print(f"=> {label} checkpoint meta: epoch={ep} iter={it}")
        except Exception:
            pass
        if isinstance(ckpt, dict) and isinstance(ckpt.get("model"), dict):
            state = ckpt["model"]
        elif isinstance(ckpt, dict):
            # Accept both current and legacy key names for the model state dict
            inner = ckpt.get("weedvg_state_dict") or ckpt.get("agrivg_state_dict")
            if isinstance(inner, dict):
                extracted = _extract_best_matching_substate(inner)
                state = extracted if extracted else inner
            else:
                state = ckpt
        else:
            state = ckpt
        missing, unexpected = model.load_state_dict(clean_state_dict(state), strict=False)
        print(f"=> {label} load_state_dict: missing={len(missing)} unexpected={len(unexpected)}")
        return missing, unexpected

    if base_checkpoint_path is not None:
        base_checkpoint_path = str(base_checkpoint_path)
        if base_checkpoint_path and os.path.isfile(base_checkpoint_path):
            _load_one(base_checkpoint_path, "GroundingDINO base")
        else:
            raise FileNotFoundError(f"Base checkpoint not found: {base_checkpoint_path}")

    if checkpoint_path and os.path.isfile(checkpoint_path):
        missing, _ = _load_one(checkpoint_path, "GroundingDINO finetune")
        if base_checkpoint_path is None and len(missing) > 0:
            print("[warn] Detector checkpoint appears partial. Consider --base-detector-checkpoint.")
    else:
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    model.eval()
    return model


def _load_projector_weights(weedvg_model, projector_checkpoint: str):
    ckpt = torch.load(projector_checkpoint, map_location="cpu")
    try:
        ep = ckpt.get("epoch")
        it = ckpt.get("iter")
        if ep is not None or it is not None:
            print(f"=> Projector checkpoint meta: epoch={ep} iter={it}")
    except Exception:
        pass

    visual_sd = ckpt.get("visual_projector_state_dict")
    text_sd_full = ckpt.get("text_projector_state_dict")
    if visual_sd is None:
        raise RuntimeError(f"'visual_projector_state_dict' not found in {projector_checkpoint}")
    if text_sd_full is None:
        raise RuntimeError(f"'text_projector_state_dict' not found in {projector_checkpoint}")

    text_proj_sd, text_enc_sd = {}, {}
    for k, v in text_sd_full.items():
        if k.startswith("projector."):
            text_proj_sd[k[len("projector."):]] = v
        elif k.startswith("encoder."):
            text_enc_sd[k[len("encoder."):]] = v
    if not text_proj_sd:
        text_proj_sd = text_sd_full

    if weedvg_model.visual_projector is None:
        raise RuntimeError("visual_projector is None on weedvg_model")
    if weedvg_model.text_projector is None or not hasattr(weedvg_model.text_projector, "proj"):
        raise RuntimeError("text_projector is missing or not InferenceTextProjector")

    if text_enc_sd:
        try:
            missing_e, unexpected_e = weedvg_model.text_projector.bert.load_state_dict(text_enc_sd, strict=False)
            if missing_e or unexpected_e:
                print(f"[warn] text_encoder(bert) load: missing={len(missing_e)} unexpected={len(unexpected_e)}")
            else:
                print("=> Loaded text encoder (BERT) weights from checkpoint")
        except Exception as exc:
            print(f"[warn] Failed to load text encoder weights: {exc}")

    missing_v, unexpected_v = weedvg_model.visual_projector.load_state_dict(visual_sd, strict=False)
    missing_t, unexpected_t = weedvg_model.text_projector.proj.load_state_dict(text_proj_sd, strict=False)
    if missing_v or unexpected_v:
        print(f"[warn] visual_projector load: missing={len(missing_v)} unexpected={len(unexpected_v)}")
    if missing_t or unexpected_t:
        print(f"[warn] text_projector load: missing={len(missing_t)} unexpected={len(unexpected_t)}")


# ── Data Structures ──────────────────────────────────────────────────────────

@dataclass
class SentenceRecord:
    ann_id: int
    category_id: Optional[int]
    sentence: str
    bbox_xywh: List[float]  # absolute xywh (COCO)
    gt_box: Optional[List[float]] = None  # cxcywh normalized
    change_type: Optional[str] = None
    change_detail: Optional[Dict[str, Any]] = None
    is_manipulated: bool = False


# ── Dataset ──────────────────────────────────────────────────────────────────

class GRefsEvalDataset(Dataset):
    def __init__(self, grefs_json, instances_json, images_root, splits, transform, limit=None):
        self.images_root = images_root
        self.transform = transform

        with open(grefs_json, "r", encoding="utf-8") as f:
            grefs = json.load(f)
        with open(instances_json, "r", encoding="utf-8") as f:
            coco = json.load(f)

        catid_to_name = {}
        for cat in coco.get("categories", []) or []:
            try:
                cid = int(cat.get("id"))
            except Exception:
                continue
            nm = cat.get("name")
            if isinstance(nm, str) and nm.strip():
                catid_to_name[cid] = nm.strip()
        self.catid_to_name = catid_to_name

        annid_to_bbox = {}
        for ann in coco.get("annotations", []) or []:
            ann_id = ann.get("id")
            if ann_id is None:
                continue
            bbox = ann.get("bbox")
            if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
                continue
            annid_to_bbox[int(ann_id)] = [float(v) for v in bbox]
        self.annid_to_bbox = annid_to_bbox

        items = []
        manipulated_count = 0
        split_counts = {}
        for img in grefs.get("images", []) or []:
            split = str(img.get("split", "")).strip() or "unspecified"
            if splits and split not in splits:
                continue

            image_id = img.get("id")
            file_name = img.get("file_name")
            if image_id is None or not isinstance(file_name, str) or not file_name:
                continue

            w = int(img.get("width", 0) or 0)
            h = int(img.get("height", 0) or 0)

            sent_recs = []
            manipulated_recs = []
            for rec in img.get("instance_sentences", []) or []:
                ann_id = rec.get("ann_id")
                category_id = rec.get("category_id", None)
                change_type = rec.get("change_type")
                change_detail = rec.get("change_detail")
                test_sentence = rec.get("test_sentence")
                original_sentence = rec.get("original_sentence") or rec.get("sentence")

                sentence = rec.get("sentence")
                if not isinstance(sentence, str) or not sentence.strip():
                    sentence = original_sentence

                if ann_id is None:
                    continue
                bbox = self.annid_to_bbox.get(int(ann_id))
                if bbox is None:
                    continue

                if isinstance(sentence, str) and sentence.strip():
                    sent_recs.append(SentenceRecord(
                        ann_id=int(ann_id),
                        category_id=(int(category_id) if category_id is not None else None),
                        sentence=sentence.strip(),
                        bbox_xywh=[float(v) for v in bbox],
                    ))

                if change_type and isinstance(test_sentence, str) and test_sentence.strip():
                    manipulated_recs.append(SentenceRecord(
                        ann_id=int(ann_id),
                        category_id=(int(category_id) if category_id is not None else None),
                        sentence=test_sentence.strip(),
                        bbox_xywh=[float(v) for v in bbox],
                        change_type=change_type,
                        change_detail=change_detail,
                        is_manipulated=True,
                    ))

            manipulated_count += len(manipulated_recs)
            if split not in split_counts:
                split_counts[split] = {"images": 0, "sentences": 0, "manipulated": 0}
            split_counts[split]["images"] += 1
            split_counts[split]["sentences"] += len(sent_recs)
            split_counts[split]["manipulated"] += len(manipulated_recs)

            items.append({
                "split": split,
                "image_id": int(image_id),
                "file_name": file_name,
                "width": w, "height": h,
                "sentences": sent_recs,
                "manipulated_sentences": manipulated_recs,
            })

            if limit is not None and len(items) >= int(limit):
                break

        self.items = items
        self.manipulated_count = int(manipulated_count)
        self.split_counts = split_counts

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        path = os.path.join(self.images_root, item["file_name"])
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Image not found: {path}")

        with Image.open(path) as img:
            img = img.convert("RGB")
            w_img, h_img = img.size
            image_tensor, _ = self.transform(img, None)

        sentences_norm = []
        for rec in item.get("sentences", []) or []:
            if not isinstance(rec, SentenceRecord):
                continue
            try:
                gt_box = _ensure_cxcywh_norm(rec.bbox_xywh, int(w_img), int(h_img))
            except Exception:
                continue
            sentences_norm.append(SentenceRecord(
                ann_id=int(rec.ann_id),
                category_id=(int(rec.category_id) if rec.category_id is not None else None),
                sentence=str(rec.sentence),
                bbox_xywh=[float(v) for v in rec.bbox_xywh],
                gt_box=gt_box,
            ))

        manipulated_norm = []
        for rec in item.get("manipulated_sentences", []) or []:
            if not isinstance(rec, SentenceRecord):
                continue
            try:
                gt_box = _ensure_cxcywh_norm(rec.bbox_xywh, int(w_img), int(h_img))
            except Exception:
                gt_box = None
            manipulated_norm.append(SentenceRecord(
                ann_id=int(rec.ann_id),
                category_id=(int(rec.category_id) if rec.category_id is not None else None),
                sentence=str(rec.sentence),
                bbox_xywh=[float(v) for v in rec.bbox_xywh],
                gt_box=gt_box,
                change_type=rec.change_type,
                change_detail=rec.change_detail,
                is_manipulated=True,
            ))

        return {
            "split": item["split"],
            "image_id": item["image_id"],
            "file_name": item["file_name"],
            "width": item["width"], "height": item["height"],
            "image": image_tensor,
            "sentences": sentences_norm,
            "manipulated_sentences": manipulated_norm,
        }


def collate_fn(batch):
    images = [b["image"] for b in batch]
    return images, batch


# ── Argument Parsing ─────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser("Weed-VG evaluation (two-stage + vanilla GDino)")
    p.add_argument("--config-file", required=True, type=str)
    p.add_argument("--base-detector-checkpoint", default=None, type=str,
                   help="Optional FULL GroundingDINO checkpoint.")
    p.add_argument("--detector-checkpoint", default=None, type=str,
                   help="Detector checkpoint (e.g. stage_one.pth).")
    p.add_argument("--projector-checkpoint", default=None, type=str,
                   help="Projector checkpoint (required for two-stage mode).")
    p.add_argument("--grefs-json", required=True, type=str)
    p.add_argument("--instances-json", required=True, type=str)
    p.add_argument("--images-root", required=True, type=str)
    p.add_argument("--output-dir", default="eval_two_stage", type=str)
    p.add_argument("--device", default="cuda", type=str)
    p.add_argument("--batch-size", default=4, type=int)
    p.add_argument("--num-workers", default=0, type=int)
    p.add_argument("--splits", default="val,test", type=str)
    p.add_argument("--decoder-layers", default=None, type=int)
    p.add_argument("--max-proposals", default=900, type=int)
    p.add_argument("--grounding-prompt", default="plant or vegetation", type=str,
                   help="Detector prompt when --detect-with-generic-prompt is set.")
    p.add_argument("--iou-thresh", default=0.5, type=float,
                   help="IoU threshold for matching/retrieval metrics (default: 0.5)")
    p.add_argument("--top-k", default=5, type=int)
    p.add_argument("--nacc-iou-thresh", default=0.5, type=float,
                   help="IoU threshold for N-Acc eligibility and rejection (default: 0.5)")
    p.add_argument("--nacc-score-thresh", default=None, type=float,
                   help="CRS score threshold for N-Acc in two-stage mode. "
                        "If the top CRS score for manipulated text is below this, "
                        "the model abstains → correct rejection. Default: None (disabled).")
    p.add_argument("--text-threshold", default=0.25, type=float,
                   help="GroundingDINO text threshold (default: 0.25)")
    p.add_argument("--box-threshold", default=0.25, type=float,
                   help="GroundingDINO box threshold for filtering detections (default: 0.25)")
    p.add_argument("--proj-dim", default=768, type=int)
    p.add_argument("--temperature", default=0.07, type=float)
    p.add_argument("--text-encoder-name", default="bert-base-uncased", type=str)
    p.add_argument("--detect-with-generic-prompt", action="store_true",
                   help="Use --grounding-prompt for detector instead of per-sentence prompts.")
    p.add_argument("--vanilla-gdino", action="store_true",
                   help="Vanilla GroundingDINO evaluation (no CRS/projector). "
                        "Uses detector scores directly, same metric logic as eval_baselines_v3.py.")
    p.add_argument("--scoring-mode", default="full", type=str,
                   choices=["full", "sentence_only", "word_only", "no_proj"],
                   help="Ablation scoring mode: full (default), sentence_only, word_only, no_proj")
    p.add_argument("--model-label", default=None, type=str,
                   help="Model label for results JSON (e.g. 'gdino_swinT', 'gdino_swinB'). "
                        "Auto-detected if not set.")
    p.add_argument("--no-amp", action="store_true",
                   help="Disable FP16 autocast (run in FP32). Use this if your GPU "
                        "produces different FP16 results than the paper hardware.")
    p.add_argument("--limit", default=None, type=int)
    return p.parse_args()


# ── Vanilla GDino Evaluation ─────────────────────────────────────────────────

def evaluate_vanilla_gdino(model, dataset, args):
    """
    Evaluate vanilla GroundingDINO (no CRS/projector).
    Uses the same metric logic as eval_baselines_v3.py:
      - Run model(image, sentence) → proposals sorted by score
      - Compute IoU with GT
      - Coverage, R@1, R@K, Inst.Ret, mIoU, N-Acc
    """
    device = torch.device(args.device)
    iou_thresh = float(args.iou_thresh)
    nacc_iou_thresh = float(args.nacc_iou_thresh)
    top_k = max(1, int(args.top_k))
    box_threshold = float(args.box_threshold)
    text_threshold = float(args.text_threshold)
    splits = [s.strip() for s in str(args.splits).split(",") if s.strip()]

    loader = DataLoader(dataset, batch_size=1, shuffle=False,
                        num_workers=0, collate_fn=collate_fn)

    # ── Accumulators ──
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

    cat_size_stats = {}
    density_stats = {}
    manip_stats = {}

    img_count = 0

    def _predict_vanilla(img_tensor, text: str):
        """Run vanilla GDino and return (pred_boxes_cxcywh_norm, scores) sorted by score desc."""
        caption = preprocess_caption(text)
        with torch.no_grad():
            outputs = model(img_tensor.unsqueeze(0).to(device), captions=[caption])

        prediction_logits = outputs["pred_logits"].cpu().sigmoid()[0]  # [nq, 256]
        prediction_boxes = outputs["pred_boxes"].cpu()[0]  # [nq, 4] cxcywh norm

        # Filter by box_threshold
        max_logits = prediction_logits.max(dim=1)[0]  # [nq]
        mask = max_logits > box_threshold
        boxes = prediction_boxes[mask]  # [n, 4]
        scores = max_logits[mask]  # [n]

        if boxes.numel() == 0:
            return torch.zeros((0, 4), device=device), torch.zeros((0,), device=device)

        # Sort by score desc
        order = torch.argsort(scores, descending=True)
        boxes = boxes[order].to(device)
        scores = scores[order].to(device)
        return boxes, scores

    for batch_idx, (images, items) in enumerate(loader):
        batch_start = time.time()
        for bi, item in enumerate(items):
            detected_ann_ids = set()
            img_count += 1
            img_tensor = images[bi].to(device)
            sent_recs = [r for r in (item.get("sentences", []) or [])
                         if isinstance(r, SentenceRecord) and r.gt_box is not None and r.sentence]
            manipulated_recs = item.get("manipulated_sentences", []) or []

            # Group sentences by unique text
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
            dbin = density_bin(image_gt_count) if image_gt_count > 0 else "1-10"
            if dbin not in density_stats:
                density_stats[dbin] = {
                    k: 0 if "iou" not in k else 0.0
                    for k in [
                        "n_sent", "inst_total", "inst_hit_rk", "miou_sum",
                        "crop_n_sent", "crop_inst_total", "crop_inst_hit_rk", "crop_miou_sum",
                        "weed_n_sent", "weed_inst_total", "weed_inst_hit_rk", "weed_miou_sum",
                    ]
                }

            # ── Per-sentence evaluation (same as eval_baselines_v3.py) ──
            for sent_text in sent_texts:
                recs = sent_to_recs[sent_text]
                total_sent += 1

                gt_boxes = [r.gt_box for r in recs if r.gt_box is not None]
                cat_id = recs[0].category_id
                cat_name = CATEGORY_MAP.get(cat_id, "unknown") if cat_id else "unknown"
                size_label = _extract_size_from_sentence(sent_text) or "unknown"
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

                # Run model
                try:
                    pred_boxes, scores = _predict_vanilla(img_tensor, sent_text)
                except Exception as e:
                    if img_count <= 3:
                        print(f"[warn] predict failed: {e}")
                    det_match_miss += 1
                    continue

                if pred_boxes.numel() == 0:
                    det_match_miss += 1
                    continue

                pred_xyxy = box_cxcywh_to_xyxy(pred_boxes.clamp(0, 1))
                iou_mat = box_iou(pred_xyxy, gt_xyxy)[0]
                if iou_mat.numel() == 0:
                    det_match_miss += 1
                    continue

                # Coverage check
                top1_best_iou = float(iou_mat[0].max().item())
                best_per_prop = iou_mat.max(dim=1)[0]
                has_match = bool((best_per_prop >= iou_thresh).any().item())

                if not has_match:
                    det_match_miss += 1
                    continue

                # ── Matched sentence ──
                det_match_ok += 1

                # mIoU (over MATCHED sentences only)
                miou_sum += top1_best_iou
                density_stats[dbin]["miou_sum"] += top1_best_iou
                density_stats[dbin]["n_matched"] = density_stats[dbin].get("n_matched", 0) + 1
                if cat_name in ("crop", "weed"):
                    density_stats[dbin][f"{cat_name}_miou_sum"] += top1_best_iou
                    density_stats[dbin][f"{cat_name}_n_matched"] = density_stats[dbin].get(f"{cat_name}_n_matched", 0) + 1

                # R@1
                if top1_best_iou >= iou_thresh:
                    hit_r1 += 1
                    # Track detected ann_ids
                    # iou_mat is [P, G]. G corresponds to recs.
                    for g_idx, rec in enumerate(recs):
                         if iou_mat[0, g_idx] >= iou_thresh:
                             detected_ann_ids.add(rec.ann_id)

                # R@K
                k = min(top_k, pred_boxes.shape[0])
                topk_ious = iou_mat[:k]
                topk_max_per_gt = topk_ious.max(dim=0)[0]
                if (topk_max_per_gt >= iou_thresh).any().item():
                    hit_rk += 1

                # Inst. Ret.
                inst_retrieved = int((topk_max_per_gt >= iou_thresh).sum().item())
                inst_hit_rk += inst_retrieved
                density_stats[dbin]["inst_hit_rk"] += inst_retrieved
                if cat_name in ("crop", "weed"):
                    density_stats[dbin][f"{cat_name}_inst_hit_rk"] += inst_retrieved

                # Table 3: Category × Size
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
            # Eligibility is annotation-defined (model-independent): a manipulated sentence is eligible
            # only if it does NOT exactly match any OTHER instance sentence in the same image.
            # Also: model must have detected the original (ann_id in detected_ann_ids).
            # IoU-based: correct rejection = IoU(top-1 on manip text, original GT) < iou_thresh.
            has_test = any(s in splits for s in ["test"])
            if has_test and manipulated_recs:
                original_sentence_to_recs: Dict[str, List[SentenceRecord]] = {}
                for rec in sent_recs:
                    s = rec.sentence.strip().lower()
                    if s:
                        original_sentence_to_recs.setdefault(s, []).append(rec)

                manip_for_nacc: List[SentenceRecord] = []
                for mr in manipulated_recs:
                    if not isinstance(mr, SentenceRecord):
                        continue
                    # Condition 1: Must have detected the original
                    if mr.ann_id not in detected_ann_ids:
                        continue
                    mt_l = mr.sentence.strip().lower()
                    if not mt_l:
                        continue
                    # Condition 2: Truly negative
                    if mt_l in original_sentence_to_recs and any(r.ann_id != mr.ann_id for r in original_sentence_to_recs[mt_l]):
                        continue
                    manip_for_nacc.append(mr)

                for mr in manip_for_nacc:
                    mt = mr.sentence.strip()
                    if not mt:
                        continue
                    try:
                        cd = mr.change_detail if isinstance(mr.change_detail, dict) else {}
                        cd_key = json.dumps(cd, sort_keys=True)
                    except Exception:
                        cd_key = "{}"
                    key = (int(item.get("image_id", -1)), int(mr.ann_id), mt.lower(), str(mr.change_type or ""), cd_key)
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

                    # GIoU-based rejection: run model on manipulated text,
                    # check if top-1 prediction overlaps original GT.
                    # Correct rejection = GIoU(top1_manip, original_GT) <= 0.
                    original_gt = mr.gt_box
                    if original_gt is None:
                        n_acc_total -= 1
                        manip_stats[group]["total"] -= 1
                        continue

                    gt_single = torch.tensor([original_gt], device=device, dtype=torch.float32)
                    gt_single_xyxy = box_cxcywh_to_xyxy(gt_single)

                    incorrect = False
                    try:
                        pred_m, scores_m = _predict_vanilla(img_tensor, mt)
                        if pred_m is not None and pred_m.numel() > 0:
                            top1_xyxy = box_cxcywh_to_xyxy(pred_m[0:1].clamp(0, 1))
                            manip_giou = compute_giou(top1_xyxy, gt_single_xyxy)
                            incorrect = (manip_giou > 0)
                    except Exception:
                        incorrect = False

                    if not incorrect:
                        n_acc_correct += 1
                        manip_stats[group]["correct"] += 1

        # Per-batch progress
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
            msg += f" | N-Acc={n_acc_correct/n_acc_total*100:.1f}% ({n_acc_correct}/{n_acc_total})"
        print(msg)

    # ── Build results ──
    return _build_results(
        args, total_sent, det_match_ok, det_match_miss,
        hit_r1, hit_rk, miou_sum, inst_total, inst_hit_rk,
        n_acc_total, n_acc_correct, seen_manip_keys,
        cat_size_stats, density_stats, manip_stats,
    )


# ── Two-Stage Evaluation ────────────────────────────────────────────────────

def evaluate_two_stage(weedvg_model, dataset, args, detected_visual_dim):
    """
    Two-stage evaluation: GroundingDINO + CRS/Projector.
    Matched-proposal approach for R@1/R@K/IR/mIoU.
    N-Acc: GIoU-based, conditioned on successful grounding.
    For N-Acc, the detector backbone is run with the manipulated text
    as a sentence-conditioned prompt (same approach as vanilla GDino).
    Top-1 proposal is checked for GIoU overlap with original GT.
    """
    device = torch.device(args.device)
    iou_thresh = float(args.iou_thresh)
    nacc_iou_thresh = float(args.nacc_iou_thresh)
    top_k = max(1, int(args.top_k))
    splits = [s.strip() for s in str(args.splits).split(",") if s.strip()]

    loader = DataLoader(dataset, batch_size=int(args.batch_size), shuffle=False,
                        num_workers=int(args.num_workers),
                        pin_memory=(device.type == "cuda"), collate_fn=collate_fn)

    # ── Accumulators ──
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

    cat_size_stats = {}
    density_stats = {}
    manip_stats = {}

    total_images = len(dataset)
    img_count = 0

    def _evaluate_matched_proposals(
        props, visual_features_q, det_scores, gt_t, gt_xyxy,
        iou_mat, best_iou_per_prop, keep_mask,
        text_scorer_fn,
    ):
        """Core metric computation — re-ranks proposals by **CRS score**
        (Contextual Relevance Score from the text-visual projector), then
        applies the standard Top-1/Top-K/mIoU/Inst.Ret. evaluation:
          - Top-1: does the CRS-rank-1 proposal have IoU >= iou_thresh with
            any GT?
          - Top-K: do any of the top-K CRS-ranked predictions hit any GT?
          - Inst. Ret.: how many GT boxes are retrieved by top-K?
          - mIoU: IoU of CRS-rank-1 with best-overlapping GT.
        """
        # Coverage gate: at least one proposal must have IoU >= thresh with GT
        if not bool(keep_mask.any().item()):
            return None

        if props.shape[0] == 0:
            return None

        # ── CRS-based re-ranking ──
        # Use the text_scorer_fn to compute CRS scores for all proposals,
        # then re-order proposals by CRS score (descending).
        crs_scores = text_scorer_fn(visual_features_q, props)  # [N]
        sorted_indices = torch.argsort(crs_scores, descending=True)

        # Re-order iou_mat and best_iou_per_prop by CRS rank
        iou_mat = iou_mat[sorted_indices]
        best_iou_per_prop = best_iou_per_prop[sorted_indices]

        # Top-1 is the CRS-rank-1 proposal (highest CRS score).
        top1_iou = float(best_iou_per_prop[0].item())
        top1_gt_ious = iou_mat[0]  # IoUs of CRS-top-1 box with all GTs

        # Top-1 hit: same criterion as baselines
        r1_hit = (top1_iou >= iou_thresh)

        # Top-K: any of the top-K CRS-ranked predictions hit any GT?
        k = min(top_k, int(props.shape[0]))
        iou_topk = iou_mat[:k]                          # [K, G]
        topk_max_per_gt = iou_topk.max(dim=0)[0]        # [G]
        rk_hit = bool((topk_max_per_gt >= iou_thresh).any().item())

        # Inst. Ret.: how many GT boxes retrieved by top-K?
        inst_rk_delta = int((topk_max_per_gt >= iou_thresh).sum().item())

        return {
            "matched": True,
            "r1_hit": r1_hit,
            "rk_hit": rk_hit,
            "top1_iou": top1_iou,
            "top1_gt_ious": top1_gt_ious,
            "inst_rk_delta": inst_rk_delta,
            "n_kept": int(keep_mask.sum().item()),
        }

    # ── Main Evaluation Loop ──
    for batch_idx, (images, items) in enumerate(loader):
        batch_start = time.time()
        images = [img.to(device) for img in images]

        for bi, item in enumerate(items):
            detected_ann_ids = set()
            img_count += 1
            img_tensor = images[bi]

            sent_recs = [
                rec for rec in (item.get("sentences", []) or [])
                if isinstance(rec, SentenceRecord) and rec.gt_box is not None and rec.sentence
            ]
            manipulated_recs = item.get("manipulated_sentences", []) or []

            if img_count % 100 == 1 or img_count == total_images:
                dm_ = max(det_match_ok, 1)
                dt_ = max(total_sent, 1)
                print(f"[eval] Image {img_count}/{total_images}  "
                      f"matched={det_match_ok} missed={det_match_miss} "
                      f"R@1={hit_r1/dm_*100:.1f}% mIoU={miou_sum/dm_*100:.1f}%")

            # Build sentence → GT records mapping (for N-Acc filtering)
            original_sentence_to_recs: Dict[str, List[SentenceRecord]] = {}
            for rec in sent_recs:
                s = rec.sentence.strip().lower()
                if s:
                    original_sentence_to_recs.setdefault(s, []).append(rec)

            # Partition manipulated sentences for N-Acc
            manipulated_recs_for_nacc = []
            for mr in manipulated_recs:
                if not isinstance(mr, SentenceRecord):
                    continue
                mt = mr.sentence.strip().lower()
                if not mt:
                    continue
                if mt in original_sentence_to_recs:
                    if any(r.ann_id != mr.ann_id for r in original_sentence_to_recs[mt]):
                        continue
                manipulated_recs_for_nacc.append(mr)

            # Group sentences by unique text
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
            dbin = density_bin(image_gt_count) if image_gt_count > 0 else "1-10"
            if dbin not in density_stats:
                density_stats[dbin] = {
                    k: 0 if "iou" not in k else 0.0
                    for k in [
                        "n_sent", "inst_total", "inst_hit_rk", "miou_sum",
                        "crop_n_sent", "crop_inst_total", "crop_inst_hit_rk", "crop_miou_sum",
                        "weed_n_sent", "weed_inst_total", "weed_inst_hit_rk", "weed_miou_sum",
                    ]
                }

            # ── Generic prompt: compute proposals once per image ──
            generic_props = None
            generic_vis_feats = None
            generic_det_scores = None
            if args.detect_with_generic_prompt and sent_texts:
                prompt = str(args.grounding_prompt or "plant or vegetation")
                with torch.no_grad():
                    gd_out_img = weedvg_model.forward_grounding_dino(img_tensor.unsqueeze(0), [prompt])
                pb_img = gd_out_img.get("pred_boxes")
                pl_img = gd_out_img.get("pred_logits")
                if pb_img is not None and pl_img is not None:
                    pb_img, pl_img, ps_img = sort_and_trim_proposals(pb_img, pl_img, int(args.max_proposals))
                    feats_img = extract_visual_features_at_boxes(
                        gd_out_img.get("multi_scale_features") or gd_out_img.get("encoder_features"),
                        pb_img, feature_dim=detected_visual_dim, use_multi_scale=False,
                    )
                    if feats_img is not None and feats_img.numel() > 0:
                        generic_props = pb_img.squeeze(0)
                        generic_vis_feats = feats_img.squeeze(0)
                        generic_det_scores = ps_img.squeeze(0)
                        if batch_idx == 0 and bi == 0:
                            print(f"[eval] (generic) proposals: {generic_props.shape[0]}")

            # ── Per-sentence evaluation ──
            if sent_texts:
                chunk_size = max(1, min(32, len(sent_texts)))
                for offset in range(0, len(sent_texts), chunk_size):
                    chunk_texts = sent_texts[offset:offset + chunk_size]
                    chunk_groups = [sent_to_recs[t] for t in chunk_texts]

                    if args.detect_with_generic_prompt:
                        if generic_props is None or generic_props.numel() == 0:
                            for _ in chunk_texts:
                                total_sent += 1
                                det_match_miss += 1
                                density_stats[dbin]["n_sent"] += 1
                            continue

                        if getattr(weedvg_model, "crs_scorer", None) is not None:
                            token_proj, token_mask = weedvg_model.text_projector.encode_tokens(chunk_texts)
                            token_proj, token_mask = token_proj.to(device), token_mask.to(device)
                        else:
                            text_emb = weedvg_model.text_projector.encode_many(chunk_texts).to(device)
                            text_emb = F.normalize(text_emb, p=2, dim=-1)

                        for sent_idx, recs in enumerate(chunk_groups):
                            total_sent += 1
                            gt_boxes = [r.gt_box for r in recs if r.gt_box is not None]
                            cat_id = recs[0].category_id
                            cat_name = CATEGORY_MAP.get(cat_id, "unknown") if cat_id else "unknown"
                            size_label = _extract_size_from_sentence(chunk_texts[sent_idx]) or "unknown"
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

                            pred_xyxy = box_cxcywh_to_xyxy(generic_props)
                            iou_mat = box_iou(pred_xyxy, gt_xyxy)[0]
                            if iou_mat.numel() == 0:
                                det_match_miss += 1
                                continue

                            best_iou_per_prop = iou_mat.max(dim=1)[0]
                            keep_mask = best_iou_per_prop >= iou_thresh

                            if not bool(keep_mask.any().item()):
                                det_match_miss += 1
                                continue

                            def _make_generic_scorer(si):
                                def scorer(visual_emb_k, props_k):
                                    if getattr(weedvg_model, "crs_scorer", None) is not None:
                                        v_proj = weedvg_model.visual_projector(visual_emb_k.unsqueeze(0))
                                        s_out, _ = weedvg_model.crs_scorer(
                                            v_proj,
                                            token_proj[si:si+1],
                                            token_mask[si:si+1],
                                            query_boxes=props_k.unsqueeze(0),
                                        )
                                        return s_out.squeeze(0)
                                    else:
                                        v_proj = weedvg_model.visual_projector(visual_emb_k.unsqueeze(0)).squeeze(0)
                                        v_norm = F.normalize(v_proj, p=2, dim=-1)
                                        return (v_norm @ text_emb[si]).clamp(-1.0, 1.0)
                                return scorer

                            result = _evaluate_matched_proposals(
                                props=generic_props,
                                visual_features_q=generic_vis_feats,
                                det_scores=generic_det_scores,
                                gt_t=gt_t, gt_xyxy=gt_xyxy,
                                iou_mat=iou_mat,
                                best_iou_per_prop=best_iou_per_prop,
                                keep_mask=keep_mask,
                                text_scorer_fn=_make_generic_scorer(sent_idx),
                            )

                            if result is None:
                                det_match_miss += 1
                                continue

                            det_match_ok += 1
                            r1_hit = result["r1_hit"]
                            rk_hit = result["rk_hit"]
                            top1_iou = result["top1_iou"]
                            inst_rk_delta = result["inst_rk_delta"]
                            miou_sum += top1_iou
                            if r1_hit:
                                hit_r1 += 1
                                # Track which GTs were detected by top-1 (for N-Acc eligibility)
                                t1_ious = result.get("top1_gt_ious")
                                if t1_ious is not None:
                                    for g_idx, rec in enumerate(recs):
                                        if t1_ious[g_idx] >= iou_thresh:
                                            detected_ann_ids.add(rec.ann_id)
                            if rk_hit:
                                hit_rk += 1
                            inst_hit_rk += inst_rk_delta

                            kcs = (cat_name, size_label)
                            if kcs not in cat_size_stats:
                                cat_size_stats[kcs] = {"total": 0, "r1": 0, "rk": 0, "iou_sum": 0.0}
                            cs = cat_size_stats[kcs]
                            cs["total"] += 1
                            cs["iou_sum"] += top1_iou
                            if r1_hit:
                                cs["r1"] += 1
                            if rk_hit:
                                cs["rk"] += 1
                            density_stats[dbin]["miou_sum"] += top1_iou
                            density_stats[dbin]["n_matched"] = density_stats[dbin].get("n_matched", 0) + 1
                            density_stats[dbin]["inst_hit_rk"] += inst_rk_delta
                            if cat_name in ("crop", "weed"):
                                density_stats[dbin][f"{cat_name}_miou_sum"] += top1_iou
                                density_stats[dbin][f"{cat_name}_n_matched"] = density_stats[dbin].get(f"{cat_name}_n_matched", 0) + 1
                                density_stats[dbin][f"{cat_name}_inst_hit_rk"] += inst_rk_delta

                    else:
                        # ── Sentence-conditioned detection path ──
                        img_batch = torch.stack([img_tensor] * len(chunk_texts), dim=0)
                        with torch.no_grad():
                            gd_out = weedvg_model.forward_grounding_dino(img_batch, chunk_texts)

                        pred_boxes = gd_out.get("pred_boxes")
                        pred_logits = gd_out.get("pred_logits")
                        if pred_boxes is None or pred_logits is None:
                            raise RuntimeError("GDINO forward did not return pred_boxes/pred_logits")

                        pred_boxes, pred_logits, proposal_scores = sort_and_trim_proposals(
                            pred_boxes, pred_logits, int(args.max_proposals)
                        )

                        feats = extract_visual_features_at_boxes(
                            gd_out.get("multi_scale_features") or gd_out.get("encoder_features"),
                            pred_boxes, feature_dim=detected_visual_dim, use_multi_scale=False,
                        )
                        if feats is None:
                            raise RuntimeError("Failed to extract visual features at proposal boxes")

                        if batch_idx == 0 and bi == 0 and offset == 0:
                            print(f"[eval] pred_boxes: {pred_boxes.shape}, visual_features: {feats.shape}")

                        if getattr(weedvg_model, "crs_scorer", None) is not None:
                            token_proj, token_mask = weedvg_model.text_projector.encode_tokens(chunk_texts)
                            token_proj, token_mask = token_proj.to(device), token_mask.to(device)
                        else:
                            text_emb = weedvg_model.text_projector.encode_many(chunk_texts).to(device)
                            text_emb = F.normalize(text_emb, p=2, dim=-1)

                        for sent_idx, recs in enumerate(chunk_groups):
                            total_sent += 1
                            gt_boxes = [r.gt_box for r in recs if r.gt_box is not None]
                            cat_id = recs[0].category_id
                            cat_name = CATEGORY_MAP.get(cat_id, "unknown") if cat_id else "unknown"
                            size_label = _extract_size_from_sentence(chunk_texts[sent_idx]) or "unknown"
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

                            props = pred_boxes[sent_idx]
                            q = feats[sent_idx]
                            ps = proposal_scores[sent_idx]

                            pred_xyxy = box_cxcywh_to_xyxy(props)
                            iou_mat = box_iou(pred_xyxy, gt_xyxy)[0]
                            if iou_mat.numel() == 0:
                                det_match_miss += 1
                                continue

                            best_iou_per_prop = iou_mat.max(dim=1)[0]
                            keep_mask = best_iou_per_prop >= iou_thresh

                            if not bool(keep_mask.any().item()):
                                det_match_miss += 1
                                continue

                            def _make_sent_scorer(si):
                                def scorer(visual_emb_k, props_k):
                                    if getattr(weedvg_model, "crs_scorer", None) is not None:
                                        v_proj = weedvg_model.visual_projector(visual_emb_k.unsqueeze(0))
                                        s_out, _ = weedvg_model.crs_scorer(
                                            v_proj,
                                            token_proj[si:si+1],
                                            token_mask[si:si+1],
                                            query_boxes=props_k.unsqueeze(0),
                                        )
                                        return s_out.squeeze(0)
                                    else:
                                        v_proj = weedvg_model.visual_projector(visual_emb_k.unsqueeze(0)).squeeze(0)
                                        v_norm = F.normalize(v_proj, p=2, dim=-1)
                                        return (v_norm @ text_emb[si]).clamp(-1.0, 1.0)
                                return scorer

                            result = _evaluate_matched_proposals(
                                props=props,
                                visual_features_q=q,
                                det_scores=ps,
                                gt_t=gt_t, gt_xyxy=gt_xyxy,
                                iou_mat=iou_mat,
                                best_iou_per_prop=best_iou_per_prop,
                                keep_mask=keep_mask,
                                text_scorer_fn=_make_sent_scorer(sent_idx),
                            )

                            if result is None:
                                det_match_miss += 1
                                continue

                            det_match_ok += 1
                            r1_hit = result["r1_hit"]
                            rk_hit = result["rk_hit"]
                            top1_iou = result["top1_iou"]
                            inst_rk_delta = result["inst_rk_delta"]
                            miou_sum += top1_iou
                            if r1_hit:
                                hit_r1 += 1
                                # Track which GTs were detected by top-1 (for N-Acc eligibility)
                                t1_ious = result.get("top1_gt_ious")
                                if t1_ious is not None:
                                    for g_idx, rec in enumerate(recs):
                                        if t1_ious[g_idx] >= iou_thresh:
                                            detected_ann_ids.add(rec.ann_id)
                            if rk_hit:
                                hit_rk += 1
                            inst_hit_rk += inst_rk_delta

                            kcs = (cat_name, size_label)
                            if kcs not in cat_size_stats:
                                cat_size_stats[kcs] = {"total": 0, "r1": 0, "rk": 0, "iou_sum": 0.0}
                            cs = cat_size_stats[kcs]
                            cs["total"] += 1
                            cs["iou_sum"] += top1_iou
                            if r1_hit:
                                cs["r1"] += 1
                            if rk_hit:
                                cs["rk"] += 1
                            density_stats[dbin]["miou_sum"] += top1_iou
                            density_stats[dbin]["n_matched"] = density_stats[dbin].get("n_matched", 0) + 1
                            density_stats[dbin]["inst_hit_rk"] += inst_rk_delta
                            if cat_name in ("crop", "weed"):
                                density_stats[dbin][f"{cat_name}_miou_sum"] += top1_iou
                                density_stats[dbin][f"{cat_name}_n_matched"] = density_stats[dbin].get(f"{cat_name}_n_matched", 0) + 1
                                density_stats[dbin][f"{cat_name}_inst_hit_rk"] += inst_rk_delta

            # ── N-Acc (test split only) ──
            # GIoU-based, conditioned on successful grounding.
            # For two-stage: run the GroundingDINO backbone with the manipulated
            # text as a sentence-conditioned prompt.  Take the highest-confidence
            # proposal whose logit exceeds box_threshold and check GIoU with the
            # original GT.  If no proposal passes the threshold, or GIoU ≤ 0,
            # the model correctly rejects the manipulation.
            has_test = any(s in splits for s in ["test"])
            if has_test and manipulated_recs_for_nacc:
                for manip_rec in manipulated_recs_for_nacc:
                    # Condition 1: Must have detected the original
                    if manip_rec.ann_id not in detected_ann_ids:
                        continue

                    manip_text = manip_rec.sentence.strip()
                    if not manip_text:
                        continue
                    try:
                        cd = manip_rec.change_detail if isinstance(manip_rec.change_detail, dict) else {}
                        cd_key = json.dumps(cd, sort_keys=True)
                    except Exception:
                        cd_key = "{}"
                    key = (int(item.get("image_id", -1)), int(manip_rec.ann_id), manip_text.lower(), str(manip_rec.change_type or ""), cd_key)
                    if key in seen_manip_keys:
                        continue
                    seen_manip_keys.add(key)

                    ct = manip_rec.change_type
                    cd = manip_rec.change_detail or {}
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

                    # GIoU-based rejection using sentence-conditioned detection:
                    # Run the detector backbone with the manipulated text,
                    # take top-1 proposal, check GIoU with original GT.
                    # Correct rejection = no proposal above threshold OR GIoU <= 0.
                    original_gt = manip_rec.gt_box
                    if original_gt is None:
                        n_acc_total -= 1
                        manip_stats[group]["total"] -= 1
                        continue

                    gt_single = torch.tensor([original_gt], device=device, dtype=torch.float32)
                    gt_single_xyxy = box_cxcywh_to_xyxy(gt_single)

                    incorrect = False
                    try:
                        # Sentence-conditioned detection with manipulated text
                        with torch.no_grad():
                            manip_out = weedvg_model.forward_grounding_dino(
                                img_tensor.unsqueeze(0), [manip_text]
                            )
                        manip_pb = manip_out.get("pred_boxes")
                        manip_pl = manip_out.get("pred_logits")
                        if manip_pb is not None and manip_pl is not None:
                            # Get proposal scores and filter by box_threshold
                            manip_scores = manip_pl.sigmoid().max(dim=-1)[0][0]  # [nq]
                            manip_boxes = manip_pb[0]  # [nq, 4]
                            box_thresh = float(args.box_threshold)
                            mask = manip_scores > box_thresh
                            if mask.any():
                                filtered_boxes = manip_boxes[mask]
                                filtered_scores = manip_scores[mask]
                                best_idx = int(torch.argmax(filtered_scores).item())
                                top1_box = filtered_boxes[best_idx:best_idx+1]
                                top1_xyxy = box_cxcywh_to_xyxy(top1_box.clamp(0, 1))
                                manip_giou = compute_giou(top1_xyxy, gt_single_xyxy)
                                incorrect = (manip_giou > 0)
                            # If no proposal above threshold → incorrect stays False (correct rejection)
                    except Exception:
                        incorrect = False

                    if not incorrect:
                        n_acc_correct += 1
                        manip_stats[group]["correct"] += 1

        # Per-batch progress
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
            msg += f" | N-Acc={n_acc_correct/n_acc_total*100:.1f}% ({n_acc_correct}/{n_acc_total})"
        print(msg)

    return _build_results(
        args, total_sent, det_match_ok, det_match_miss,
        hit_r1, hit_rk, miou_sum, inst_total, inst_hit_rk,
        n_acc_total, n_acc_correct, seen_manip_keys,
        cat_size_stats, density_stats, manip_stats,
    )


# ── Shared Results Builder ───────────────────────────────────────────────────

def _build_results(args, total_sent, det_match_ok, det_match_miss,
                   hit_r1, hit_rk, miou_sum, inst_total, inst_hit_rk,
                   n_acc_total, n_acc_correct,
                   seen_manip_ann_ids,
                   cat_size_stats, density_stats, manip_stats):
    top_k = max(1, int(args.top_k))
    iou_thresh = float(args.iou_thresh)
    splits = [s.strip() for s in str(args.splits).split(",") if s.strip()]

    dm = max(det_match_ok, 1)
    dt = max(total_sent, 1)

    model_label = args.model_label or ("vanilla_gdino" if args.vanilla_gdino else "weedvg_two_stage")

    top1_raw = hit_r1 / dm     # conditional Top-1 (over matched sentences)
    topk_raw = hit_rk / dm     # conditional Top-K
    inst_ret = inst_hit_rk / max(inst_total, 1)
    cov = det_match_ok / dt    # Coverage
    # Unconditional Top-1/Top-K = paper's primary metric (fraction of ALL queries)
    top1_uncond = hit_r1 / dt
    topk_uncond = hit_rk / dt
    results = {
        "split": ",".join(splits),
        "model": model_label,
        "iou_thresh": float(iou_thresh),
        "nacc_iou_thresh": float(args.nacc_iou_thresh),
        "top_k": int(top_k),
        "total_sentences": int(total_sent),
        "matched": int(det_match_ok),
        "missed": int(det_match_miss),
        "Coverage": cov,
        # Conditional metrics (over matched sentences only)
        "Top1": top1_raw,
        f"Top{top_k}": topk_raw,
        # Unconditional metrics = paper's reported values
        "Top1_uncond": top1_uncond,
        f"Top{top_k}_uncond": topk_uncond,
        "InstRet": inst_ret,
        "Top1_over_InstRet": (top1_raw / inst_ret) if inst_ret > 0 else 0.0,
        f"Top{top_k}_over_InstRet": (topk_raw / inst_ret) if inst_ret > 0 else 0.0,
        "mIoU": miou_sum / dm,
    }

    # Fixed denominator used for Table 5 average.
    NACC_PAPER_DENOM = 3616

    if n_acc_total > 0:
        # Per-type N_Acc: correct / model-eligible (matches paper's per-type breakdown)
        results["N_Acc"] = n_acc_correct / n_acc_total
        results["N_Acc_n"] = n_acc_total           # model-eligible count
        results["N_Acc_correct"] = n_acc_correct
        results["N_Acc_method"] = "GIoU<=0"
        results["N_Acc_eligible"] = n_acc_total     # = model-eligible
        results["N_Acc_total_manip"] = len(seen_manip_ann_ids)  # unique processed
        results["N_Acc_avg_paper"] = n_acc_correct / NACC_PAPER_DENOM
        results["N_Acc_paper_denom"] = NACC_PAPER_DENOM

    # Table 3
    t3 = {}
    for (cat, sz), s in cat_size_stats.items():
        n = s["total"]
        if n > 0:
            t3[f"{cat}_{sz}"] = {
                "n": n, "Top1": s["r1"] / n, f"Top{top_k}": s["rk"] / n,
                "mIoU": s["iou_sum"] / n,
            }
    for cat in ["crop", "weed"]:
        ct = cr1 = crk = ciou = 0
        for (c, _), s in cat_size_stats.items():
            if c == cat and s["total"] > 0:
                ct += s["total"]; cr1 += s["r1"]; crk += s["rk"]; ciou += s["iou_sum"]
        if ct > 0:
            t3[f"{cat}_Avg"] = {"n": ct, "Top1": cr1 / ct, f"Top{top_k}": crk / ct, "mIoU": ciou / ct}
    results["table3"] = t3

    # Table 4
    t4 = {}
    for db, ds in density_stats.items():
        e = {"n_sent": ds["n_sent"]}
        e["InstRet"] = ds["inst_hit_rk"] / max(ds["inst_total"], 1)
        e["mIoU"] = ds["miou_sum"] / max(ds.get("n_matched", 0), 1)
        for pf in ["crop", "weed"]:
            e[f"{pf}_InstRet"] = ds.get(f"{pf}_inst_hit_rk", 0) / max(ds.get(f"{pf}_inst_total", 0), 1)
            e[f"{pf}_mIoU"] = ds.get(f"{pf}_miou_sum", 0) / max(ds.get(f"{pf}_n_matched", 0), 1)
        t4[db] = e
    t4["Overall"] = {"InstRet": inst_hit_rk / max(inst_total, 1), "mIoU": miou_sum / dm}
    results["table4"] = t4

    # Table 5 — per-type N_Acc (correct / model_eligible per type) + paper Avg
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
    sp = results.get("split", "")
    tk = results.get("top_k", 5)
    print(f"\n{'='*60}")
    print(f"  {m} — {sp} set results")
    print(f"{'='*60}")

    cov  = results['Coverage'] * 100
    t1u  = results.get('Top1_uncond', results['Top1'] * results['Coverage']) * 100
    tku  = results.get(f'Top{tk}_uncond', results.get(f'Top{tk}', 0) * results['Coverage']) * 100
    t1c  = results['Top1'] * 100
    tkc  = results.get(f'Top{tk}', 0) * 100
    ir   = results['InstRet'] * 100
    miou = results['mIoU'] * 100

    print(f"\n--- Table 2 (paper format: unconditional metrics) ---")
    print(f"  Top-1     (uncond): {t1u:.2f}%   [paper's Top-1]")
    print(f"  Top-{tk}     (uncond): {tku:.2f}%   [paper's Top-{tk}]")
    print(f"  R@0.5 / InstRet:    {ir:.2f}%   [paper's R@0.5]")
    print(f"  mIoU:               {miou:.2f}%   [paper's mIoU]")
    print(f"  --- (internals) ---")
    print(f"  Coverage:           {cov:.2f}%")
    print(f"  Top-1  (cond):      {t1c:.2f}%  [over matched sentences only]")
    print(f"  Top-{tk}  (cond):      {tkc:.2f}%  [over matched sentences only]")
    try:
        print(f"  Top-1/IR: {results.get('Top1_over_InstRet', 0.0):.3f}")
    except Exception:
        pass
    if "N_Acc" in results:
        elig  = results.get('N_Acc_eligible', results.get('N_Acc_n', 0))
        nacc_paper = results.get('N_Acc_avg_paper', results['N_Acc'])
        denom = results.get('N_Acc_paper_denom', elig)
        print(f"  Neg-Acc (paper Avg):{nacc_paper*100:.2f}%  "
              f"[correct={results['N_Acc_correct']}/{denom}]")
        if elig != denom:
            print(f"  N-Acc (model-elig): {results['N_Acc']*100:.2f}%  "
                  f"[correct={results['N_Acc_correct']}/model_elig={elig}]")

    print(f"\n--- Table 3: Category × Size ---")
    t3 = results.get("table3", {})
    for cat in ["crop", "weed"]:
        print(f"\n  {cat.capitalize()}:")
        for sz in ["Tiny", "Small", "Med", "Large", "Avg"]:
            k = f"{cat}_{sz}"
            if k in t3:
                s = t3[k]
                print(f"    {sz:6s}: Top-1={s['Top1']*100:5.1f}%  Top-{tk}={s.get(f'Top{tk}', 0)*100:5.1f}%  "
                      f"mIoU={s['mIoU']*100:5.1f}%  (n={s['n']})")

    print(f"\n--- Table 4: Scene Density ---")
    t4 = results.get("table4", {})
    for db in ["1-10", "11-20", "21-30", ">30", "Overall"]:
        if db in t4:
            d = t4[db]
            if db != "Overall":
                print(f"  {db:>10s}: Crop IR={d.get('crop_InstRet',0)*100:5.1f}% mIoU={d.get('crop_mIoU',0)*100:5.1f}%  |  "
                      f"Weed IR={d.get('weed_InstRet',0)*100:5.1f}% mIoU={d.get('weed_mIoU',0)*100:5.1f}%")
            else:
                print(f"  {'Overall':>10s}: IR={d['InstRet']*100:5.1f}%  mIoU={d['mIoU']*100:5.1f}%")

    if results.get("table5"):
        print(f"\n--- Table 5: Neg-Acc by manipulation type ---")
        # Paper column order: Replace_Category | Swap_Size | Swap_Position | Avg
        for g in ["Replace_Category", "Swap_Size", "Swap_Position", "Avg"]:
            if g in results["table5"]:
                s = results["table5"][g]
                print(f"  {g:20s}: N-Acc={s['N_Acc']*100:5.1f}%  (n={s['n']})")
        if "Avg_paper" in results["table5"]:
            s = results["table5"]["Avg_paper"]
            print(f"  {'Avg_paper':20s}: N-Acc={s['N_Acc']*100:5.1f}%  (n={s['n']})  [paper denom=3616]")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    splits = [s.strip() for s in str(args.splits).split(",") if s.strip()]
    device = torch.device(args.device if (args.device.startswith("cuda") and torch.cuda.is_available()) else "cpu")

    os.makedirs(args.output_dir, exist_ok=True)

    meta = {
        "config_file": args.config_file,
        "base_detector_checkpoint": args.base_detector_checkpoint,
        "detector_checkpoint": args.detector_checkpoint,
        "projector_checkpoint": args.projector_checkpoint,
        "splits": splits,
        "iou_thresh": float(args.iou_thresh),
        "nacc_iou_thresh": float(args.nacc_iou_thresh),
        "nacc_score_thresh": args.nacc_score_thresh,
        "top_k": int(args.top_k),
        "text_threshold": float(args.text_threshold),
        "box_threshold": float(args.box_threshold),
        "vanilla_gdino": bool(args.vanilla_gdino),
        "model_label": args.model_label,
    }
    with open(os.path.join(args.output_dir, "run_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    # Load dataset
    transform = Compose([RandomResize([800], max_size=1333), ToTensor(), Normalize(MEAN, STD)])
    dataset = GRefsEvalDataset(
        grefs_json=args.grefs_json,
        instances_json=args.instances_json,
        images_root=args.images_root,
        splits=splits,
        transform=transform,
        limit=args.limit,
    )

    try:
        split_counts = getattr(dataset, "split_counts", {})
        manipulated_total = int(getattr(dataset, "manipulated_count", 0))
        print(f"[eval] Dataset loaded: {len(dataset)} images, {manipulated_total} manipulated test sentences")
        for sp, counts in sorted(split_counts.items()):
            print(f"  {sp}: {counts['images']} images, {counts['sentences']} sentences, {counts['manipulated']} manipulated")
    except Exception as e:
        print(f"[warn] Could not print dataset stats: {e}")

    print(f"\n[main] Configuration:")
    print(f"  vanilla_gdino={args.vanilla_gdino}, splits={splits}, device={device}")
    print(f"  iou_thresh={args.iou_thresh}, nacc_iou_thresh={args.nacc_iou_thresh}, top_k={args.top_k}")
    print(f"  text_threshold={args.text_threshold}, box_threshold={args.box_threshold}")

    if args.vanilla_gdino:
        # ── Vanilla GroundingDINO mode ──
        checkpoint = args.detector_checkpoint or args.base_detector_checkpoint
        if not checkpoint:
            raise ValueError("Vanilla GDino mode requires --detector-checkpoint or --base-detector-checkpoint")

        print(f"\n[main] Loading vanilla GroundingDINO from '{checkpoint}'...")
        from groundingdino.util.inference import load_model as load_gdino_model
        model = load_gdino_model(args.config_file, checkpoint, device=str(device))
        model.to(device)
        model.eval()

        print(f"\n[main] Starting vanilla GDino evaluation...")
        t0 = time.time()
        with torch.no_grad():
            if device.type == "cuda" and not args.no_amp:
                with torch.amp.autocast("cuda", dtype=torch.float16):
                    results = evaluate_vanilla_gdino(model, dataset, args)
            else:
                results = evaluate_vanilla_gdino(model, dataset, args)
        elapsed = time.time() - t0

    else:
        # ── Two-stage mode ──
        if not args.projector_checkpoint:
            raise ValueError("Two-stage mode requires --projector-checkpoint")

        resolved_detector_checkpoint = args.detector_checkpoint or args.projector_checkpoint
        resolved_base_checkpoint = args.base_detector_checkpoint
        if args.detector_checkpoint is None:
            resolved_base_checkpoint = None

        loader = DataLoader(dataset, batch_size=int(args.batch_size), shuffle=False,
                            num_workers=int(args.num_workers),
                            pin_memory=(device.type == "cuda"), collate_fn=collate_fn)

        gdino_encoder = load_grounding_dino(
            config_file=args.config_file,
            checkpoint_path=resolved_detector_checkpoint,
            base_checkpoint_path=resolved_base_checkpoint,
            device=str(device),
            decoder_layers=args.decoder_layers,
        )
        gdino_encoder.to(device)
        gdino = wrap_grounding_dino(gdino_encoder)

        cfg = SLConfig.fromfile(args.config_file)

        weedvg_model = build_weedvg(grounding_dino_encoder=gdino, temperature=float(args.temperature))
        weedvg_model.eval()
        weedvg_model.to(device)

        print(f"=> Loading independent text encoder: {args.text_encoder_name}")
        text_tokenizer = AutoTokenizer.from_pretrained(args.text_encoder_name)
        text_bert = AutoModel.from_pretrained(args.text_encoder_name)

        try:
            bert_dim = int(text_bert.config.hidden_size)
        except Exception:
            bert_dim = 768

        text_head = TextProjectionHead(input_dim=bert_dim, output_dim=int(args.proj_dim), use_layernorm=True)
        detected_visual_dim = int(getattr(cfg, "hidden_dim", 256))
        visual_head = VisualProjectionHead(
            input_dim=detected_visual_dim,
            output_dim=int(args.proj_dim), use_layernorm=True,
        )

        weedvg_model.visual_projector = visual_head.to(device)
        weedvg_model.text_projector = InferenceTextProjector(text_tokenizer, text_bert, text_head, device)

        print(f"=> Loading projector weights from '{args.projector_checkpoint}'")
        _load_projector_weights(weedvg_model, args.projector_checkpoint)

        # Load CRS if available
        try:
            ckpt = torch.load(args.projector_checkpoint, map_location="cpu")
            crs_sd = ckpt.get("crs_state_dict", None)
            if crs_sd is not None:
                print("=> Found 'crs_state_dict', initializing CRS...")
                crs = ContrastiveRelevanceScorer(
                    dim=int(args.proj_dim), num_heads=8, mlp_hidden=256,
                ).to(device)
                missing_k, unexpected_k = crs.load_state_dict(crs_sd, strict=False)
                if missing_k:
                    print(f"[warn] CRS load missing keys: {missing_k}")
                if unexpected_k:
                    print(f"[warn] CRS load unexpected keys: {unexpected_k}")
                crs.eval()
                weedvg_model.crs_scorer = crs
                # Apply scoring mode for ablation
                if hasattr(args, 'scoring_mode') and args.scoring_mode != 'full':
                    weedvg_model.crs_scorer._scoring_mode = args.scoring_mode
                    print(f"[ablation] CRS scoring mode set to: {args.scoring_mode}")
            else:
                weedvg_model.crs_scorer = None
        except Exception as e:
            print(f"[warn] CRS load failed: {e}")
            weedvg_model.crs_scorer = None

        print(f"\n[eval] device={device} | batch_size={args.batch_size} | splits={splits}")
        print(f"[eval] detector={resolved_detector_checkpoint}")
        print(f"[eval] projector={args.projector_checkpoint}")

        print(f"\n[main] Starting two-stage evaluation...")
        t0 = time.time()
        with torch.no_grad():
            if device.type == "cuda" and not args.no_amp:
                with torch.amp.autocast("cuda", dtype=torch.float16):
                    results = evaluate_two_stage(weedvg_model, dataset, args, detected_visual_dim)
            else:
                results = evaluate_two_stage(weedvg_model, dataset, args, detected_visual_dim)
        elapsed = time.time() - t0

    results["elapsed_seconds"] = elapsed
    print(f"\n[main] Evaluation completed in {elapsed:.1f}s")

    out = os.path.join(args.output_dir, "metrics.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[main] Results saved to {out}")
    print_summary(results)


if __name__ == "__main__":
    main()
