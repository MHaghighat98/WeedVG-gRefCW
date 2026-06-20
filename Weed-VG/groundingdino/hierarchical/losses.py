"""Loss functions for hierarchical contrastive learning with GroundingDINO."""

import math

import torch
import torch.nn as nn
from groundingdino.util.box_ops import box_cxcywh_to_xyxy, box_iou


def _sanitize_boxes(boxes):
    return torch.nan_to_num(boxes, nan=0.5, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)




def pairwise_eiou_cost(pred_boxes, gt_boxes, eps=1e-6):
    """Compute pairwise base EIoU loss (without focal weighting) between two box sets."""

    if pred_boxes.numel() == 0 or gt_boxes.numel() == 0:
        device = pred_boxes.device if pred_boxes.numel() else gt_boxes.device
        return torch.zeros((pred_boxes.shape[0], gt_boxes.shape[0]), device=device)

    pred_xy = box_cxcywh_to_xyxy(pred_boxes)[:, None, :]
    gt_xy = box_cxcywh_to_xyxy(gt_boxes)[None, :, :]

    px1 = pred_xy[..., 0]
    py1 = pred_xy[..., 1]
    px2 = pred_xy[..., 2]
    py2 = pred_xy[..., 3]

    gx1 = gt_xy[..., 0]
    gy1 = gt_xy[..., 1]
    gx2 = gt_xy[..., 2]
    gy2 = gt_xy[..., 3]

    pw = (px2 - px1).clamp(min=eps)
    ph = (py2 - py1).clamp(min=eps)
    gw = (gx2 - gx1).clamp(min=eps)
    gh = (gy2 - gy1).clamp(min=eps)

    inter_x1 = torch.maximum(px1, gx1)
    inter_y1 = torch.maximum(py1, gy1)
    inter_x2 = torch.minimum(px2, gx2)
    inter_y2 = torch.minimum(py2, gy2)
    inter_w = (inter_x2 - inter_x1).clamp(min=0.0)
    inter_h = (inter_y2 - inter_y1).clamp(min=0.0)
    inter_area = inter_w * inter_h

    area_p = pw * ph
    area_g = gw * gh
    union = area_p + area_g - inter_area
    iou = inter_area / (union.clamp_min(eps))

    ctr_px = (px1 + px2) * 0.5
    ctr_py = (py1 + py2) * 0.5
    ctr_gx = (gx1 + gx2) * 0.5
    ctr_gy = (gy1 + gy2) * 0.5
    center_dist = (ctr_px - ctr_gx) ** 2 + (ctr_py - ctr_gy) ** 2

    c_x1 = torch.minimum(px1, gx1)
    c_y1 = torch.minimum(py1, gy1)
    c_x2 = torch.maximum(px2, gx2)
    c_y2 = torch.maximum(py2, gy2)
    c_w = (c_x2 - c_x1).clamp(min=eps)
    c_h = (c_y2 - c_y1).clamp(min=eps)
    diag_c = c_w**2 + c_h**2 + eps

    w_diff = pw - gw
    h_diff = ph - gh

    eiou = (1.0 - iou) + center_dist / diag_c + (w_diff**2) / (c_w**2 + eps) + (h_diff**2) / (c_h**2 + eps)
    return eiou




def pairwise_iou_and_rel_size(pred_boxes, gt_boxes, eps=1e-6):
    """Return pairwise IoU matrix and relative-size term matrix between two box sets.

    The relative-size term is (|pw-gw|/gw + |ph-gh|/gh), which is scale-invariant.
    """
    if pred_boxes.numel() == 0 or gt_boxes.numel() == 0:
        device = pred_boxes.device if pred_boxes.numel() else gt_boxes.device
        # return iou and size matrices
        return (
            torch.zeros((pred_boxes.shape[0], gt_boxes.shape[0]), device=device),
            torch.zeros((pred_boxes.shape[0], gt_boxes.shape[0]), device=device),
        )

    pred_xy = box_cxcywh_to_xyxy(pred_boxes)[:, None, :]
    gt_xy = box_cxcywh_to_xyxy(gt_boxes)[None, :, :]

    px1 = pred_xy[..., 0]
    py1 = pred_xy[..., 1]
    px2 = pred_xy[..., 2]
    py2 = pred_xy[..., 3]

    gx1 = gt_xy[..., 0]
    gy1 = gt_xy[..., 1]
    gx2 = gt_xy[..., 2]
    gy2 = gt_xy[..., 3]

    pw = (px2 - px1).clamp(min=eps)
    ph = (py2 - py1).clamp(min=eps)
    gw = (gx2 - gx1).clamp(min=eps)
    gh = (gy2 - gy1).clamp(min=eps)

    inter_x1 = torch.maximum(px1, gx1)
    inter_y1 = torch.maximum(py1, gy1)
    inter_x2 = torch.minimum(px2, gx2)
    inter_y2 = torch.minimum(py2, gy2)
    inter_w = (inter_x2 - inter_x1).clamp(min=0.0)
    inter_h = (inter_y2 - inter_y1).clamp(min=0.0)
    inter_area = inter_w * inter_h

    area_p = pw * ph
    area_g = gw * gh
    union = area_p + area_g - inter_area
    iou = inter_area / union.clamp_min(eps)

    # relative size: sum of relative width and height absolute errors
    size_rel = (torch.abs(pw - gw) / gw) + (torch.abs(ph - gh) / gh)
    size_rel = torch.nan_to_num(size_rel, nan=0.0, posinf=1e6, neginf=0.0)

    return iou, size_rel


def pairwise_center_cost(pred_boxes, gt_boxes, eps=1e-6):
    """Compute pairwise squared center distance between two box sets (cx,cy).

    Returns a matrix [N_pred, N_gt] of squared L2 distances in normalized coordinates.
    """
    if pred_boxes.numel() == 0 or gt_boxes.numel() == 0:
        device = pred_boxes.device if pred_boxes.numel() else gt_boxes.device
        return torch.zeros((pred_boxes.shape[0], gt_boxes.shape[0]), device=device)

    # pred_boxes and gt_boxes are in cxcywh format
    pred_centers = pred_boxes[:, :2][:, None, :]
    gt_centers = gt_boxes[:, :2][None, :, :]
    diff = pred_centers - gt_centers
    dist2 = (diff**2).sum(dim=-1)
    return dist2


def pairwise_l1_cost(pred_boxes, gt_boxes, eps=1e-6):
    """Compute pairwise L1 distance between boxes (cxcywh) for matching."""

    if pred_boxes.numel() == 0 or gt_boxes.numel() == 0:
        device = pred_boxes.device if pred_boxes.numel() else gt_boxes.device
        return torch.zeros((pred_boxes.shape[0], gt_boxes.shape[0]), device=device)

    diff = torch.abs(pred_boxes[:, None, :] - gt_boxes[None, :, :]).clamp_min(eps)
    return diff.sum(dim=-1)






def compute_iou_simple(pred_boxes, gt_boxes, eps=1e-6):
    """Compute simple IoU between matched box pairs (cxcywh)."""
    pred_xy = box_cxcywh_to_xyxy(pred_boxes)
    gt_xy = box_cxcywh_to_xyxy(gt_boxes)

    px1, py1, px2, py2 = pred_xy.unbind(-1)
    gx1, gy1, gx2, gy2 = gt_xy.unbind(-1)

    inter_x1 = torch.maximum(px1, gx1)
    inter_y1 = torch.maximum(py1, gy1)
    inter_x2 = torch.minimum(px2, gx2)
    inter_y2 = torch.minimum(py2, gy2)
    inter_w = (inter_x2 - inter_x1).clamp(min=0.0)
    inter_h = (inter_y2 - inter_y1).clamp(min=0.0)
    inter_area = inter_w * inter_h

    area_p = (px2 - px1).clamp(min=eps) * (py2 - py1).clamp(min=eps)
    area_g = (gx2 - gx1).clamp(min=eps) * (gy2 - gy1).clamp(min=eps)
    union = area_p + area_g - inter_area
    iou = inter_area / (union.clamp_min(eps))
    return iou


def dynamic_interp_iou_loss(pred_boxes, gt_boxes, alpha=0.99, eps=1e-6):
    """
    Compute Interpolated IoU Loss with a fixed interpolation weight.
    L = L_IoU(B_pred, B_gt) + L_IoU(B_int, B_gt)
    where B_int = alpha * B_gt + (1 - alpha) * B_pred and alpha is constant.
    The project previously used a dynamically-clamped alpha based on (1 - IoU).
    This variant uses a fixed alpha (default 0.98) as requested.
    """
    if pred_boxes.numel() == 0 or gt_boxes.numel() == 0:
        zero = torch.zeros(1, device=pred_boxes.device if pred_boxes.numel() else gt_boxes.device)
        return zero.sum(), zero[:0], zero[:0]

    iou_pred = compute_iou_simple(pred_boxes, gt_boxes, eps)

    # Fixed alpha: use constant interpolation weight for all pairs
    if isinstance(alpha, float) or isinstance(alpha, int):
        alpha_val = pred_boxes.new_full(iou_pred.shape, float(alpha))
    else:
        # allow tensor-like alpha of matching shape
        alpha_val = alpha

    # Interpolated box
    alpha_exp = alpha_val.unsqueeze(-1)
    box_int = alpha_exp * gt_boxes + (1.0 - alpha_exp) * pred_boxes

    iou_int = compute_iou_simple(box_int, gt_boxes, eps)

    loss_pred = 1.0 - iou_pred
    loss_int = 1.0 - iou_int

    total_per_box = loss_pred + loss_int
    return total_per_box.sum(), total_per_box, iou_pred


def pairwise_interp_iou_cost(pred_boxes, gt_boxes, alpha=0.99, eps=1e-6):
    """Compute pairwise Interpolated IoU loss cost between two box sets.

    Returns a matrix [N_pred, N_gt] where each entry is:
        (1 - IoU(pred, gt)) + (1 - IoU(alpha*gt + (1-alpha)*pred, gt))
    matching the per-pair loss used in `dynamic_interp_iou_loss`.
    """

    if pred_boxes.numel() == 0 or gt_boxes.numel() == 0:
        device = pred_boxes.device if pred_boxes.numel() else gt_boxes.device
        return torch.zeros((pred_boxes.shape[0], gt_boxes.shape[0]), device=device)

    # Broadcast boxes: pred [N,1,4], gt [1,M,4]
    pred = pred_boxes[:, None, :]
    gt = gt_boxes[None, :, :]

    pred_xy = box_cxcywh_to_xyxy(pred.reshape(-1, 4)).reshape(pred.shape[0], pred.shape[1], 4)
    gt_xy = box_cxcywh_to_xyxy(gt.reshape(-1, 4)).reshape(gt.shape[0], gt.shape[1], 4)

    px1, py1, px2, py2 = pred_xy.unbind(-1)
    gx1, gy1, gx2, gy2 = gt_xy.unbind(-1)

    inter_x1 = torch.maximum(px1, gx1)
    inter_y1 = torch.maximum(py1, gy1)
    inter_x2 = torch.minimum(px2, gx2)
    inter_y2 = torch.minimum(py2, gy2)
    inter_w = (inter_x2 - inter_x1).clamp(min=0.0)
    inter_h = (inter_y2 - inter_y1).clamp(min=0.0)
    inter_area = inter_w * inter_h

    area_p = (px2 - px1).clamp(min=eps) * (py2 - py1).clamp(min=eps)
    area_g = (gx2 - gx1).clamp(min=eps) * (gy2 - gy1).clamp(min=eps)
    union = area_p + area_g - inter_area
    iou_pred = inter_area / union.clamp_min(eps)

    # Interpolated box in cxcywh space
    if isinstance(alpha, float) or isinstance(alpha, int):
        alpha_val = float(alpha)
        box_int = (alpha_val * gt) + ((1.0 - alpha_val) * pred)
    else:
        # Tensor alpha: broadcast to [N,M,1]
        alpha_t = alpha
        if not torch.is_tensor(alpha_t):
            alpha_t = pred_boxes.new_tensor(alpha_t)
        if alpha_t.dim() == 0:
            alpha_t = alpha_t.view(1, 1, 1)
        elif alpha_t.dim() == 1:
            alpha_t = alpha_t.view(-1, 1, 1)
        alpha_t = alpha_t.to(device=pred_boxes.device, dtype=pred_boxes.dtype)
        box_int = (alpha_t * gt) + ((1.0 - alpha_t) * pred)

    box_int_xy = box_cxcywh_to_xyxy(box_int.reshape(-1, 4)).reshape(box_int.shape[0], box_int.shape[1], 4)
    ix1, iy1, ix2, iy2 = box_int_xy.unbind(-1)

    inter2_x1 = torch.maximum(ix1, gx1)
    inter2_y1 = torch.maximum(iy1, gy1)
    inter2_x2 = torch.minimum(ix2, gx2)
    inter2_y2 = torch.minimum(iy2, gy2)
    inter2_w = (inter2_x2 - inter2_x1).clamp(min=0.0)
    inter2_h = (inter2_y2 - inter2_y1).clamp(min=0.0)
    inter2_area = inter2_w * inter2_h

    area_int = (ix2 - ix1).clamp(min=eps) * (iy2 - iy1).clamp(min=eps)
    union2 = area_int + area_g - inter2_area
    iou_int = inter2_area / union2.clamp_min(eps)

    cost = (1.0 - iou_pred) + (1.0 - iou_int)
    return torch.nan_to_num(cost, nan=1.0, posinf=1e6, neginf=1.0)




def compute_center_error(pred_boxes, gt_boxes, eps=1e-6):
    """Compute normalized squared center distance for matched pairs.

    Returns a tensor shaped [N] with values center_dist / diag_c.
    """
    if pred_boxes.numel() == 0 or gt_boxes.numel() == 0:
        return torch.zeros(pred_boxes.shape[0], device=pred_boxes.device)

    pred_xy = box_cxcywh_to_xyxy(pred_boxes)
    gt_xy = box_cxcywh_to_xyxy(gt_boxes)

    px1, py1, px2, py2 = pred_xy.unbind(-1)
    gx1, gy1, gx2, gy2 = gt_xy.unbind(-1)

    ctr_px = (px1 + px2) * 0.5
    ctr_py = (py1 + py2) * 0.5
    ctr_gx = (gx1 + gx2) * 0.5
    ctr_gy = (gy1 + gy2) * 0.5
    center_dist = (ctr_px - ctr_gx) ** 2 + (ctr_py - ctr_gy) ** 2

    c_x1 = torch.minimum(px1, gx1)
    c_y1 = torch.minimum(py1, gy1)
    c_x2 = torch.maximum(px2, gx2)
    c_y2 = torch.maximum(py2, gy2)
    c_w = (c_x2 - c_x1).clamp(min=eps)
    c_h = (c_y2 - c_y1).clamp(min=eps)
    diag_c = c_w**2 + c_h**2 + eps

    return center_dist / diag_c


def compute_total_loss(
    detr_outputs,
    targets,
    hmlc_loss,
    features,
    pos_mask=None,
    hierarchical_labels=None,
    lambda_weights=None,
    lambda_hmlc: float = 1.0,
    lambda_diiou: float = 1.0,
    lambda_suppress: float = 0.0,
    return_debug: bool = False,
    return_hmlc_info: bool = False,
    initial_boxes=None,
    epoch: int = 0,
    hmlc_loss_start_epoch: int = 0,
    diiou_loss_start_epoch: int = 0,
    ohem_keep_ratio: float = 1.0,
    ohem_nms_iou: float = 0.7,
    precomputed_matches=None,
    matching_cost: str = "composite",
    matching_interp_alpha: float = 0.99,
):
    """
    Compute the total loss as a weighted sum of HMLC + Dynamic InterpIoU.

    Args:
        detr_outputs: dict with 'pred_boxes'
        targets: list of dicts with 'boxes'
        hmlc_loss: HMLC loss module
        features: [N, n_views, D] tensor for contrastive loss
        hierarchical_labels: [N, num_levels] hierarchical labels for HMLC loss
        lambda_diiou: weight for Dynamic InterpIoU loss
        lambda_hmlc: weight for HMLC loss
        ohem_keep_ratio: Ratio of hard examples to keep (0.0 < ratio <= 1.0). Default 1.0 (keep all).
        ohem_nms_iou: IoU threshold for NMS deduplication of hard examples. Default 0.7.

    Returns:
        dict with 'diiou', 'hmlc', 'total'
    """
    from scipy.optimize import linear_sum_assignment

    # Curriculum learning
    if epoch < diiou_loss_start_epoch:
        lambda_diiou = 0.0
    if epoch < hmlc_loss_start_epoch:
        lambda_hmlc = 0.0

    # Simplified bbox losses without external dependencies
    pred_boxes_input = detr_outputs.get("pred_boxes")
    if pred_boxes_input is None:
        raise ValueError("detr_outputs must provide 'pred_boxes'")

    pred_boxes_list = []
    if isinstance(pred_boxes_input, (list, tuple)):
        for pb in pred_boxes_input:
            if pb is None:
                continue
            if not torch.is_tensor(pb):
                raise TypeError(f"Entries in pred_boxes list must be tensors. Got {type(pb)}")
            sanitized = _sanitize_boxes(pb)
            pred_boxes_list.append(sanitized)
    elif torch.is_tensor(pred_boxes_input):
        sanitized = _sanitize_boxes(pred_boxes_input)
        if sanitized.dim() == 2:
            pred_boxes_list.append(sanitized)
        elif sanitized.dim() == 3:
            pred_boxes_list.extend(sanitized.unbind(dim=0))
        else:
            raise ValueError(f"pred_boxes tensor must have shape [B,N,4] or [N,4], got {tuple(sanitized.shape)}")
    else:
        raise TypeError(f"Unsupported type for pred_boxes: {type(pred_boxes_input)}")

    if not pred_boxes_list:
        device = targets[0]["boxes"].device if targets else (features.device if torch.is_tensor(features) else torch.device("cpu"))
    else:
        device = pred_boxes_list[0].device

    B = len(pred_boxes_list)

    # Accumulators
    all_diiou_losses = []
    all_matched_boxes = []
    all_center_errors = []
    matched_iou_sum = torch.tensor(0.0, device=device)
    matched_iou_count = 0
    # Track matched IoU gain (predicted IoU - initial/proposal IoU)
    matched_iou_gain_sum = torch.tensor(0.0, device=device)
    matched_iou_gain_count = 0

    debug_info = [] if return_debug else None

    for batch_idx in range(B):
        pred_boxes_i = pred_boxes_list[batch_idx]
        initial_boxes_i = initial_boxes[batch_idx] if (initial_boxes is not None) else None
        if batch_idx >= len(targets):
            break

        target_i = targets[batch_idx]
        gt_boxes = target_i["boxes"]

        if len(gt_boxes) == 0:
            continue

        gt_boxes = _sanitize_boxes(gt_boxes)

        if pred_boxes_i.numel() == 0:
            continue

        # Curriculum scalar used for post-matching weighting (center-first, then scale)
        ramp_start = int(diiou_loss_start_epoch)
        ramp_len = 15
        t = float(epoch - ramp_start) / float(max(1, ramp_len))
        t = max(0.0, min(1.0, t))

        if precomputed_matches is not None:
            if batch_idx < len(precomputed_matches) and precomputed_matches[batch_idx] is not None:
                pred_indices, gt_indices = precomputed_matches[batch_idx]
                if torch.is_tensor(pred_indices):
                    pred_indices = pred_indices.detach().cpu().numpy()
                if torch.is_tensor(gt_indices):
                    gt_indices = gt_indices.detach().cpu().numpy()
            else:
                pred_indices, gt_indices = [], []
        else:
            # Matching is performed in box space, so decoding to (cx,cy,w,h) is required.
            # Use initial_boxes for matching if available to reflect proposal-based assignment.
            boxes_for_matching = initial_boxes_i if initial_boxes_i is not None else pred_boxes_i

            cost_mode = str(matching_cost or "composite").lower()
            if cost_mode in ("interp", "interp_iou", "interpiou", "interp-iou"):
                cost_tensor = pairwise_interp_iou_cost(boxes_for_matching, gt_boxes, alpha=matching_interp_alpha)
            else:
                # Composite (IoU + center + relative size) cost.
                iou_mat, size_rel_mat = pairwise_iou_and_rel_size(boxes_for_matching, gt_boxes)
                center_cost_mat = pairwise_center_cost(boxes_for_matching, gt_boxes)

                lambda_center = 1.0
                lambda_size = 0.5 * t

                eps = 1e-6
                center_mean = center_cost_mat.mean().clamp_min(eps)
                size_mean = size_rel_mat.mean().clamp_min(eps)
                center_norm = (center_cost_mat / center_mean).clamp(min=0.0, max=10.0)
                size_norm = (size_rel_mat / size_mean).clamp(min=0.0, max=10.0)

                cost_tensor = (1.0 - iou_mat) + (lambda_center * center_norm) + (lambda_size * size_norm)

            cost_tensor = torch.nan_to_num(cost_tensor, nan=1.0, posinf=1e6, neginf=1.0)
            cost_matrix = cost_tensor.detach().cpu().numpy()
            pred_indices, gt_indices = linear_sum_assignment(cost_matrix)

        matched_pred = pred_boxes_i[pred_indices]
        matched_gt = gt_boxes[gt_indices]

        # Compute Dynamic InterpIoU Loss
        _, interp_loss_per_box, per_pair_iou = dynamic_interp_iou_loss(matched_pred, matched_gt)

        # (1) InterpIoU pairwise reweighting disabled: use uniform pair weights.
        # Previous behavior applied a per-pair reweight based on IoU/center/size
        # difficulty; this reweighting has been removed to avoid creating
        # an additional curriculum gradient path and excessive corrections.
        pair_reweights = None

        # For curriculum weighting, measure center/size error using the *reference/proposal*
        # boxes when available (initial_boxes). This keeps weights stable and prevents
        # weights from collapsing as predictions improve.
        if initial_boxes_i is not None and initial_boxes_i.numel() > 0:
            try:
                boxes_for_weight = initial_boxes_i[pred_indices]
            except Exception:
                boxes_for_weight = matched_pred
        else:
            boxes_for_weight = matched_pred

        # Weighting signals (detached): center and relative size error
        # use normalized squared center distance (center_dist / diag_c)
        ctr_error = compute_center_error(boxes_for_weight, matched_gt)

        # Relative size error (scale mismatch): |pw-gw|/gw + |ph-gh|/gh
        eps = 1e-6
        pw = boxes_for_weight[:, 2].clamp_min(eps)
        ph = boxes_for_weight[:, 3].clamp_min(eps)
        gw = matched_gt[:, 2].clamp_min(eps)
        gh = matched_gt[:, 3].clamp_min(eps)
        size_rel_error = (torch.abs(pw - gw) / gw) + (torch.abs(ph - gh) / gh)
        size_rel_error = torch.nan_to_num(size_rel_error, nan=0.0, posinf=1e6, neginf=0.0)

        # Original behavior: no post-match weighting.
        # OHEM (below) is the only mechanism that changes which pairs contribute.
        pair_weights = pair_reweights if pair_reweights is not None else torch.ones_like(interp_loss_per_box)
        total_per_box = interp_loss_per_box

        if total_per_box.numel() > 0:
            all_diiou_losses.append(total_per_box)
            all_matched_boxes.append(matched_pred)
            all_center_errors.append(ctr_error)
            matched_iou_sum += per_pair_iou.sum()
            matched_iou_count += per_pair_iou.numel()

            # If initial (pre-refine) boxes were provided, compute their IoU
            # against matched GT boxes so we can report an IoU gain metric.
            if initial_boxes_i is not None and initial_boxes_i.numel() > 0:
                try:
                    init_matched = initial_boxes_i[pred_indices]
                    init_xy = box_cxcywh_to_xyxy(init_matched)
                    gt_xy = box_cxcywh_to_xyxy(matched_gt)
                    iou_init_mat, _ = box_iou(init_xy, gt_xy)
                    n_diag = min(iou_init_mat.shape[0], iou_init_mat.shape[1])
                    if n_diag > 0:
                        iou_init_diag = iou_init_mat.diag()[:n_diag]
                        per_iou = per_pair_iou[:n_diag]
                        gain = per_iou - iou_init_diag
                        matched_iou_gain_sum += gain.sum()
                        matched_iou_gain_count += gain.numel()
                except Exception:
                    pass

        if return_debug:
            pairs = []
            for idx_pair, (pi, gi) in enumerate(zip(pred_indices, gt_indices)):
                pairs.append(
                    {
                        "pred_index": int(pi),
                        "gt_index": int(gi),
                        "pred_box_cxcywh": matched_pred[idx_pair].detach().cpu().tolist(),
                        "gt_box_cxcywh": matched_gt[idx_pair].detach().cpu().tolist(),
                        "iou": float(per_pair_iou[idx_pair].detach().cpu().item()),
                        "pair_reweight": float(pair_weights[idx_pair].detach().cpu().item()) if pair_weights.numel() > 0 else 1.0,
                        "interp_iou_loss": float(interp_loss_per_box[idx_pair].detach().cpu().item()) if interp_loss_per_box.numel() > 0 else 0.0,
                        "ctr_error": float(ctr_error[idx_pair].detach().cpu().item()) if ctr_error.numel() > 0 else 0.0,
                        "size_rel_error": float(size_rel_error[idx_pair].detach().cpu().item()) if size_rel_error.numel() > 0 else 0.0,
                        "pair_weight": float(pair_weights[idx_pair].detach().cpu().item()) if pair_weights.numel() > 0 else 1.0,
                        "hard_weight": 0.0,
                    }
                )
            debug_info.append(
                {
                    "batch_index": int(batch_idx),
                    "num_gt": int(gt_boxes.shape[0]),
                    "num_pred": int(pred_boxes_i.shape[0]),
                    "weight_source": "initial_boxes" if (initial_boxes_i is not None) else "pred_boxes",
                    "pairs": pairs,
                }
            )

    # OHEM Logic: Sort and keep top K hard examples.
    # Ranking is based on the per-match Dynamic InterpIoU loss magnitude itself
    # (higher loss = harder example). Note: NMS (IoU-based suppression) is removed.
    diiou_loss_count = 0
    if len(all_diiou_losses) > 0:
        flat_losses = torch.cat(all_diiou_losses)
        num_matches = flat_losses.numel()

        if 0.0 < ohem_keep_ratio < 1.0:
            num_keep = max(1, int(num_matches * ohem_keep_ratio))
            if num_keep < num_matches:
                # Sort descending by loss (hardest first)
                _, top_indices = torch.topk(flat_losses, num_keep)
                # Select corresponding losses
                selected_losses = flat_losses[top_indices]

                diiou_loss_sum = selected_losses.sum()
                diiou_loss_count = num_keep
            else:
                diiou_loss_sum = flat_losses.sum()
                diiou_loss_count = num_matches
        else:
            diiou_loss_sum = flat_losses.sum()
            diiou_loss_count = num_matches

        if diiou_loss_count > 0:
            diiou_loss_value = lambda_diiou * (diiou_loss_sum / diiou_loss_count)
        else:
            diiou_loss_value = torch.tensor(0.0, device=device)

        mean_iou_value = matched_iou_sum / max(1, matched_iou_count)
    else:
        diiou_loss_value = torch.tensor(0.0, device=device)
        mean_iou_value = torch.tensor(0.0, device=device)

    # Compute HMLC contrastive loss
    import torch.nn.functional as F

    features = torch.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
    features = F.normalize(features, p=2, dim=-1)

    hmlc_per_level_info = None
    if hierarchical_labels is not None:
        if return_hmlc_info:
            try:
                hmlc_out = hmlc_loss(features, hierarchical_labels, lambda_weights=lambda_weights, return_info=True)
            except TypeError:
                hmlc_out = hmlc_loss(features, hierarchical_labels, lambda_weights=lambda_weights)

            if isinstance(hmlc_out, tuple) and len(hmlc_out) == 2:
                hmlc_scalar, hmlc_per_level_info = hmlc_out
                hmlc = lambda_hmlc * hmlc_scalar
            else:
                hmlc = lambda_hmlc * hmlc_out
        else:
            hmlc = lambda_hmlc * hmlc_loss(features, hierarchical_labels, lambda_weights=lambda_weights)
    elif pos_mask is not None:
        hmlc = lambda_hmlc * hmlc_loss(features, pos_mask=pos_mask)
    else:
        raise ValueError("Either hierarchical_labels or pos_mask must be provided")

    total = diiou_loss_value + hmlc

    # Mean IoU gain (predicted IoU - initial IoU) for matched pairs
    if 'matched_iou_gain_count' in locals() and matched_iou_gain_count > 0:
        mean_iou_gain = matched_iou_gain_sum / max(1, matched_iou_gain_count)
    else:
        mean_iou_gain = torch.tensor(0.0, device=device)

    result = {
        "diiou": diiou_loss_value,
        "hmlc": hmlc,
        "total": total,
        "iou": mean_iou_value.detach(),
        "matched_iou_gain": mean_iou_gain.detach(),
        "diiou_count": int(diiou_loss_count),
    }

    if return_debug:
        result["debug"] = debug_info
    if hmlc_per_level_info is not None:
        result["hmlc_per_level"] = hmlc_per_level_info
    return result








