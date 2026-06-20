# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Utilities for bounding box manipulation and IoU/CIoU utilities.
"""

import torch
from torchvision.ops.boxes import box_area


def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h), (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)


def box_xyxy_to_cxcywh(x):
    x0, y0, x1, y1 = x.unbind(-1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2, (x1 - x0), (y1 - y0)]
    return torch.stack(b, dim=-1)


# modified from torchvision to also return the union
def box_iou(boxes1, boxes2):
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    # import ipdb; ipdb.set_trace()
    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    union = area1[:, None] + area2 - inter

    iou = inter / (union + 1e-6)
    return iou, union


def generalized_box_iou(boxes1, boxes2):
    """
    Generalized IoU implementation (kept for compatibility).

    The boxes should be in [x0, y0, x1, y1] format

    Returns a [N, M] pairwise matrix, where N = len(boxes1)
    and M = len(boxes2)
    """
    # degenerate boxes gives inf / nan results
    # so do an early check
    assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
    assert (boxes2[:, 2:] >= boxes2[:, :2]).all()
    # except:
    #     import ipdb; ipdb.set_trace()
    iou, union = box_iou(boxes1, boxes2)

    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    area = wh[:, :, 0] * wh[:, :, 1]

    return iou - (area - union) / (area + 1e-6)


# modified from torchvision to also return the union
def box_iou_pairwise(boxes1, boxes2):
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, :2], boxes2[:, :2])  # [N,2]
    rb = torch.min(boxes1[:, 2:], boxes2[:, 2:])  # [N,2]

    wh = (rb - lt).clamp(min=0)  # [N,2]
    inter = wh[:, 0] * wh[:, 1]  # [N]

    union = area1 + area2 - inter

    iou = inter / union
    return iou, union


def generalized_box_iou_pairwise(boxes1, boxes2):
    """
    Generalized IoU pairwise helper (kept for compatibility).

    Input:
        - boxes1, boxes2: N,4
    Output:
        - giou: N, 4
    """
    # degenerate boxes gives inf / nan results
    # so do an early check
    assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
    assert (boxes2[:, 2:] >= boxes2[:, :2]).all()
    assert boxes1.shape == boxes2.shape
    iou, union = box_iou_pairwise(boxes1, boxes2)  # N, 4

    lt = torch.min(boxes1[:, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, 2:], boxes2[:, 2:])

    wh = (rb - lt).clamp(min=0)  # [N,2]
    area = wh[:, 0] * wh[:, 1]

    return iou - (area - union) / area


def complete_box_iou_pairwise(boxes1, boxes2):
    """
    Complete IoU (CIoU) from https://arxiv.org/abs/1911.08287

    Considers overlap area, central point distance, and aspect ratio.
    Good for scale convergence.

    Input:
        - boxes1, boxes2: N,4 (xyxy format) or different shapes for pairwise
    Output:
        - ciou: [N1, N2] if different shapes, else [N]
    """
    # degenerate boxes gives inf / nan results
    # so do an early check
    assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
    assert (boxes2[:, 2:] >= boxes2[:, :2]).all()

    # 1. IoU
    iou, union = box_iou(boxes1, boxes2)

    # 2. Distance term (DIoU)
    # Center points
    c1_x = (boxes1[:, 0] + boxes1[:, 2]) / 2
    c1_y = (boxes1[:, 1] + boxes1[:, 3]) / 2
    c2_x = (boxes2[:, 0] + boxes2[:, 2]) / 2
    c2_y = (boxes2[:, 1] + boxes2[:, 3]) / 2

    # Euclidean distance between centers [N1, N2]
    center_dist_sq = (c1_x.unsqueeze(1) - c2_x.unsqueeze(0)) ** 2 + (c1_y.unsqueeze(1) - c2_y.unsqueeze(0)) ** 2

    # Enclosing box diagonal distance
    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])
    wh = (rb - lt).clamp(min=0)  # [N1,N2,2]
    diag_dist_sq = wh[:, :, 0] ** 2 + wh[:, :, 1] ** 2 + 1e-7

    diou_term = center_dist_sq / diag_dist_sq

    # 3. Aspect Ratio term (v)
    w1 = boxes1[:, 2] - boxes1[:, 0]
    h1 = boxes1[:, 3] - boxes1[:, 1]
    w2 = boxes2[:, 2] - boxes2[:, 0]
    h2 = boxes2[:, 3] - boxes2[:, 1]

    # Use arctan for aspect ratio consistency
    # v = (4 / pi^2) * (arctan(w_gt/h_gt) - arctan(w/h))^2
    import math

    v = (4 / (math.pi**2)) * torch.pow(
        torch.atan(w2.unsqueeze(0) / (h2.unsqueeze(0) + 1e-7)) - torch.atan(w1.unsqueeze(1) / (h1.unsqueeze(1) + 1e-7)), 2
    )

    # Alpha trade-off parameter
    with torch.no_grad():
        alpha = v / (1 - iou + v + 1e-7)

    ciou = iou - diou_term - alpha * v
    return ciou


def masks_to_boxes(masks):
    """Compute the bounding boxes around the provided masks

    The masks should be in format [N, H, W] where N is the number of masks, (H, W) are the spatial dimensions.

    Returns a [N, 4] tensors, with the boxes in xyxy format
    """
    if masks.numel() == 0:
        return torch.zeros((0, 4), device=masks.device)

    h, w = masks.shape[-2:]

    y = torch.arange(0, h, dtype=torch.float)
    x = torch.arange(0, w, dtype=torch.float)
    y, x = torch.meshgrid(y, x)

    x_mask = masks * x.unsqueeze(0)
    x_max = x_mask.flatten(1).max(-1)[0]
    x_min = x_mask.masked_fill(~(masks.bool()), 1e8).flatten(1).min(-1)[0]

    y_mask = masks * y.unsqueeze(0)
    y_max = y_mask.flatten(1).max(-1)[0]
    y_min = y_mask.masked_fill(~(masks.bool()), 1e8).flatten(1).min(-1)[0]

    return torch.stack([x_min, y_min, x_max, y_max], 1)


if __name__ == "__main__":
    x = torch.rand(5, 4)
    y = torch.rand(3, 4)
    iou, union = box_iou(x, y)
    import ipdb

    ipdb.set_trace()
