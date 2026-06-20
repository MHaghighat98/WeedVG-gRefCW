"""Contrastive Relevance Scorer (CRS) and related utilities.

Shared between training and evaluation scripts to avoid code duplication.
"""

import math

import torch
import torch.nn.functional as F


class SineBoxPositionalEmbedding(torch.nn.Module):
    """DETR-style sine/cos positional embedding for boxes.

    Input boxes are expected in normalized cxcywh in [0,1].
    Output is a D-dim embedding that can be added to query features.
    """

    def __init__(self, dim: int, temperature: float = 10000.0, scale: float | None = None):
        super().__init__()
        dim = int(dim)
        if dim <= 0:
            raise ValueError("box positional embedding dim must be > 0")
        if dim % 8 != 0:
            raise ValueError(f"box positional embedding dim must be divisible by 8, got {dim}")
        self.dim = dim
        self.temperature = float(temperature)
        self.scale = float(scale) if scale is not None else float(2.0 * math.pi)

    def _encode_1d(self, x: torch.Tensor, d: int) -> torch.Tensor:
        x = x.clamp(0.0, 1.0) * self.scale
        half = d // 2
        dim_t = torch.arange(half, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2.0 * (dim_t / float(half)))
        pos = x[:, None] / dim_t[None, :]
        return torch.cat([pos.sin(), pos.cos()], dim=1)

    def forward(self, boxes_cxcywh: torch.Tensor) -> torch.Tensor:
        if boxes_cxcywh is None or boxes_cxcywh.numel() == 0:
            return boxes_cxcywh
        if boxes_cxcywh.dim() != 2 or boxes_cxcywh.shape[-1] < 4:
            raise ValueError(f"boxes must be [N,4+] cxcywh, got shape {getattr(boxes_cxcywh, 'shape', None)}")
        boxes = boxes_cxcywh[:, :4].to(dtype=torch.float32)
        d_each = self.dim // 4
        cx = self._encode_1d(boxes[:, 0], d_each)
        cy = self._encode_1d(boxes[:, 1], d_each)
        w = self._encode_1d(boxes[:, 2], d_each)
        h = self._encode_1d(boxes[:, 3], d_each)
        out = torch.cat([cx, cy, w, h], dim=1)
        return out.to(dtype=boxes_cxcywh.dtype, device=boxes_cxcywh.device)


class ContrastiveRelevanceScorer(torch.nn.Module):
    """CRS: Contrastive Relevance Scoring module.

    Architecture:
      self_attn -> norm0 -> mhca (cross-attn) -> norm1 -> ffn -> norm2
      -> gate_mlp -> cosine similarity with temperature scaling.

    Supports both batched [B, N, D] and unbatched [N, D] inputs.
    Ablation scoring modes ('sentence_only', 'word_only', 'no_proj') can be
    activated by setting ``self._scoring_mode`` on the instance.
    """

    def __init__(self, dim: int, *, num_heads: int = 8, mlp_hidden: int = 256, init_temperature: float = 0.07):
        super().__init__()
        dim = int(dim)
        num_heads = int(num_heads)
        if dim <= 0:
            raise ValueError("CRS dim must be > 0")
        if num_heads <= 0:
            raise ValueError("CRS num_heads must be > 0")
        if dim % num_heads != 0:
            raise ValueError(f"CRS dim={dim} must be divisible by num_heads={num_heads}")

        self.self_attn = torch.nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)
        self.mhca = torch.nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)
        self.dropout = torch.nn.Dropout(0.1)
        self.norm0 = torch.nn.LayerNorm(dim)
        self.norm1 = torch.nn.LayerNorm(dim)
        self.norm2 = torch.nn.LayerNorm(dim)
        ffn_hidden_dim = 1024
        self.ffn = torch.nn.Sequential(
            torch.nn.Linear(dim, ffn_hidden_dim),
            torch.nn.GELU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(ffn_hidden_dim, dim),
        )
        self.gate_mlp = torch.nn.Sequential(
            torch.nn.Linear(dim * 2, int(mlp_hidden)),
            torch.nn.GELU(),
            torch.nn.Linear(int(mlp_hidden), 1),
        )
        self.log_temperature = torch.nn.Parameter(torch.tensor(float(init_temperature)).log())
        self.box_pos = SineBoxPositionalEmbedding(dim)

    def temperature(self) -> torch.Tensor:
        return self.log_temperature.exp().clamp(min=0.01, max=100.0)

    def forward(self, query_feats, token_feats, token_pad_mask, query_boxes=None):
        """Compute CRS logits. Supports both batched [B, N, D] and unbatched [N, D] inputs."""
        is_batched = query_feats.dim() == 3
        if is_batched:
            B = query_feats.shape[0]
            all_logits = []
            all_ws = []
            for b in range(B):
                qb = query_feats[b]
                tb = token_feats[b] if token_feats.dim() == 3 else token_feats
                pm = token_pad_mask[b] if token_pad_mask.dim() == 2 else token_pad_mask
                boxes_b = query_boxes[b] if query_boxes is not None and query_boxes.dim() == 3 else query_boxes
                logits_b, ws_b = self._forward_single(qb, tb, pm, boxes_b)
                all_logits.append(logits_b)
                all_ws.append(ws_b)
            return torch.stack(all_logits), torch.stack(all_ws)
        else:
            return self._forward_single(query_feats, token_feats, token_pad_mask, query_boxes)

    def _forward_single(self, q, t, token_pad_mask, query_boxes=None):
        """Single-sample forward."""
        if q is None or q.numel() == 0:
            return q.new_zeros((0,)), q.new_zeros(())
        if t is None or t.numel() == 0:
            return q.new_zeros((q.shape[0],)), q.new_full((), 0.5)

        if q.dim() != 2:
            q = q.reshape(-1, q.shape[-1])
        if t.dim() != 2:
            t = t.reshape(-1, t.shape[-1])
        token_pad_mask = token_pad_mask.to(dtype=torch.bool).reshape(-1)

        # Add box positional embedding
        if query_boxes is not None and torch.is_tensor(query_boxes) and query_boxes.numel() > 0:
            qb = query_boxes
            if qb.dim() != 2:
                qb = qb.reshape(-1, qb.shape[-1])
            if qb.shape[-1] >= 4 and qb.shape[0] == q.shape[0]:
                try:
                    q = q + self.box_pos(qb[:, :4].to(dtype=q.dtype, device=q.device))
                except Exception:
                    pass

        q_b = q.unsqueeze(0)
        t_b = t.unsqueeze(0)
        pad_b = token_pad_mask.unsqueeze(0)

        # Self-attention (inter-proposal)
        self_out, _ = self.self_attn(q_b, q_b, q_b, need_weights=False)
        q_b = self.norm0(q_b + self.dropout(self_out))

        # Cross-attention (query -> text)
        scoring_mode = getattr(self, '_scoring_mode', 'full')
        if scoring_mode != 'no_proj':
            attn_out, _ = self.mhca(q_b, t_b, t_b, key_padding_mask=pad_b)
            q_b = self.norm1(q_b + self.dropout(attn_out))
        else:
            q_b = self.norm1(q_b)

        # FFN
        ffn_out = self.ffn(q_b)
        q_b = self.norm2(q_b + self.dropout(ffn_out))

        q_p = q_b.squeeze(0)

        # Global text feature via valid-mask max pooling
        valid = ~token_pad_mask
        if not bool(valid.any()):
            fs = t.new_zeros((t.shape[1],))
        else:
            neg_inf = torch.finfo(t.dtype).min
            t_masked = t.masked_fill(~valid.unsqueeze(-1), neg_inf)
            fs = t_masked.max(dim=0)[0]
            fs = torch.nan_to_num(fs, nan=0.0, posinf=0.0, neginf=0.0)

        # Gate
        fs_rep = fs.unsqueeze(0).expand(q_p.shape[0], -1)
        wq = torch.sigmoid(self.gate_mlp(torch.cat([q_p, fs_rep], dim=-1))).squeeze(-1)
        ws = wq.mean()

        # Cosine similarities
        qn = F.normalize(q_p, p=2, dim=-1)
        fsn = F.normalize(fs, p=2, dim=-1)
        tn = F.normalize(t, p=2, dim=-1)

        temp = self.temperature()
        s_sent = (qn * fsn.unsqueeze(0)).sum(dim=-1) / temp
        s_word = (qn @ tn.t()) / temp
        s_word_max = s_word.max(dim=1)[0]

        # Ablation scoring modes
        if scoring_mode == 'sentence_only':
            logits = s_sent
        elif scoring_mode == 'word_only':
            logits = s_word_max
        else:
            logits = wq * s_sent + (1.0 - wq) * s_word_max
        return logits, ws


def sort_and_trim_proposals(pred_boxes, pred_logits, max_proposals):
    """Sort GroundingDINO proposals by confidence (descending) and keep top-K.

    Args:
        pred_boxes: [B, N, 4] predicted boxes
        pred_logits: [B, N, C] predicted logits
        max_proposals: maximum number of proposals to keep

    Returns:
        pred_boxes, pred_logits, proposal_scores (all trimmed to top-K)
    """
    if pred_boxes is None or pred_logits is None:
        return pred_boxes, pred_logits, None

    proposal_scores = pred_logits.sigmoid().max(dim=-1)[0]
    sorted_scores, sorted_indices = proposal_scores.sort(dim=1, descending=True)

    gather_idx_boxes = sorted_indices.unsqueeze(-1).expand(-1, -1, pred_boxes.shape[-1])
    gather_idx_logits = sorted_indices.unsqueeze(-1).expand(-1, -1, pred_logits.shape[-1])
    pred_boxes = torch.gather(pred_boxes, 1, gather_idx_boxes)
    pred_logits = torch.gather(pred_logits, 1, gather_idx_logits)
    proposal_scores = sorted_scores

    top_k = max(1, min(int(max_proposals), int(pred_boxes.shape[1])))
    if pred_boxes.shape[1] > top_k:
        pred_boxes = pred_boxes[:, :top_k]
        pred_logits = pred_logits[:, :top_k]
        proposal_scores = proposal_scores[:, :top_k]

    return pred_boxes, pred_logits, proposal_scores
