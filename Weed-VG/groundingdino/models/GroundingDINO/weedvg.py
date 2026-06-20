"""Weed-VG Thin Wrapper.

Provides utility helpers around a frozen (optionally partially trainable)
GroundingDINO model so the training code can consistently extract encoder
features, multi-scale tensors, and proposal metadata. The original efficient
decoder stage has been removed; downstream tasks now consume the native
GroundingDINO decoder outputs directly.
"""
from typing import List, Sequence
import torch
import torch.nn as nn
import torch.nn.functional as F
from groundingdino.models.GroundingDINO.ms_deform_attn import MultiScaleDeformableAttention
from groundingdino.util.misc import nested_tensor_from_tensor_list
import math



class TextProjectionHead(nn.Module):
    """Projects text embeddings into the visual feature space."""

    def __init__(self, input_dim, output_dim, hidden_dim=None, use_layernorm=True):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = input_dim
        self.net = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, output_dim))
        self.norm = nn.LayerNorm(output_dim) if use_layernorm else None

    def forward(self, x):
        x = self.net(x)
        if self.norm is not None:
            x = self.norm(x)
        return F.normalize(x, p=2, dim=-1, eps=1e-06)

class VisualProjectionHead(nn.Module):
    """Projects visual/query features into a compact joint embedding space.
    Updated to use a 2-layer MLP (SimCLR/MoCo style) for better decoupling.
    """

    def __init__(self, input_dim, output_dim, hidden_dim=None, use_layernorm=True):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = input_dim
        self.net = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, output_dim))
        self.norm = nn.LayerNorm(output_dim) if use_layernorm else None

    def forward(self, x):
        x = self.net(x)
        if self.norm is not None:
            x = self.norm(x)
        return F.normalize(x, p=2, dim=-1, eps=1e-06)

class WeedVG(nn.Module):
    """Wrapper around GroundingDINO for Weed-VG training flows."""

    def __init__(self, grounding_dino_encoder, temperature=0.07):
        super().__init__()
        self.grounding_dino = grounding_dino_encoder
        self.grounding_trainable = False
        self.set_grounding_trainable(False)
        self.text_projector = None
        self.visual_projector = None
        self.temperature = float(temperature)

    def train(self, mode: bool=True):
        """Override default train() to keep frozen submodules in eval mode."""
        super().train(mode)
        if self.grounding_trainable:
            self.grounding_dino.train(mode)
            if hasattr(self.grounding_dino, 'model'):
                self.grounding_dino.model.train(mode)
        else:
            self.grounding_dino.eval()
            if hasattr(self.grounding_dino, 'model'):
                self.grounding_dino.model.eval()
        if self.grounding_trainable:
            for head_module in self._prediction_head_modules():
                head_module.train(mode)
            base_model = getattr(self.grounding_dino, 'model', self.grounding_dino)
            transformer = getattr(base_model, 'transformer', None)
            decoder = getattr(transformer, 'decoder', None) if transformer is not None else None
            if decoder is not None:
                try:
                    any_trainable = any((p.requires_grad for p in decoder.parameters()))
                except Exception:
                    any_trainable = False
                if any_trainable:
                    decoder.train(mode)
        return self

    def set_grounding_trainable(self, trainable: bool, freeze_box_head: bool=False, train_decoder: bool=False):
        """Enable or disable gradient flow through the GroundingDINO encoder.

        Args:
            trainable: If True, enable training for prediction heads (excluding box head if freeze_box_head=True).
            freeze_box_head: If True, keep the box head frozen even when trainable=True.
            train_decoder: If True, also unfreeze the transformer decoder layers.
        """
        self.grounding_trainable = bool(trainable)
        for param in self.grounding_dino.parameters():
            param.requires_grad = False
        if hasattr(self.grounding_dino, 'model'):
            for param in self.grounding_dino.model.parameters():
                param.requires_grad = False
        if self.grounding_trainable:
            head_modules = self._prediction_head_modules()
            print(f'[debug] Found {len(head_modules)} prediction head modules (primary lookup)')
            total_head_params = 0
            for i, head_module in enumerate(head_modules):
                if freeze_box_head and hasattr(head_module, '__class__') and ('bbox' in str(head_module.__class__).lower()):
                    print(f'[debug]   Skipping box head module {i} (freeze_box_head=True)')
                    continue
                module_params = sum((p.numel() for p in head_module.parameters()))
                total_head_params += module_params
                print(f'[debug]   Head module {i}: {module_params:,} parameters')
                for param in head_module.parameters():
                    param.requires_grad = True
            if train_decoder:
                base_model = getattr(self.grounding_dino, 'model', self.grounding_dino)
                transformer = getattr(base_model, 'transformer', None)
                if transformer is not None:
                    decoder = getattr(transformer, 'decoder', None)
                    if decoder is not None:
                        print('[debug] Unfreezing transformer decoder parameters')
                        decoder_params = 0
                        for param in decoder.parameters():
                            param.requires_grad = True
                            decoder_params += param.numel()

    def _prediction_head_modules(self):
        """Collect lightweight prediction heads that may be fine-tuned."""
        modules = []
        base_model = getattr(self.grounding_dino, 'model', self.grounding_dino)
        candidate_attrs = [(base_model, 'bbox_embed'), (base_model, 'class_embed')]
        transformer = getattr(base_model, 'transformer', None)
        if transformer is not None:
            candidate_attrs.extend([(transformer, 'enc_out_bbox_embed'), (transformer, 'enc_out_class_embed')])
            decoder = getattr(transformer, 'decoder', None)
            if decoder is not None:
                candidate_attrs.extend([(decoder, 'bbox_embed'), (decoder, 'class_embed')])
        seen = set()
        for module_obj, attr in candidate_attrs:
            if module_obj is None:
                continue
            mod = getattr(module_obj, attr, None)
            if mod is None:
                continue
            if isinstance(mod, (list, tuple)):
                iterable = mod
            elif isinstance(mod, nn.ModuleList):
                iterable = list(mod)
            else:
                iterable = [mod]
            for item in iterable:
                if not isinstance(item, nn.Module):
                    continue
                if id(item) in seen:
                    continue
                seen.add(id(item))
                modules.append(item)
        return modules

    def forward_grounding_dino(self, images, captions):
        """Extract features using GroundingDINO encoder (frozen or trainable).

        GroundingDINO's forward supports passing either:
          - a list/tuple of image tensors [C,H,W] (preferred; enables padding masks)
          - a batched tensor [B,C,H,W]

        IMPORTANT: When passing a batched tensor, we convert it to a list to
        ensure NestedTensor masks have consistent batch dimension across feature
        levels. This avoids rare-but-real shape mismatches in mask flattening.
        """
        if isinstance(images, (list, tuple)):
            image_list = list(images)
        elif isinstance(images, torch.Tensor) and images.dim() == 4:
            image_list = list(images.unbind(0))
        else:
            raise ValueError(f'Images must be a list/tuple of [C,H,W] tensors or a [B,C,H,W] tensor; got {type(images)}')
        if not image_list:
            raise ValueError('No images provided')
        for i, im in enumerate(image_list):
            if not isinstance(im, torch.Tensor) or im.dim() != 3:
                raise ValueError(f"Image[{i}] must be a tensor [C,H,W], got {type(im)} with shape {getattr(im, 'shape', None)}")
        if captions is not None and isinstance(captions, (list, tuple)):
            if len(captions) != len(image_list):
                raise ValueError(f'captions length ({len(captions)}) must match batch size ({len(image_list)})')
        elif captions is not None and isinstance(captions, torch.Tensor):
            if captions.dim() != 2:
                raise ValueError(f'Captions must be [B, seq_len] or list, got shape {captions.shape}')
        elif captions is not None:
            raise ValueError(f'captions must be list/tuple or tensor, got {type(captions)}')
        device = image_list[0].device
        samples = nested_tensor_from_tensor_list(image_list)
        was_training = self.grounding_dino.training
        inner_was_training = hasattr(self.grounding_dino, 'model') and self.grounding_dino.model.training
        if not self.grounding_trainable:
            self.grounding_dino.eval()
            if hasattr(self.grounding_dino, 'model'):
                self.grounding_dino.model.eval()
        try:
            with torch.set_grad_enabled(self.grounding_trainable):
                outputs = self.grounding_dino(samples, captions=captions, return_backbone_features=True, return_multi_scale=True, unset_image_tensor=False)
        except Exception as e:
            raise RuntimeError(f'GroundingDINO feature extraction failed: {e}')
        finally:
            if not self.grounding_trainable:
                if was_training:
                    self.grounding_dino.train()
                if inner_was_training and hasattr(self.grounding_dino, 'model'):
                    self.grounding_dino.model.train()
        if outputs is None:
            raise RuntimeError('GroundingDINO returned None outputs')
        encoder_features = outputs.get('encoder_features')
        multi_scale_features = outputs.get('multi_scale_features')
        if multi_scale_features is not None and len(multi_scale_features) > 0:
            features = multi_scale_features
            if not isinstance(features, (list, tuple)):
                raise ValueError('multi_scale_features must be a list or tuple')
            spatial_shapes = []
            level_start_index = [0]
            for i, feat in enumerate(features):
                if not hasattr(feat, 'shape') or len(feat.shape) != 4:
                    raise ValueError(f"Feature {i} must be a 4D tensor [B, C, H, W], got shape {getattr(feat, 'shape', 'no shape')}")
                B, C, H, W = feat.shape
                spatial_shapes.append([H, W])
                if i > 0:
                    prev_H, prev_W = spatial_shapes[i - 1]
                    level_start_index.append(level_start_index[-1] + prev_H * prev_W)
            spatial_shapes = torch.tensor(spatial_shapes, device=device, dtype=torch.long)
            level_start_index = torch.tensor(level_start_index, device=device, dtype=torch.long)
            flattened_features = []
            for feat in features:
                B, C, H, W = feat.shape
                flattened_features.append(feat.flatten(2).permute(0, 2, 1))
            if not flattened_features:
                raise ValueError('No valid features found in multi_scale_features')
            encoder_features = torch.cat(flattened_features, dim=1)
            expected_total_positions = sum((H * W for H, W in spatial_shapes.tolist()))
            if encoder_features.shape[1] != expected_total_positions:
                raise ValueError(f'Feature concatenation mismatch: expected {expected_total_positions} positions, got {encoder_features.shape[1]}')
        else:
            if isinstance(encoder_features, list):
                encoder_features = encoder_features[0] if encoder_features else None
            if encoder_features is None or encoder_features.dim() != 4:
                raise ValueError(f"Single-scale encoder_features must be a 4D tensor, got {type(encoder_features)} with shape {getattr(encoder_features, 'shape', 'no shape')}")
            B, C, H, W = encoder_features.shape
            spatial_shapes = torch.tensor([[H, W]], device=images.device, dtype=torch.long)
            level_start_index = torch.tensor([0], device=images.device, dtype=torch.long)
            encoder_features = encoder_features.flatten(2).permute(0, 2, 1)
        return {'encoder_features': encoder_features, 'multi_scale_features': multi_scale_features, 'spatial_shapes': spatial_shapes, 'level_start_index': level_start_index, 'pred_boxes': outputs.get('pred_boxes'), 'pred_logits': outputs.get('pred_logits'), 'decoder_queries': outputs.get('decoder_queries')}


    def print_summary(self):
        """Print a formatted summary of the model parameters."""
        print('=' * 60)
        print('Weed-VG Model Summary')
        print('=' * 60)
        total_params = sum((p.numel() for p in self.parameters()))
        trainable_params = sum((p.numel() for p in self.parameters() if p.requires_grad))
        print(f"{'Component':<30} | {'Params':>12} | {'Trainable':>10}")
        print('-' * 60)
        for name, child in self.named_children():
            c_total = sum((p.numel() for p in child.parameters()))
            c_train = sum((p.numel() for p in child.parameters() if p.requires_grad))
            print(f'{name:<30} | {c_total:>12,} | {c_train:>10,}')
        print('-' * 60)
        print(f"{'TOTAL':<30} | {total_params:>12,} | {trainable_params:>10,}")
        print('=' * 60)

    @torch.no_grad()
    def forward_inference(self, image_tensor, boxes, text, query_features: torch.Tensor | None=None):
        """Compute similarity scores between a sentence and given proposals.

        Args:
            image_tensor: normalized image tensor [1, 3, H, W].
            boxes: proposal boxes [1, N, 4] or [N, 4] in normalized cxcywh.
            text: sentence string.

        Returns:
            scores: [N] similarity scores.
            refined_boxes: [N, 4] (cx, cy, w, h) possibly refined.
        """
        if image_tensor.dim() == 3:
            image_tensor = image_tensor.unsqueeze(0)
        if boxes.dim() == 2:
            boxes = boxes.unsqueeze(0)
        device = image_tensor.device
        boxes = boxes.to(device)
        ref_boxes = boxes
        if query_features is None:
            gd_out = self.forward_grounding_dino(image_tensor, captions=[text])
            q = gd_out.get('decoder_queries')
            if q is not None:
                query_features = q
            else:
                encoder_features = gd_out['encoder_features']
                B, num_queries, _ = boxes.shape
                query_features = encoder_features[:, :num_queries, :]
        else:
            if query_features.dim() == 2:
                query_features = query_features.unsqueeze(0)
            query_features = query_features.to(device)
        if self.visual_projector is not None:
            visual_emb = self.visual_projector(query_features).squeeze(0)
        else:
            visual_emb = F.normalize(query_features.squeeze(0), p=2, dim=-1)
        if self.text_projector is None:
            raise RuntimeError('WeedVG.text_projector must be set with a text encoder + projection head for inference.')
        if hasattr(self.text_projector, 'encode_one'):
            text_emb = self.text_projector.encode_one(text).to(device)
        else:
            raise RuntimeError('text_projector missing encode_one(text) method; please adapt it for inference.')
        text_emb = F.normalize(text_emb, p=2, dim=-1)
        scores = (visual_emb @ text_emb).clamp(-1.0, 1.0)
        return (scores, ref_boxes.squeeze(0))

class DeformableFeatureRefiner(nn.Module):
    """Lightweight MS-DeformAttn head that refines query features using multi-scale context."""

    def __init__(self, embed_dim: int, num_heads: int, num_levels: int, num_points: int, num_layers: int=1, ffn_dim: int=1024, dropout: float=0.1) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_levels = num_levels
        self.num_layers = max(1, int(num_layers))
        self.layers = nn.ModuleList()
        self.dropout = nn.Dropout(dropout)
        for _ in range(self.num_layers):
            block = nn.ModuleDict({'attn1': MultiScaleDeformableAttention(embed_dim=embed_dim, num_heads=num_heads, num_levels=num_levels, num_points=num_points, batch_first=True), 'norm1': nn.LayerNorm(embed_dim), 'ffn': nn.Sequential(nn.Linear(embed_dim, ffn_dim), nn.Linear(ffn_dim, ffn_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(ffn_dim, embed_dim)), 'norm3': nn.LayerNorm(embed_dim)})
            self.layers.append(block)

    def forward(self, query_features: torch.Tensor, feature_pyramid: Sequence[torch.Tensor] | torch.Tensor | None, reference_boxes: torch.Tensor) -> torch.Tensor:
        if feature_pyramid is None:
            return query_features
        pyramid = self._ensure_tensor_list(feature_pyramid)
        if not pyramid:
            return query_features
        pyramid = self._adjust_pyramid_levels(pyramid, self.num_levels)
        value, spatial_shapes, level_start_index = self._flatten_pyramid(pyramid, query_features.device)
        if spatial_shapes.shape[0] != self.num_levels:
            return query_features
        ref_boxes = torch.nan_to_num(reference_boxes, nan=0.5, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
        reference_points = ref_boxes.unsqueeze(2).expand(-1, -1, self.num_levels, -1)
        refined_features = query_features
        for layer in self.layers:
            attn_out1 = layer['attn1'](query=refined_features, value=value, reference_points=reference_points, spatial_shapes=spatial_shapes, level_start_index=level_start_index)
            enhanced1 = layer['norm1'](refined_features + self.dropout(attn_out1))
            ffn_out = layer['ffn'](enhanced1)
            refined_features = layer['norm3'](enhanced1 + self.dropout(ffn_out))
        return refined_features

    @staticmethod
    def _adjust_pyramid_levels(pyramid: List[torch.Tensor], num_levels: int) -> List[torch.Tensor]:
        if num_levels <= 0:
            return pyramid
        if len(pyramid) == num_levels:
            return pyramid
        if len(pyramid) > num_levels:
            return list(pyramid[:num_levels])
        out = list(pyramid)
        while len(out) < num_levels:
            last = out[-1]
            if hasattr(last, 'tensors'):
                last = last.tensors
            if not torch.is_tensor(last) or last.dim() != 4:
                break
            _, _, h, w = last.shape
            if h < 2 or w < 2:
                next_level = last
            else:
                next_level = F.max_pool2d(last, kernel_size=2, stride=2)
            out.append(next_level)
        if len(out) < num_levels and out:
            out.extend([out[-1]] * (num_levels - len(out)))
        return out

    @staticmethod
    def _ensure_tensor_list(feature_pyramid: Sequence[torch.Tensor] | torch.Tensor | None) -> List[torch.Tensor]:
        if feature_pyramid is None:
            return []
        if isinstance(feature_pyramid, torch.Tensor):
            return [feature_pyramid]
        return list(feature_pyramid)

    def _flatten_pyramid(self, pyramid: Sequence[torch.Tensor], device: torch.device) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
        flattened: List[torch.Tensor] = []
        spatial_shapes: List[list[int]] = []
        for feat in pyramid:
            tensor = feat
            if hasattr(tensor, 'tensors'):
                tensor = tensor.tensors
            if tensor.dim() != 4:
                raise ValueError(f'Expected feature map shaped [B,C,H,W], got {tensor.shape}')
            B, C, H, W = tensor.shape
            flattened.append(tensor.flatten(2).transpose(1, 2))
            spatial_shapes.append([H, W])
        if not flattened:
            return None, None, None
        value = torch.cat(flattened, dim=1)
        shapes_tensor = torch.as_tensor(spatial_shapes, device=device, dtype=torch.long)
        starts = torch.cat([shapes_tensor.new_zeros(1), torch.cumsum((shapes_tensor[:, 0] * shapes_tensor[:, 1])[:-1], dim=0)])
        return value, shapes_tensor, starts

def build_weedvg(grounding_dino_encoder, temperature=0.07, **_unused_kwargs):
    """Build Weed-VG wrapper using the original GroundingDINO decoder outputs.

    Args:
        grounding_dino_encoder: GroundingDINO model
        temperature: Temperature for similarity score scaling (default 0.07, must match training)
    """
    model = WeedVG(grounding_dino_encoder=grounding_dino_encoder, temperature=temperature)
    return model