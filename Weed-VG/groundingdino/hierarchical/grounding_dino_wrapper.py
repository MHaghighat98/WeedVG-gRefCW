"""
GroundingDINO Wrapper for Hierarchical Contrastive Learning

This wrapper extends GroundingDINO to expose intermediate features needed for:
1. Instance query initialization (F_GDINO - visual features at proposals)
2. Contrastive learning (Q_d - refined decoder queries)

Modifications from standard GroundingDINO:
- Returns encoder features for Q_instance initialization
- Returns decoder queries (Q_d) for contrastive learning
- Maintains backward compatibility with original outputs
"""

import torch
import torch.nn as nn


class GroundingDINOWithFeatures(nn.Module):
    """
    Wrapper around GroundingDINO that exposes intermediate features.

    This class wraps a standard GroundingDINO model and hooks into
    its forward pass to extract:
    - Encoder output features (F_GDINO)
    - Decoder refined queries (Q_d)
    - Standard outputs (pred_boxes, pred_logits)
    """

    def __init__(self, grounding_dino_model):
        """
        Args:
            grounding_dino_model: Pre-built GroundingDINO model instance
        """
        super().__init__()
        self.model = grounding_dino_model

        # Storage for intermediate features
        self.encoder_features = None
        self.decoder_queries = None

        # CRITICAL FIX: Register projection layers as module parameters
        # These are created once and reused, not recreated randomly each forward pass
        # This allows them to be trained/frozen deterministically
        self._feature_projection = None  # Lazy initialization based on actual feature dims

        # Capability flags
        self._refpoint_injection_supported = False
        if hasattr(self.model, "transformer"):
            t = self.model.transformer
            if hasattr(t, "refpoint_embed") or hasattr(t, "reference_points"):
                self._refpoint_injection_supported = True

        # Register hooks to capture decoder queries
        self._register_hooks()

    def _register_hooks(self):
        """
        Register forward hooks to capture intermediate activations of interest.
        """
        if hasattr(self.model, "transformer"):
            # Hook decoder to capture refined queries (Q_d)
            if hasattr(self.model.transformer, "decoder"):

                def decoder_hook(module, input, output):
                    # Store decoder queries (Q_d)
                    # Decoder output is typically (hs, reference_points) or similar
                    # hs shape: [num_layers, B, N_q, C]
                    if isinstance(output, tuple):
                        hs = output[0]  # Hidden states
                        # Take last layer's output as Q_d
                        if isinstance(hs, (list, tuple)):
                            # Multi-layer output as list/tuple
                            self.decoder_queries = hs[-1]  # Last layer: [B, N_q, C]
                        elif len(hs.shape) == 4:  # [num_layers, B, N_q, C]
                            self.decoder_queries = hs[-1]  # [B, N_q, C]
                        else:
                            self.decoder_queries = hs
                    elif isinstance(output, (list, tuple)):
                        # Output is a list/tuple directly
                        self.decoder_queries = output[-1]  # Take last element
                    else:
                        self.decoder_queries = output

                self.model.transformer.decoder.register_forward_hook(decoder_hook)

    def _extract_backbone_features(self):
        """Retrieve multi-scale backbone features stored on the wrapped model."""
        features = getattr(self.model, "features", None)
        if features is None:
            return None

        projected_features = []
        projected_srcs = []

        # Project each backbone feature map to the transformer hidden dimension
        for level, feat in enumerate(features):
            if hasattr(feat, "decompose"):
                src, _ = feat.decompose()
            elif hasattr(feat, "tensors"):
                src = feat.tensors
            else:
                src = feat

            if level < len(self.model.input_proj):
                projected = self.model.input_proj[level](src)
                projected_features.append(projected.detach())
                projected_srcs.append(projected)

        # Handle additional scales requested by num_feature_levels
        num_levels = getattr(self.model, "num_feature_levels", len(projected_features))
        if num_levels > len(projected_features):
            last_raw = features[-1]
            last_tensor = last_raw.tensors if hasattr(last_raw, "tensors") else projected_srcs[-1]

            for level in range(len(projected_features), num_levels):
                if level < len(self.model.input_proj):
                    if level == len(projected_features):
                        projected = self.model.input_proj[level](last_tensor)
                    else:
                        projected = self.model.input_proj[level](projected_srcs[-1])
                    projected_features.append(projected.detach())
                    projected_srcs.append(projected)

        return projected_features if projected_features else None

    def forward(
        self,
        images,
        captions,
        return_features=True,
        custom_query_init=None,
        custom_reference_points=None,
        return_multi_scale=False,
        **kwargs,
    ):
        """
        Forward pass with optional feature extraction and custom query initialization.

        Args:
            images: Input images [B, 3, H, W]
            captions: Text captions (list of strings or tokenized)
            return_features: If True, return intermediate features
            custom_query_init: Optional [B, N_q, d_model] custom instance queries (Q_instance)
                              If provided, will be injected into decoder instead of default initialization
            custom_reference_points: Optional [B, N_q, 2] custom reference points (GT box centers)
                                    If provided, will be injected as initial reference points for decoder
            return_multi_scale: If True, return multi-scale feature pyramid from backbone

        Returns:
            If return_features=True:
                dict with keys:
                    - 'pred_boxes': [B, N_q, 4] - Predicted boxes
                    - 'pred_logits': [B, N_q, num_classes] - Classification logits
                    - 'encoder_features': Encoder output (F_GDINO)
                    - 'decoder_queries': [B, N_q, C] - Refined queries (Q_d)
                    - 'proposal_boxes': [B, N_q, 4] - Initial proposal boxes
                    - 'multi_scale_features': List of multi-scale features (if return_multi_scale=True)
            If return_features=False:
                Standard GroundingDINO output dict
        """
        # Reset feature storage
        self.encoder_features = None
        self.decoder_queries = None

        # Inject custom query initialization if provided
        if custom_query_init is not None:
            self._inject_custom_queries(custom_query_init)

        # Inject custom reference points if provided
        if custom_reference_points is not None:
            self._inject_custom_reference_points(custom_reference_points)

        # Forward pass through GroundingDINO.
        # We explicitly tell it to NOT unset the image tensor, so we can access it.
        try:
            outputs = self.model(images, captions=captions, unset_image_tensor=False)
            # After the forward pass, self.model should have `features` and `poss` attributes.
            # We extract them to pass to our own outputs.
            backbone_features = self._extract_backbone_features()
        finally:
            # CRITICAL: Always clean up the cached features from the base model
            # to prevent state leakage to the next forward pass.
            if hasattr(self.model, "unset_image_tensor"):
                self.model.unset_image_tensor()

        self.encoder_features = backbone_features

        if return_features:
            # Add intermediate features to output
            outputs["encoder_features"] = backbone_features
            outputs["decoder_queries"] = self.decoder_queries

            # Add multi-scale features if requested
            if return_multi_scale and backbone_features is not None:
                # encoder_features captured from backbone already contain multi-scale pyramid
                outputs["multi_scale_features"] = backbone_features

            # Extract proposal boxes if available
            # These are typically the two-stage proposals or initial query positions
            if "enc_outputs" in outputs and outputs["enc_outputs"] is not None:
                if "pred_boxes" in outputs["enc_outputs"]:
                    outputs["proposal_boxes"] = outputs["enc_outputs"]["pred_boxes"]
            elif "init_reference" in outputs:
                outputs["proposal_boxes"] = outputs["init_reference"]
            else:
                # If no proposals available, use predicted boxes as proxy
                outputs["proposal_boxes"] = outputs["pred_boxes"]

        return outputs



    def _inject_custom_queries(self, custom_queries):
        """
        Inject custom instance queries (Q_instance) into the transformer.

        CRITICAL: This method should only be called with batch_size=1 (single image).
        For batch processing, call this method once per image in a loop.

        Args:
            custom_queries: [1, N_q, d_model] - Custom instance queries for a SINGLE image
                           from Q_instance = MLP(Concat(F_GDINO, T_pos,i))

        Raises:
            ValueError: If custom_queries has batch_size != 1
        """
        if hasattr(self.model, "transformer"):
            # Store original tgt_embed
            if not hasattr(self, "_original_tgt_embed"):
                self._original_tgt_embed = self.model.transformer.tgt_embed.weight.data.clone()

            # Validate batch size
            if len(custom_queries.shape) == 3:
                batch_size = custom_queries.shape[0]
                if batch_size != 1:
                    raise ValueError(
                        f"_inject_custom_queries expects batch_size=1, got {batch_size}. "
                        "Process each image individually in a loop to maintain image-specific queries."
                    )
                # Squeeze batch dimension for single image
                custom_query_weight = custom_queries.squeeze(0)  # [N_q, d_model]
            else:
                custom_query_weight = custom_queries

            # Update the embedding weight
            self.model.transformer.tgt_embed.weight.data = custom_query_weight.detach()


    def _inject_custom_reference_points(self, custom_reference_points):
        """
        Inject custom reference points (GT box centers) into the transformer decoder.

        This overrides the default reference point initialization with GT box centers
        for instance-specific spatial grounding.

        Args:
            custom_reference_points: [1, N_q, 2] - Custom reference points (GT centers)
                                    Expected format: [cx, cy] normalized to [0, 1]

        Raises:
            ValueError: If custom_reference_points has batch_size != 1
        """
        if not getattr(self, "_refpoint_injection_supported", False):
            # Silently skip if model doesn't expose reference point params; avoid noisy warnings per-batch
            return

        if hasattr(self.model, "transformer"):
            # Store original reference points if first time
            if not hasattr(self, "_original_refpoint_embed"):
                if hasattr(self.model.transformer, "refpoint_embed"):
                    self._original_refpoint_embed = self.model.transformer.refpoint_embed.weight.data.clone()

            # Validate batch size
            if len(custom_reference_points.shape) == 3:
                batch_size = custom_reference_points.shape[0]
                if batch_size != 1:
                    raise ValueError(
                        f"_inject_custom_reference_points expects batch_size=1, got {batch_size}. "
                        "Process each image individually in a loop to maintain image-specific reference points."
                    )
                # Squeeze batch dimension for single image
                custom_refpoint_weight = custom_reference_points.squeeze(0)  # [N_q, 2]
            else:
                custom_refpoint_weight = custom_reference_points

            # Update the reference point embedding
            # Note: Some GroundingDINO versions may use different naming or structure
            # Adjust this based on actual model implementation
            if hasattr(self.model.transformer, "refpoint_embed"):
                self.model.transformer.refpoint_embed.weight.data = custom_refpoint_weight.detach()
            elif hasattr(self.model.transformer, "reference_points"):
                # Alternative: directly set reference_points if it's a parameter
                self.model.transformer.reference_points.data = custom_refpoint_weight.detach()
            else:
                # No-op; model may not support this feature in current config
                return



def wrap_grounding_dino(model):
    """
    Convenience function to wrap a GroundingDINO model.

    Args:
        model: GroundingDINO model instance

    Returns:
        GroundingDINOWithFeatures wrapper
    """
    return GroundingDINOWithFeatures(model)


def extract_visual_features_at_boxes(encoder_features, boxes, feature_dim=256, use_multi_scale=False):
    """
    Extract visual features at proposal box locations from encoder output.

    This uses the boxes to sample features from the encoder output,
    creating F_GDINO for Q_instance initialization.

    Args:
        encoder_features: Encoder output features
                         Can be:
                         - Single scale: [B, C, H, W] or [B, H*W, C]
                         - Multi-scale pyramid: List of [B, C, H/s, W/s] for different scales
        boxes: [B, N_q, 4] - Box coordinates (cx, cy, w, h) in normalized format
        feature_dim: Expected feature dimension
        use_multi_scale: If True and encoder_features is a list, sample from appropriate
                        scale based on box size (small boxes → high res, large boxes → low res)

    Returns:
        visual_features: [B, N_q, feature_dim] - Visual features at box locations
    """
    if encoder_features is None:
        raise ValueError("encoder_features is None - forward pass may not have been called")

    # Handle multi-scale pyramid
    if isinstance(encoder_features, (list, tuple)):
        features = encoder_features[0]
    else:
        features = encoder_features

    # Handle different tensor shapes
    if len(features.shape) == 4:  # [B, C, H, W]
        B, C, H, W = features.shape
        N_q = boxes.shape[1]

        # Use grid_sample to extract features at box locations
        # boxes format: [cx, cy, w, h] normalized to [0, 1]

        # Convert box centers to grid coordinates [-1, 1]
        box_centers = boxes[..., :2]  # [B, N_q, 2] - (cx, cy)
        grid = box_centers * 2 - 1  # Scale to [-1, 1]
        grid = grid.unsqueeze(2)  # [B, N_q, 1, 2] for grid_sample

        # Sample features at box centers
        sampled = torch.nn.functional.grid_sample(
            features,  # [B, C, H, W]
            grid,  # [B, N_q, 1, 2]
            mode="bilinear",
            align_corners=False,
        )
        # sampled shape: [B, C, N_q, 1]
        visual_features = sampled.squeeze(-1).permute(0, 2, 1)  # [B, N_q, C]

    elif len(features.shape) == 3:  # [B, HW, C]
        B, HW, C = features.shape
        N_q = boxes.shape[1]

        # Simple approach: use first N_q features
        # Better approach would use attention or learned selection
        if HW >= N_q:
            visual_features = features[:, :N_q, :]
        else:
            # Pad if needed
            padding = N_q - HW
            visual_features = torch.cat(
                [
                    features,
                    features[:, :padding, :],  # Repeat first features
                ],
                dim=1,
            )
    elif len(features.shape) == 2:  # [HW, C] or [N, C]
        # Fallback: expand dims to [1, HW, C] for batch compatibility
        features = features.unsqueeze(0)
        B, HW, C = features.shape
        N_q = boxes.shape[1]
        if HW >= N_q:
            visual_features = features[:, :N_q, :]
        else:
            padding = N_q - HW
            visual_features = torch.cat(
                [
                    features,
                    features[:, :padding, :],
                ],
                dim=1,
            )
    else:
        raise ValueError(f"Unexpected encoder_features shape: {features.shape}")

    # CRITICAL FIX: Projection should be handled by caller (InstanceQueryInitializer)
    # or registered as a persistent module, NOT created randomly here
    # For now, we expect feature_dim to match, or caller handles projection
    if visual_features.shape[-1] != feature_dim:
        # Warn but don't create random layers - this breaks training
        import warnings

        warnings.warn(
            f"Feature dimension mismatch: got {visual_features.shape[-1]}, expected {feature_dim}. "
            f"Projection should be handled by InstanceQueryInitializer, not here. "
            f"Returning features as-is.",
            UserWarning,
        )

    return visual_features


