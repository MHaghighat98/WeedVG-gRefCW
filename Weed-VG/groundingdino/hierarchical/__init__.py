# Hierarchical contrastive learning utilities

from .annotation import load_gref_annotations
from .batching import UniqueNegSentenceBatchSampler
from .grounding_dino_wrapper import GroundingDINOWithFeatures, wrap_grounding_dino, extract_visual_features_at_boxes
from .hierarchy import (
    HierarchicalContrastiveLoss,
    build_hierarchical_labels,
)
from .relevance_scorer import (
    ContrastiveRelevanceScorer,
    SineBoxPositionalEmbedding,
    sort_and_trim_proposals,
)

__all__ = [
    "load_gref_annotations",
    "UniqueNegSentenceBatchSampler",
    "GroundingDINOWithFeatures",
    "wrap_grounding_dino",
    "extract_visual_features_at_boxes",
    "HierarchicalContrastiveLoss",
    "build_hierarchical_labels",
    "ContrastiveRelevanceScorer",
    "SineBoxPositionalEmbedding",
    "sort_and_trim_proposals",
]
