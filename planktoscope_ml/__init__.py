"""Planktoscope multiclass classifier + embedding-based data mining toolkit.

Modules
-------
preprocessing : pad-to-square transforms that preserve ROI aspect ratio.
embeddings    : CNN/ViT feature extractor (DINOv2 or EfficientNet-B0).
clustering    : stateful UMAP + HDBSCAN hierarchical clustering.
dataset       : class enumeration, floor filtering, splits, imbalance handling.
index         : LanceDB vector store + query-by-example retrieval.
"""

from .preprocessing import PadToSquare, build_transform
from .embeddings import FeatureExtractor
from .clustering import ClusteringSession

__all__ = ["PadToSquare", "build_transform", "FeatureExtractor", "ClusteringSession"]
