"""DINOv2 feature extractor for Planktoscope images.

Kept at the repo root for backwards compatibility with deep_features_umap.ipynb,
which does `from feature_extractor import FeatureExtractor`. It now wraps the
shared planktoscope_ml backbone (DINOv2 ViT-S/14, 384-d) with pad-to-square
preprocessing, so notebook clusters live in the SAME feature space as the
LanceDB mining index and the trained classifiers.

The public API is unchanged from the old EfficientNet-B0 version:

    extractor = FeatureExtractor(device=device)
    features, valid_indices = extractor.extract(list_of_paths)

`features` is now (N, 384) instead of (N, 1280); UMAP/HDBSCAN are
dimension-agnostic, so the notebook works without changes.
"""

import numpy as np
import torch
from tqdm.auto import tqdm
from PIL import Image

from planktoscope_ml.embeddings import build_backbone, _auto_device
from planktoscope_ml.preprocessing import build_transform


class FeatureExtractor:
    """DINOv2 ViT-S/14 feature extractor.

    Parameters
    ----------
    device : torch.device | str | None
        Compute device. Auto-detected (MPS -> CUDA -> CPU) when None.
    batch_size : int
        Images processed per forward pass.
    backbone : str
        Backbone name (default 'dinov2_vits14'); 'efficientnet_b0' still works
        if you need the old 1280-d behavior.
    """

    EMBEDDING_DIM = 384  # DINOv2 ViT-S/14 (was 1280 for EfficientNet-B0)

    def __init__(self, device=None, batch_size: int = 64,
                 backbone: str = "dinov2_vits14"):
        self.device = torch.device(device or _auto_device())
        self.batch_size = batch_size
        model, dim = build_backbone(backbone)
        self.EMBEDDING_DIM = dim  # instance attr reflects the actual backbone
        self._model = model.eval().to(self.device)
        self._transform = build_transform(image_size=224, train=False)
        print(f"FeatureExtractor ready  |  backbone: {backbone}  |  "
              f"device: {self.device}  |  dim: {self.EMBEDDING_DIM}")

    def _load(self, path):
        try:
            return Image.open(path).convert("RGB")
        except Exception:
            return None

    @torch.no_grad()
    def extract(self, paths, desc: str = "Extracting features"):
        """Extract features for a list of image paths (str or Path).

        Returns
        -------
        features : np.ndarray, shape (n_valid, EMBEDDING_DIM), float32
        valid_indices : list[int]
            Indices into *paths* for images that loaded successfully.
        """
        paths = list(paths)
        embeddings, valid_indices = [], []
        for start in tqdm(range(0, len(paths), self.batch_size), desc=desc):
            batch_paths = paths[start:start + self.batch_size]
            batch_imgs, batch_idx = [], []
            for i, p in enumerate(batch_paths):
                img = self._load(p)
                if img is not None:
                    batch_imgs.append(self._transform(img))
                    batch_idx.append(start + i)
            if not batch_imgs:
                continue
            feats = self._model(torch.stack(batch_imgs).to(self.device))
            embeddings.append(feats.float().cpu().numpy())
            valid_indices.extend(batch_idx)

        if not embeddings:
            return np.empty((0, self.EMBEDDING_DIM), dtype=np.float32), []
        return np.vstack(embeddings).astype(np.float32), valid_indices
