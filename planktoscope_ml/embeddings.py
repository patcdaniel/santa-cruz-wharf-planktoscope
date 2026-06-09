"""Feature extraction for clustering, retrieval, and as a classifier backbone.

Two backbones are supported:

- ``dinov2_vits14`` (default): self-supervised DINOv2 ViT-S/14, 384-d. Its
  features separate plankton morphology far better than ImageNet-supervised
  features out of the box, which matters for both clustering and
  query-by-example mining. Requires ``timm``.
- ``efficientnet_b0``: ImageNet-supervised, 1280-d. No extra dependency;
  matches the original ``feature_extractor.py`` pipeline.
"""

import numpy as np
import torch
from tqdm.auto import tqdm
from PIL import Image

from .preprocessing import build_transform

EMBEDDING_DIMS = {
    "dinov2_vits14": 384,
    "efficientnet_b0": 1280,
}


def _auto_device():
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def build_backbone(backbone: str = "dinov2_vits14"):
    """Construct a feature-extracting backbone (no classification head).

    Returns ``(model, embedding_dim)``. Shared by FeatureExtractor (frozen,
    eval) and the classifier in train.py (trainable head on top).
    """
    if backbone not in EMBEDDING_DIMS:
        raise ValueError(f"Unknown backbone {backbone!r}; "
                         f"choose from {list(EMBEDDING_DIMS)}")
    if backbone == "dinov2_vits14":
        try:
            import timm
        except ImportError as e:
            raise ImportError(
                "DINOv2 backbone requires timm. Install with `pip install "
                "timm`, or use backbone='efficientnet_b0'."
            ) from e
        model = timm.create_model(
            "vit_small_patch14_dinov2.lvd142m",
            pretrained=True, num_classes=0, img_size=224)
    else:
        from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
        model = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        model.classifier = torch.nn.Identity()
    return model, EMBEDDING_DIMS[backbone]


class FeatureExtractor:
    """Extract fixed-length embeddings from Planktoscope ROIs.

    Parameters
    ----------
    backbone : str
        ``"dinov2_vits14"`` or ``"efficientnet_b0"``.
    device : str | None
        Auto-detected (MPS -> CUDA -> CPU) when None.
    batch_size : int
    image_size : int
    """

    def __init__(self, backbone: str = "dinov2_vits14", device=None,
                 batch_size: int = 64, image_size: int = 224):
        if backbone not in EMBEDDING_DIMS:
            raise ValueError(f"Unknown backbone {backbone!r}; "
                             f"choose from {list(EMBEDDING_DIMS)}")
        self.backbone = backbone
        self.embedding_dim = EMBEDDING_DIMS[backbone]
        self.batch_size = batch_size
        self.device = torch.device(device or _auto_device())
        self._transform = build_transform(image_size=image_size, train=False)
        self._model = self._build_model().eval().to(self.device)
        print(f"FeatureExtractor ready | backbone: {backbone} | "
              f"device: {self.device} | dim: {self.embedding_dim}")

    def _build_model(self):
        model, _ = build_backbone(self.backbone)
        return model

    @staticmethod
    def _load(path):
        try:
            return Image.open(path).convert("RGB")
        except Exception:
            return None

    @torch.no_grad()
    def extract(self, paths, desc: str = "Extracting features"):
        """Embed a list of image paths.

        Returns
        -------
        features : np.ndarray, shape (n_valid, embedding_dim), float32
        valid_indices : list[int]
            Indices into *paths* for images that loaded successfully (so the
            caller can realign metadata after dropping unreadable files).
        """
        embeddings, valid_indices = [], []
        for start in tqdm(range(0, len(paths), self.batch_size), desc=desc):
            batch_paths = paths[start:start + self.batch_size]
            imgs, idx = [], []
            for i, p in enumerate(batch_paths):
                img = self._load(p)
                if img is not None:
                    imgs.append(self._transform(img))
                    idx.append(start + i)
            if not imgs:
                continue
            feats = self._model(torch.stack(imgs).to(self.device))
            embeddings.append(feats.float().cpu().numpy())
            valid_indices.extend(idx)
        if not embeddings:
            return np.empty((0, self.embedding_dim), dtype=np.float32), []
        return np.vstack(embeddings).astype(np.float32), valid_indices
