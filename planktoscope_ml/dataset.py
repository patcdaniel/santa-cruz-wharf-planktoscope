"""Training-set enumeration, scope selection, splits, and imbalance handling.

The labeled set under ``data/Training/<Class>/`` is extremely imbalanced
(~3,700:1 head-to-tail) and about half the classes have <20 images. This
module turns the folder tree into a DataFrame, lets you select a v1 class set
by a minimum-image floor, makes stratified splits, and produces the weights /
sampler needed to train despite the remaining imbalance.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, WeightedRandomSampler

IMG_EXTS = {".jpg", ".jpeg", ".png"}


def enumerate_training(training_dir="data/Training",
                       include_review: bool = False) -> pd.DataFrame:
    """Walk ``training_dir`` into a DataFrame of (path, label).

    Each top-level subdirectory is a class. Images under a ``review/`` subdir
    are staging (not yet verified) and excluded unless ``include_review``.
    """
    root = Path(training_dir)
    rows = []
    for class_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        for img in class_dir.rglob("*"):
            if img.suffix.lower() not in IMG_EXTS:
                continue
            if not include_review and "review" in img.relative_to(class_dir).parts:
                continue
            rows.append({"path": str(img), "label": class_dir.name})
    return pd.DataFrame(rows)


def count_table(df: pd.DataFrame) -> pd.DataFrame:
    """Per-class image counts, ascending."""
    return (df["label"].value_counts()
            .rename_axis("label").reset_index(name="n")
            .sort_values("n").reset_index(drop=True))


def apply_floor(df: pd.DataFrame, min_images: int = 75):
    """Keep only classes with at least ``min_images`` examples (v1 scope).

    Returns ``(kept_df, dropped)`` where ``dropped`` maps each excluded class
    to its count — these are the data-starved classes to grow by mining before
    a later training round.
    """
    counts = df["label"].value_counts()
    keep = counts[counts >= min_images].index
    dropped = counts[counts < min_images].to_dict()
    kept = df[df["label"].isin(keep)].reset_index(drop=True)
    return kept, dict(sorted(dropped.items(), key=lambda kv: kv[1]))


def build_class_index(df: pd.DataFrame) -> dict:
    return {name: i for i, name in enumerate(sorted(df["label"].unique()))}


def make_splits(df: pd.DataFrame, val_frac: float = 0.15,
                test_frac: float = 0.15, seed: int = 42) -> pd.DataFrame:
    """Add a stratified ``split`` column ('train' | 'val' | 'test')."""
    idx = np.arange(len(df))
    trainval_i, test_i = train_test_split(
        idx, test_size=test_frac, stratify=df["label"], random_state=seed)
    rel_val = val_frac / (1.0 - test_frac)
    train_i, val_i = train_test_split(
        trainval_i, test_size=rel_val,
        stratify=df["label"].iloc[trainval_i], random_state=seed)
    out = df.copy()
    out["split"] = "train"
    out.loc[out.index[val_i], "split"] = "val"
    out.loc[out.index[test_i], "split"] = "test"
    return out


def effective_num_weights(counts: np.ndarray, beta: float = 0.999) -> np.ndarray:
    """Class-balanced loss weights (Cui et al. 2019, "effective number of
    samples"). ``beta`` near 1 weights rare classes more aggressively.
    Returns weights normalized to mean 1, aligned to ``counts`` order.
    """
    counts = np.asarray(counts, dtype=np.float64)
    eff = (1.0 - np.power(beta, counts)) / (1.0 - beta)
    w = 1.0 / eff
    return w / w.mean() * 1.0


def make_sampler(labels_idx: np.ndarray) -> WeightedRandomSampler:
    """WeightedRandomSampler that draws classes ~uniformly (inverse-frequency
    per-sample weights). Use this OR loss weighting; using both double-counts.
    """
    counts = np.bincount(labels_idx)
    per_class_w = 1.0 / np.maximum(counts, 1)
    sample_w = per_class_w[labels_idx]
    return WeightedRandomSampler(
        weights=torch.as_tensor(sample_w, dtype=torch.double),
        num_samples=len(labels_idx), replacement=True)


class PlanktonDataset(Dataset):
    """Dataset over a (path, label, split) DataFrame for one split."""

    def __init__(self, df: pd.DataFrame, class_to_idx: dict, transform,
                 split: str | None = None):
        if split is not None:
            df = df[df["split"] == split]
        self.paths = df["path"].tolist()
        self.targets = np.array([class_to_idx[l] for l in df["label"]],
                                dtype=np.int64)
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = Image.open(self.paths[i]).convert("RGB")
        return self.transform(img), int(self.targets[i])
