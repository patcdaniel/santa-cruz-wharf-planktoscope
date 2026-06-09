"""Train the Planktoscope multiclass classifier.

Design choices driven by the data:
- A minimum-image floor selects the v1 class set; data-starved classes are
  reported and deferred until mining grows them (see scripts/mine.py).
- Imbalance among the kept classes is handled by class-balanced loss weighting
  (default) OR a class-balanced sampler -- using both double-counts, so they
  are mutually exclusive here.
- The backbone defaults to frozen (linear probe), which is robust with limited
  data; --finetune unfreezes it.
- Evaluation reports PER-CLASS accuracy and balanced accuracy, not just overall
  accuracy (which the head classes would dominate), plus a selective-prediction
  (abstention) table for deployment over raw ROIs.

Example
-------
    python -m planktoscope_ml.train --min-images 75 --epochs 20
    python -m planktoscope_ml.train --backbone efficientnet_b0 --finetune
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless: save figures, never open a window
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (balanced_accuracy_score, confusion_matrix,
                             precision_recall_fscore_support)
from torch.utils.data import DataLoader

from .embeddings import build_backbone, _auto_device
from .preprocessing import build_transform
from .dataset import (enumerate_training, apply_floor, build_class_index,
                      make_splits, effective_num_weights, make_sampler,
                      PlanktonDataset)


class PlanktonClassifier(nn.Module):
    def __init__(self, backbone: str, n_classes: int, freeze: bool = True):
        super().__init__()
        self.backbone, dim = build_backbone(backbone)
        self.frozen = freeze
        if freeze:
            for p in self.backbone.parameters():
                p.requires_grad = False
        self.head = nn.Linear(dim, n_classes)

    def forward(self, x):
        if self.frozen:
            with torch.no_grad():
                feats = self.backbone(x)
        else:
            feats = self.backbone(x)
        return self.head(feats)


def _run_epoch(model, loader, device, criterion, optimizer=None):
    train = optimizer is not None
    model.head.train(train)
    if not model.frozen:
        model.backbone.train(train)
    total, total_loss = 0, 0.0
    preds, gts = [], []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        with torch.set_grad_enabled(train):
            logits = model(x)
            loss = criterion(logits, y)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        total_loss += loss.item() * len(y)
        total += len(y)
        preds.append(logits.argmax(1).cpu())
        gts.append(y.cpu())
    preds = torch.cat(preds).numpy()
    gts = torch.cat(gts).numpy()
    return total_loss / total, balanced_accuracy_score(gts, preds)


def save_confusion_heatmap(cm, classes, png_path, title):
    """Row-normalized (recall) confusion-matrix heatmap."""
    cmn = cm / np.clip(cm.sum(1, keepdims=True), 1, None)
    n = len(classes)
    fig, ax = plt.subplots(figsize=(max(7, n * 0.55), max(6, n * 0.5)))
    im = ax.imshow(cmn, cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(classes, rotation=90, fontsize=7)
    ax.set_yticklabels(classes, fontsize=7)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(title)
    for i in range(n):
        for j in range(n):
            v = cmn[i, j]
            if v >= 0.01:
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6,
                        color="white" if v < 0.6 else "black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="fraction of true class")
    fig.tight_layout()
    Path(png_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path, dpi=150)
    plt.close(fig)


@torch.no_grad()
def evaluate(model, loader, device, classes, csv_path=None, png_path=None,
             title="Confusion matrix"):
    model.eval()
    probs, gts = [], []
    for x, y in loader:
        logits = model(x.to(device))
        probs.append(torch.softmax(logits, 1).cpu())
        gts.append(y)
    probs = torch.cat(probs).numpy()
    gts = torch.cat(gts).numpy()
    preds = probs.argmax(1)

    print(f"\n  Overall accuracy : {(preds == gts).mean():.3f}")
    print(f"  Balanced accuracy: {balanced_accuracy_score(gts, preds):.3f}")
    macro_f1 = precision_recall_fscore_support(
        gts, preds, labels=range(len(classes)), average="macro",
        zero_division=0)[2]
    print(f"  Macro F1         : {macro_f1:.3f}")

    cm = confusion_matrix(gts, preds, labels=range(len(classes)))
    prec, rec, f1, sup = precision_recall_fscore_support(
        gts, preds, labels=range(len(classes)), zero_division=0)
    table = pd.DataFrame({
        "class": classes, "support": sup,
        "precision": prec.round(3), "recall": rec.round(3), "f1": f1.round(3),
    }).sort_values("f1", ascending=False).reset_index(drop=True)

    print("\n  Per-class precision / recall / F1:")
    print(table.to_string(index=False))
    if csv_path:
        Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(csv_path, index=False)
        print(f"  metrics table -> {csv_path}")
    if png_path:
        save_confusion_heatmap(cm, classes, png_path, title)
        print(f"  confusion heatmap -> {png_path}")

    print("\n  Selective prediction (abstain below confidence threshold):")
    conf = probs.max(1)
    for tau in (0.5, 0.7, 0.9):
        keep = conf >= tau
        cov = keep.mean()
        sel_acc = (preds[keep] == gts[keep]).mean() if keep.any() else float("nan")
        print(f"    tau={tau:.1f}  coverage={cov:.3f}  accuracy={sel_acc:.3f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--training-dir", default="data/Training")
    ap.add_argument("--min-images", type=int, default=75,
                    help="Floor for v1 class inclusion.")
    ap.add_argument("--backbone", default="dinov2_vits14",
                    choices=["dinov2_vits14", "efficientnet_b0"])
    ap.add_argument("--finetune", action="store_true",
                    help="Unfreeze the backbone (default: frozen linear probe).")
    ap.add_argument("--imbalance", default="loss",
                    choices=["loss", "sampler", "none"])
    ap.add_argument("--cb-beta", type=float, default=0.999,
                    help="Class-balanced loss beta (lower = gentler reweighting).")
    ap.add_argument("--no-augment", action="store_true",
                    help="Disable train-time flips/rotation. Recommended for a "
                         "frozen backbone: augmentation can't adapt features and "
                         "just adds noise to the linear head, hurting tiny classes.")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--image-size", type=int, default=224)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default=None)
    ap.add_argument("--save-path", default="artifacts/classifier.pth")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device or _auto_device())

    df = enumerate_training(args.training_dir)
    df, dropped = apply_floor(df, args.min_images)
    class_to_idx = build_class_index(df)
    classes = sorted(class_to_idx, key=class_to_idx.get)
    df = make_splits(df, seed=args.seed)
    val_paths = df.loc[df.split == "val", "path"].tolist()
    test_paths = df.loc[df.split == "test", "path"].tolist()

    print(f"v1 classes ({len(classes)}, floor={args.min_images}): "
          f"{len(df):,} images")
    if dropped:
        print(f"Deferred (below floor, mine these): {dropped}")

    tf_train = build_transform(args.image_size, train=not args.no_augment)
    tf_eval = build_transform(args.image_size, train=False)
    ds_train = PlanktonDataset(df, class_to_idx, tf_train, split="train")
    ds_val = PlanktonDataset(df, class_to_idx, tf_eval, split="val")
    ds_test = PlanktonDataset(df, class_to_idx, tf_eval, split="test")

    if args.imbalance == "sampler":
        sampler = make_sampler(ds_train.targets)
        train_loader = DataLoader(ds_train, batch_size=args.batch_size,
                                  sampler=sampler, num_workers=args.num_workers)
        weight = None
    else:
        train_loader = DataLoader(ds_train, batch_size=args.batch_size,
                                  shuffle=True, num_workers=args.num_workers)
        if args.imbalance == "loss":
            counts = np.bincount(ds_train.targets, minlength=len(classes))
            w = effective_num_weights(counts, beta=args.cb_beta)
            weight = torch.tensor(w, dtype=torch.float32, device=device)
        else:
            weight = None

    val_loader = DataLoader(ds_val, batch_size=args.batch_size,
                            num_workers=args.num_workers)
    test_loader = DataLoader(ds_test, batch_size=args.batch_size,
                             num_workers=args.num_workers)

    model = PlanktonClassifier(args.backbone, len(classes),
                               freeze=not args.finetune).to(device)
    criterion = nn.CrossEntropyLoss(weight=weight)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=args.lr)

    save_path = Path(args.save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    best_val = -1.0
    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_bacc = _run_epoch(model, train_loader, device, criterion,
                                      optimizer)
        va_loss, va_bacc = _run_epoch(model, val_loader, device, criterion)
        flag = ""
        if va_bacc > best_val:
            best_val = va_bacc
            torch.save({"state_dict": model.state_dict(),
                        "class_to_idx": class_to_idx,
                        "backbone": args.backbone,
                        "image_size": args.image_size,
                        "frozen": not args.finetune,
                        "min_images": args.min_images,
                        "seed": args.seed,
                        "val_paths": val_paths,
                        "test_paths": test_paths}, save_path)
            flag = "  *saved"
        print(f"epoch {epoch:3d} | train loss {tr_loss:.3f} bacc {tr_bacc:.3f} "
              f"| val loss {va_loss:.3f} bacc {va_bacc:.3f}{flag}")

    print(f"\nBest val balanced acc: {best_val:.3f}. Evaluating best on test:")
    ckpt = torch.load(save_path, map_location=device)
    model.load_state_dict(ckpt["state_dict"])
    stem = save_path.stem
    evaluate(model, test_loader, device, classes,
             csv_path=save_path.with_name(f"{stem}_metrics.csv"),
             png_path=Path("figures") / f"{stem}_confusion.png",
             title=f"{stem} confusion (test, recall-normalized)")
    print(f"\nSaved -> {save_path}")


if __name__ == "__main__":
    main()
