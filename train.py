#!/usr/bin/env python3
"""
train.py — retrain the U-Net on all annotated experiments.

Scans DATA_ROOT for experiments that have both dataset/crops/ and
dataset/masks/ directories, combines them into one dataset, and trains
(or fine-tunes) the U-Net.  The trained model is saved to models/crystal_unet.pth
and will be picked up automatically by pipeline.py on the next run.

Usage:
    python train.py                          # use default DATA_ROOT
    python train.py --data ~/experiments     # custom data root
    python train.py --epochs 30 --lr 5e-4   # custom hyperparameters
    python train.py --scratch               # force train from scratch
"""
import argparse
import os
from pathlib import Path

from deep_learning import TORCH_AVAILABLE, CrystalDataset, UNet, UNetTrainer


def find_datasets(data_root: Path):
    """Return list of (crops_dir, masks_dir) pairs found under data_root."""
    pairs = []
    for exp in sorted(data_root.iterdir()):
        if not exp.is_dir():
            continue
        crops = exp / "dataset" / "crops"
        masks = exp / "dataset" / "masks"
        if crops.is_dir() and masks.is_dir() and any(crops.iterdir()):
            pairs.append((crops, masks))
    return pairs


def main():
    if not TORCH_AVAILABLE:
        raise SystemExit(
            "PyTorch is required for training.\n"
            "Install with: pip install torch torchvision"
        )

    _default = Path.home() / "OneDrive - KU Leuven" / "DATA" / "experiments"
    parser = argparse.ArgumentParser(description="Train U-Net on annotated crystal data.")
    parser.add_argument("--data",    default=os.environ.get("CRYSTAL_DATA_ROOT", str(_default)),
                        help="Root directory containing experiment folders")
    parser.add_argument("--epochs",  type=int,   default=50,   help="Training epochs")
    parser.add_argument("--lr",      type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--out",     default="models/crystal_unet.pth",
                        help="Output path for the trained model")
    parser.add_argument("--scratch", action="store_true",
                        help="Train from scratch even if a saved model exists")
    args = parser.parse_args()

    data_root = Path(args.data).expanduser()
    if not data_root.is_dir():
        raise SystemExit(f"Data root not found: {data_root}")

    pairs = find_datasets(data_root)
    if not pairs:
        raise SystemExit(
            f"No annotated datasets found under {data_root}\n"
            "Run 'Export Dataset' in the annotator first."
        )

    print(f"Found {len(pairs)} experiment(s) with annotations:")
    total = 0
    for crops, masks in pairs:
        n = len(list(crops.iterdir()))
        total += n
        print(f"  {crops.parent.parent.name}: {n} crops")
    print(f"Total samples: {total}\n")

    import torch
    from torch.utils.data import ConcatDataset, random_split

    datasets = [CrystalDataset(str(c), str(m)) for c, m in pairs]
    combined = ConcatDataset(datasets)

    val_n   = max(1, len(combined) // 5)
    train_n = len(combined) - val_n
    train_ds, val_ds = random_split(combined, [train_n, val_n])
    print(f"Split: {train_n} train / {val_n} val\n")

    model    = UNet(in_channels=1, num_classes=3)
    out_path = Path(args.out)

    if out_path.exists() and not args.scratch:
        model.load_state_dict(torch.load(str(out_path), map_location="cpu"))
        print(f"Fine-tuning from existing model: {out_path}")
    else:
        print("Training from scratch.")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    trainer = UNetTrainer(model, train_ds, val_ds)
    trainer.train(epochs=args.epochs, lr=args.lr)
    trainer.save_model(str(out_path))
    print(f"\nDone. Model saved to {out_path}")
    print("The pipeline will use this model automatically on the next run.")


if __name__ == "__main__":
    main()
