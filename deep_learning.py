"""
Deep Learning Module — U-Net for Crystal Segmentation
=====================================================
This module provides a U-Net architecture for semantic segmentation
of crystals in polarized microscopy images.

REQUIRES: pip install torch torchvision
(Not available in this sandbox but ready for your local machine)

Usage workflow:
1. Use the classical pipeline to generate initial detections
2. Manually correct/refine detections to build a training dataset
3. Train the U-Net on your labeled data
4. Switch the pipeline to use U-Net predictions instead of classical CV

This gives you the best of both worlds:
- Start immediately with classical CV (no training needed)
- Gradually build training data as you use the tool
- Switch to deep learning once you have enough labeled images
"""

import numpy as np
import cv2
import os
import json
from typing import List, Tuple, Optional, Dict


# ============================================================
# LABELING TOOL — Generate training data from classical detections
# ============================================================

class LabelGenerator:
    """
    Generates labeled masks from classical CV detections.
    These can be manually reviewed, then used to train the U-Net.
    """

    def __init__(self, output_dir: str = "./training_data"):
        self.output_dir = output_dir
        self.images_dir = os.path.join(output_dir, "images")
        self.masks_dir = os.path.join(output_dir, "masks")
        os.makedirs(self.images_dir, exist_ok=True)
        os.makedirs(self.masks_dir, exist_ok=True)
        self.metadata: List[Dict] = []

    def save_training_pair(self, image: np.ndarray, mask: np.ndarray,
                           sample_id: str, metadata: Optional[Dict] = None):
        """
        Save an image-mask pair for training.

        Args:
            image: Input image (preprocessed ROI).
            mask: Binary mask where 255 = crystal, 0 = background.
            sample_id: Unique identifier for this sample.
            metadata: Optional metadata dict.
        """
        img_path = os.path.join(self.images_dir, f"{sample_id}.png")
        mask_path = os.path.join(self.masks_dir, f"{sample_id}.png")

        cv2.imwrite(img_path, image)
        cv2.imwrite(mask_path, mask)

        entry = {
            "sample_id": sample_id,
            "image_path": img_path,
            "mask_path": mask_path,
            "image_shape": list(image.shape),
        }
        if metadata:
            entry.update(metadata)
        self.metadata.append(entry)

    def save_metadata(self):
        """Save metadata JSON for all training pairs."""
        path = os.path.join(self.output_dir, "metadata.json")
        with open(path, 'w') as f:
            json.dump(self.metadata, f, indent=2)


# ============================================================
# U-NET ARCHITECTURE (requires torch)
# ============================================================

TORCH_AVAILABLE = False
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    pass


if TORCH_AVAILABLE:

    class DoubleConv(nn.Module):
        """Two convolution layers with BatchNorm and ReLU."""
        def __init__(self, in_channels, out_channels):
            super().__init__()
            self.double_conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, 3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )

        def forward(self, x):
            return self.double_conv(x)

    class UNet(nn.Module):
        """
        U-Net for crystal segmentation.

        Input: (B, C, H, W) where C=1 (grayscale) or C=3 (color)
        Output: (B, num_classes, H, W) segmentation logits

        Classes:
        0 = background
        1 = nucleation
        2 = crystal
        """

        def __init__(self, in_channels: int = 1, num_classes: int = 3,
                     features: List[int] = None):
            super().__init__()
            if features is None:
                features = [32, 64, 128, 256]

            # Encoder (downsampling path)
            self.encoders = nn.ModuleList()
            self.pools = nn.ModuleList()
            prev_channels = in_channels
            for f in features:
                self.encoders.append(DoubleConv(prev_channels, f))
                self.pools.append(nn.MaxPool2d(2))
                prev_channels = f

            # Bottleneck
            self.bottleneck = DoubleConv(features[-1], features[-1] * 2)

            # Decoder (upsampling path)
            self.upconvs = nn.ModuleList()
            self.decoders = nn.ModuleList()
            rev_features = features[::-1]
            prev_channels = features[-1] * 2
            for f in rev_features:
                self.upconvs.append(
                    nn.ConvTranspose2d(prev_channels, f, kernel_size=2, stride=2)
                )
                self.decoders.append(DoubleConv(f * 2, f))
                prev_channels = f

            # Final 1x1 conv to get class predictions
            self.final_conv = nn.Conv2d(features[0], num_classes, kernel_size=1)

        def forward(self, x):
            # Encoder
            skip_connections = []
            for encoder, pool in zip(self.encoders, self.pools):
                x = encoder(x)
                skip_connections.append(x)
                x = pool(x)

            # Bottleneck
            x = self.bottleneck(x)

            # Decoder
            skip_connections = skip_connections[::-1]
            for i, (upconv, decoder) in enumerate(zip(self.upconvs, self.decoders)):
                x = upconv(x)
                skip = skip_connections[i]
                # Handle size mismatch
                if x.shape != skip.shape:
                    x = F.interpolate(x, size=skip.shape[2:])
                x = torch.cat([skip, x], dim=1)
                x = decoder(x)

            return self.final_conv(x)

        def predict(self, image: np.ndarray) -> np.ndarray:
            """
            Run inference on a single image.

            Args:
                image: (H, W) or (H, W, C) numpy array.

            Returns:
                (H, W) class prediction map.
            """
            self.eval()
            with torch.no_grad():
                if len(image.shape) == 2:
                    tensor = torch.FloatTensor(image).unsqueeze(0).unsqueeze(0)
                else:
                    tensor = torch.FloatTensor(image.transpose(2, 0, 1)).unsqueeze(0)

                # Normalize to [0, 1]
                tensor = tensor / 255.0

                logits = self.forward(tensor)
                pred = torch.argmax(logits, dim=1).squeeze(0).numpy()

            return pred.astype(np.uint8)

    class CrystalDataset(Dataset):
        """PyTorch dataset for crystal segmentation training."""

        def __init__(self, images_dir: str, masks_dir: str,
                     image_size: Tuple[int, int] = (256, 256)):
            self.images_dir = images_dir
            self.masks_dir = masks_dir
            self.image_size = image_size
            self.samples = sorted([
                f for f in os.listdir(images_dir)
                if f.endswith(('.png', '.jpg', '.tif', '.tiff'))
            ])

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, idx):
            fname = self.samples[idx]
            img = cv2.imread(os.path.join(self.images_dir, fname), cv2.IMREAD_GRAYSCALE)
            mask = cv2.imread(os.path.join(self.masks_dir, fname), cv2.IMREAD_GRAYSCALE)

            # Resize
            img = cv2.resize(img, self.image_size)
            mask = cv2.resize(mask, self.image_size, interpolation=cv2.INTER_NEAREST)

            # Normalize image
            img = img.astype(np.float32) / 255.0
            img = torch.FloatTensor(img).unsqueeze(0)  # (1, H, W)

            # Convert mask: 0=bg, 1=nucleation (values ~128), 2=crystal (values ~255)
            mask_tensor = torch.zeros(self.image_size, dtype=torch.long)
            mask_tensor[mask > 200] = 2   # crystal
            mask_tensor[(mask > 50) & (mask <= 200)] = 1  # nucleation

            return img, mask_tensor

    class UNetTrainer:
        """
        Training helper for the U-Net model.

        Example usage:
            trainer = UNetTrainer(model, train_dataset, val_dataset)
            trainer.train(epochs=50, lr=1e-3)
            trainer.save_model("crystal_unet.pth")
        """

        def __init__(self, model: 'UNet',
                     train_dataset: CrystalDataset,
                     val_dataset: Optional[CrystalDataset] = None,
                     device: str = 'cpu'):
            self.model = model.to(device)
            self.device = device
            self.train_loader = DataLoader(train_dataset, batch_size=8,
                                           shuffle=True, num_workers=0)
            self.val_loader = None
            if val_dataset:
                self.val_loader = DataLoader(val_dataset, batch_size=8,
                                              shuffle=False, num_workers=0)
            self.history: Dict[str, List[float]] = {
                'train_loss': [], 'val_loss': [], 'val_iou': []
            }

        def train(self, epochs: int = 50, lr: float = 1e-3):
            """Train the model."""
            optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
            # Use weighted loss since nucleation events are rare
            criterion = nn.CrossEntropyLoss(
                weight=torch.FloatTensor([1.0, 5.0, 2.0]).to(self.device)
            )

            for epoch in range(epochs):
                self.model.train()
                total_loss = 0
                for images, masks in self.train_loader:
                    images = images.to(self.device)
                    masks = masks.to(self.device)

                    optimizer.zero_grad()
                    outputs = self.model(images)
                    loss = criterion(outputs, masks)
                    loss.backward()
                    optimizer.step()

                    total_loss += loss.item()

                avg_loss = total_loss / len(self.train_loader)
                self.history['train_loss'].append(avg_loss)

                # Validation
                if self.val_loader:
                    val_loss, val_iou = self._validate(criterion)
                    self.history['val_loss'].append(val_loss)
                    self.history['val_iou'].append(val_iou)
                    print(f"Epoch {epoch+1}/{epochs} | "
                          f"Train Loss: {avg_loss:.4f} | "
                          f"Val Loss: {val_loss:.4f} | "
                          f"Val IoU: {val_iou:.4f}")
                else:
                    print(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_loss:.4f}")

        def _validate(self, criterion) -> Tuple[float, float]:
            """Run validation and compute loss + IoU."""
            self.model.eval()
            total_loss = 0
            total_iou = 0
            n_batches = 0

            with torch.no_grad():
                for images, masks in self.val_loader:
                    images = images.to(self.device)
                    masks = masks.to(self.device)

                    outputs = self.model(images)
                    loss = criterion(outputs, masks)
                    total_loss += loss.item()

                    # IoU for crystal class
                    preds = torch.argmax(outputs, dim=1)
                    crystal_pred = (preds == 2).float()
                    crystal_true = (masks == 2).float()
                    intersection = (crystal_pred * crystal_true).sum()
                    union = crystal_pred.sum() + crystal_true.sum() - intersection
                    iou = (intersection / (union + 1e-6)).item()
                    total_iou += iou
                    n_batches += 1

            return total_loss / max(n_batches, 1), total_iou / max(n_batches, 1)

        def save_model(self, path: str):
            """Save model weights."""
            torch.save(self.model.state_dict(), path)
            print(f"Model saved to {path}")

        def load_model(self, path: str):
            """Load model weights."""
            self.model.load_state_dict(torch.load(path, map_location=self.device))
            print(f"Model loaded from {path}")

else:
    # Stub classes when torch is not available
    class UNet:
        """Stub: Install PyTorch to use deep learning features."""
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "PyTorch is required for deep learning features.\n"
                "Install with: pip install torch torchvision\n"
                "Then you can train a U-Net on your labeled crystal data."
            )

    class CrystalDataset:
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch required. Install: pip install torch torchvision")

    class UNetTrainer:
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch required. Install: pip install torch torchvision")
