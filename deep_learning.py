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
        img_path  = os.path.join(self.images_dir, f"{sample_id}.png")
        mask_path = os.path.join(self.masks_dir,  f"{sample_id}.png")

        cv2.imwrite(img_path,  image)
        cv2.imwrite(mask_path, mask)

        entry = {
            "sample_id":   sample_id,
            "image_path":  img_path,
            "mask_path":   mask_path,
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

        Input:  (B, C, H, W) where C=1 (grayscale) or C=3 (color)
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

            # Encoder
            self.encoders = nn.ModuleList()
            self.pools    = nn.ModuleList()
            prev_channels = in_channels
            for f in features:
                self.encoders.append(DoubleConv(prev_channels, f))
                self.pools.append(nn.MaxPool2d(2))
                prev_channels = f

            # Bottleneck
            self.bottleneck = DoubleConv(features[-1], features[-1] * 2)

            # Decoder
            self.upconvs  = nn.ModuleList()
            self.decoders = nn.ModuleList()
            rev_features  = features[::-1]
            prev_channels = features[-1] * 2
            for f in rev_features:
                self.upconvs.append(
                    nn.ConvTranspose2d(prev_channels, f, kernel_size=2, stride=2)
                )
                self.decoders.append(DoubleConv(f * 2, f))
                prev_channels = f

            self.final_conv = nn.Conv2d(features[0], num_classes, kernel_size=1)

        def forward(self, x):
            skip_connections = []
            for encoder, pool in zip(self.encoders, self.pools):
                x = encoder(x)
                skip_connections.append(x)
                x = pool(x)

            x = self.bottleneck(x)

            skip_connections = skip_connections[::-1]
            for upconv, decoder, skip in zip(self.upconvs, self.decoders,
                                             skip_connections):
                x = upconv(x)
                if x.shape != skip.shape:
                    x = F.interpolate(x, size=skip.shape[2:])
                x = torch.cat([skip, x], dim=1)
                x = decoder(x)

            return self.final_conv(x)

        def predict(self, image: np.ndarray,
                    train_size: int = 256) -> np.ndarray:
            """
            Run inference on a single image of any shape.

            The image is padded to square (aspect-ratio-preserving), resized
            to train_size × train_size, run through the network, then the
            prediction is cropped back to the original dimensions.

            Args:
                image: (H, W) grayscale or (H, W, C) colour numpy array.
                train_size: The square resolution used during training (default 256).

            Returns:
                (H, W) class prediction map in original image dimensions.
            """
            self.eval()
            orig_h, orig_w = image.shape[:2]

            with torch.no_grad():
                # ── 1. pad to square ─────────────────────────────────────────
                if len(image.shape) == 2:
                    padded = CrystalDataset._pad_to_square(image)
                    tensor = torch.FloatTensor(padded).unsqueeze(0).unsqueeze(0)
                else:
                    channels = [CrystalDataset._pad_to_square(image[:, :, c])
                                for c in range(image.shape[2])]
                    padded = np.stack(channels, axis=2)
                    tensor = torch.FloatTensor(
                        padded.transpose(2, 0, 1)).unsqueeze(0)

                pad_size = padded.shape[0]  # == padded.shape[1] (square)

                # ── 2. resize to training resolution ─────────────────────────
                tensor = tensor / 255.0
                tensor = F.interpolate(tensor,
                                       size=(train_size, train_size),
                                       mode='bilinear',
                                       align_corners=False)

                # ── 3. inference ─────────────────────────────────────────────
                logits = self.forward(tensor)

                # ── 4. resize prediction back to padded square ───────────────
                logits = F.interpolate(logits.float(),
                                       size=(pad_size, pad_size),
                                       mode='bilinear',
                                       align_corners=False)
                pred = torch.argmax(logits, dim=1).squeeze(0).numpy()

                # ── 5. crop padding back to original size ────────────────────
                pad_top  = (pad_size - orig_h) // 2
                pad_left = (pad_size - orig_w) // 2
                pred = pred[pad_top:pad_top + orig_h,
                            pad_left:pad_left + orig_w]

            return pred.astype(np.uint8)

    class CrystalDataset(Dataset):
        """
        PyTorch dataset for crystal segmentation training.

        Images are padded to square before resizing so that tall/narrow
        droplet crops are not distorted.
        """

        def __init__(self, images_dir: str, masks_dir: str,
                     image_size: Tuple[int, int] = (256, 256)):
            self.images_dir = images_dir
            self.masks_dir  = masks_dir
            self.image_size = image_size
            self.samples    = sorted([
                f for f in os.listdir(images_dir)
                if f.endswith(('.png', '.jpg', '.tif', '.tiff'))
            ])

        def __len__(self):
            return len(self.samples)

        @staticmethod
        def _pad_to_square(image: np.ndarray, fill: int = 0) -> np.ndarray:
            """
            Symmetrically pad a 2-D image with `fill` to make it square.
            Aspect ratio is preserved — no stretching occurs.
            """
            h, w = image.shape[:2]
            if h == w:
                return image
            size   = max(h, w)
            pad_h  = (size - h) // 2
            pad_w  = (size - w) // 2
            pad_h2 = size - h - pad_h
            pad_w2 = size - w - pad_w
            return np.pad(image,
                          ((pad_h, pad_h2), (pad_w, pad_w2)),
                          constant_values=fill)

        def __getitem__(self, idx):
            fname = self.samples[idx]
            img  = cv2.imread(os.path.join(self.images_dir, fname),
                              cv2.IMREAD_GRAYSCALE)
            mask = cv2.imread(os.path.join(self.masks_dir,  fname),
                              cv2.IMREAD_GRAYSCALE)

            # Pad to square first — preserves aspect ratio for tall/narrow crops
            img  = self._pad_to_square(img,  fill=0)
            mask = self._pad_to_square(mask, fill=0)

            # Resize to training resolution
            img  = cv2.resize(img,  self.image_size)
            mask = cv2.resize(mask, self.image_size,
                              interpolation=cv2.INTER_NEAREST)

            # Normalise image to [0, 1]
            img = img.astype(np.float32) / 255.0
            img = torch.FloatTensor(img).unsqueeze(0)   # (1, H, W)

            # Convert mask intensities → class indices
            # 0   = background
            # 128 = nucleation   → class 1
            # 255 = crystal body → class 2
            mask_tensor = torch.zeros(self.image_size, dtype=torch.long)
            mask_tensor[mask > 200]                  = 2  # crystal
            mask_tensor[(mask > 50) & (mask <= 200)] = 1  # nucleation

            return img, mask_tensor

    class UNetTrainer:
        """
        Training helper for the U-Net model.

        Example usage:
            model   = UNet(in_channels=1, num_classes=3)
            ds      = CrystalDataset("dataset/crops", "dataset/masks")
            trainer = UNetTrainer(model, ds)
            trainer.train(epochs=50)
            trainer.save_model("crystal_unet.pth")
        """

        def __init__(self, model: 'UNet',
                     train_dataset: 'CrystalDataset',
                     val_dataset: Optional['CrystalDataset'] = None,
                     device: str = 'cpu'):
            self.model  = model.to(device)
            self.device = device
            self.train_loader = DataLoader(train_dataset, batch_size=8,
                                           shuffle=True,  num_workers=0)
            self.val_loader   = None
            if val_dataset:
                self.val_loader = DataLoader(val_dataset, batch_size=8,
                                             shuffle=False, num_workers=0)
            self.history: Dict[str, List[float]] = {
                'train_loss': [], 'val_loss': [], 'val_iou': []
            }

        def train(self, epochs: int = 50, lr: float = 1e-3):
            """Train the model."""
            optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
            # Up-weight nucleation (rare) and crystal classes
            criterion = nn.CrossEntropyLoss(
                weight=torch.FloatTensor([1.0, 5.0, 2.0]).to(self.device)
            )

            for epoch in range(epochs):
                self.model.train()
                total_loss = 0
                for images, masks in self.train_loader:
                    images = images.to(self.device)
                    masks  = masks.to(self.device)

                    optimizer.zero_grad()
                    outputs = self.model(images)
                    loss    = criterion(outputs, masks)
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()

                avg_loss = total_loss / len(self.train_loader)
                self.history['train_loss'].append(avg_loss)

                if self.val_loader:
                    val_loss, val_iou = self._validate(criterion)
                    self.history['val_loss'].append(val_loss)
                    self.history['val_iou'].append(val_iou)
                    print(f"Epoch {epoch+1}/{epochs} | "
                          f"Train Loss: {avg_loss:.4f} | "
                          f"Val Loss: {val_loss:.4f} | "
                          f"Val IoU: {val_iou:.4f}")
                else:
                    print(f"Epoch {epoch+1}/{epochs} | "
                          f"Train Loss: {avg_loss:.4f}")

        def _validate(self, criterion) -> Tuple[float, float]:
            """Run validation and compute loss + IoU."""
            self.model.eval()
            total_loss = 0
            total_iou  = 0
            n_batches  = 0

            with torch.no_grad():
                for images, masks in self.val_loader:
                    images = images.to(self.device)
                    masks  = masks.to(self.device)

                    outputs = self.model(images)
                    loss    = criterion(outputs, masks)
                    total_loss += loss.item()

                    preds        = torch.argmax(outputs, dim=1)
                    crystal_pred = (preds == 2).float()
                    crystal_true = (masks == 2).float()
                    intersection = (crystal_pred * crystal_true).sum()
                    union        = (crystal_pred.sum() + crystal_true.sum()
                                   - intersection)
                    iou          = (intersection / (union + 1e-6)).item()
                    total_iou   += iou
                    n_batches   += 1

            return (total_loss / max(n_batches, 1),
                    total_iou  / max(n_batches, 1))

        def save_model(self, path: str):
            """Save model weights."""
            torch.save(self.model.state_dict(), path)
            print(f"Model saved to {path}")

        def load_model(self, path: str):
            """Load model weights."""
            self.model.load_state_dict(
                torch.load(path, map_location=self.device))
            print(f"Model loaded from {path}")

else:
    # Stub classes when PyTorch is not available
    class UNet:
        """Stub: install PyTorch to use deep learning features."""
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "PyTorch is required for deep learning features.\n"
                "Install with: pip install torch torchvision"
            )

    class CrystalDataset:
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "PyTorch required. Install: pip install torch torchvision")

    class UNetTrainer:
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "PyTorch required. Install: pip install torch torchvision")
