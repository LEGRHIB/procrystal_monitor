"""
Preprocessing Module
====================
Handles image loading, channel/ROI detection, and preprocessing
specific to polarized microscopy in microfluidic channels.
"""

import cv2
import numpy as np
from typing import Tuple, Optional, List
from config import PipelineConfig


class ChannelDetector:
    """
    Detects the microfluidic channel boundaries in polarized microscope images.

    The channel appears as a brighter vertical (or horizontal) tube against
    a dark background. This class finds the channel walls and extracts the
    droplet region of interest (ROI).
    """

    def __init__(self, config: PipelineConfig):
        self.config = config

    def detect_channel_roi(self, image: np.ndarray) -> Tuple[int, int, int, int]:
        """
        Detect the channel region in the image.

        Returns:
            (x, y, w, h) bounding box of the channel ROI.
        """
        gray = self._to_gray(image)

        if self.config.channel.vertical_channel:
            return self._detect_vertical_channel(gray)
        else:
            return self._detect_horizontal_channel(gray)

    def _to_gray(self, image: np.ndarray) -> np.ndarray:
        """Convert to grayscale if needed."""
        if len(image.shape) == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return image.copy()

    def _detect_vertical_channel(self, gray: np.ndarray) -> Tuple[int, int, int, int]:
        """
        Detect a vertical channel by finding two dominant vertical edges.

        Strategy: Project intensity horizontally (sum each column), find the
        bright plateau that represents the channel interior.
        """
        h, w = gray.shape

        # Horizontal intensity profile (sum along rows for each column)
        col_profile = np.mean(gray, axis=0).astype(np.float64)

        # Smooth the profile to reduce noise
        kernel_size = max(5, w // 50)
        if kernel_size % 2 == 0:
            kernel_size += 1
        smoothed = cv2.GaussianBlur(col_profile.reshape(1, -1),
                                     (kernel_size, 1), 0).flatten()

        # Find the channel as the widest bright region
        # Use Otsu-like thresholding on the profile
        threshold = np.mean(smoothed)
        above = smoothed > threshold

        # Find longest contiguous run of above-threshold columns
        best_start, best_len = 0, 0
        current_start, current_len = 0, 0
        for i in range(len(above)):
            if above[i]:
                if current_len == 0:
                    current_start = i
                current_len += 1
            else:
                if current_len > best_len:
                    best_start = current_start
                    best_len = current_len
                current_len = 0
        if current_len > best_len:
            best_start = current_start
            best_len = current_len

        margin = self.config.channel.wall_margin_px
        x = max(0, best_start + margin)
        w_roi = max(10, best_len - 2 * margin)
        y = 0
        h_roi = h

        return (x, y, w_roi, h_roi)

    def _detect_horizontal_channel(self, gray: np.ndarray) -> Tuple[int, int, int, int]:
        """Detect a horizontal channel (transpose logic of vertical)."""
        h, w = gray.shape
        row_profile = np.mean(gray, axis=1).astype(np.float64)

        kernel_size = max(5, h // 50)
        if kernel_size % 2 == 0:
            kernel_size += 1
        smoothed = cv2.GaussianBlur(row_profile.reshape(-1, 1),
                                     (1, kernel_size), 0).flatten()

        threshold = np.mean(smoothed)
        above = smoothed > threshold

        best_start, best_len = 0, 0
        current_start, current_len = 0, 0
        for i in range(len(above)):
            if above[i]:
                if current_len == 0:
                    current_start = i
                current_len += 1
            else:
                if current_len > best_len:
                    best_start = current_start
                    best_len = current_len
                current_len = 0
        if current_len > best_len:
            best_start = current_start
            best_len = current_len

        margin = self.config.channel.wall_margin_px
        x = 0
        w_roi = w
        y = max(0, best_start + margin)
        h_roi = max(10, best_len - 2 * margin)

        return (x, y, w_roi, h_roi)

    def extract_roi(self, image: np.ndarray,
                    roi: Optional[Tuple[int, int, int, int]] = None) -> np.ndarray:
        """Extract the channel ROI from the image."""
        if roi is None:
            roi = self.detect_channel_roi(image)
        x, y, w, h = roi
        return image[y:y+h, x:x+w].copy()


class ImagePreprocessor:
    """
    Preprocessing pipeline for polarized microscopy images.

    Steps:
    1. Noise reduction (Gaussian blur or bilateral filter)
    2. Contrast enhancement (CLAHE)
    3. Background normalization
    """

    def __init__(self, config: PipelineConfig):
        self.config = config

    def preprocess(self, image: np.ndarray,
                   denoise: bool = True,
                   enhance_contrast: bool = True) -> np.ndarray:
        """
        Full preprocessing pipeline.

        Args:
            image: Input image (BGR or grayscale).
            denoise: Apply denoising.
            enhance_contrast: Apply CLAHE contrast enhancement.

        Returns:
            Preprocessed image.
        """
        result = image.copy()

        if denoise:
            result = self.denoise(result)

        if enhance_contrast:
            result = self.enhance_contrast(result)

        return result

    def denoise(self, image: np.ndarray) -> np.ndarray:
        """Apply bilateral filter (edge-preserving denoising)."""
        if len(image.shape) == 3:
            return cv2.bilateralFilter(image, d=9, sigmaColor=75, sigmaSpace=75)
        else:
            return cv2.bilateralFilter(image, d=9, sigmaColor=75, sigmaSpace=75)

    def enhance_contrast(self, image: np.ndarray) -> np.ndarray:
        """Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)."""
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

        if len(image.shape) == 3:
            # Apply CLAHE to L channel in LAB color space
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            l = clahe.apply(l)
            lab = cv2.merge([l, a, b])
            return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        else:
            return clahe.apply(image)

    def compute_reference_model(self, reference_images: List[np.ndarray]) -> np.ndarray:
        """
        Build a reference model from multiple blank/clear droplet images.

        Uses median to robustly estimate the background.

        Args:
            reference_images: List of images of clear (crystal-free) droplets.

        Returns:
            Reference model image (same shape as input images).
        """
        if len(reference_images) == 1:
            return reference_images[0].copy()

        stack = np.stack(reference_images, axis=0)
        reference = np.median(stack, axis=0).astype(np.uint8)
        return reference

    def subtract_reference(self, image: np.ndarray,
                           reference: np.ndarray) -> np.ndarray:
        """
        Subtract reference (clear droplet) from current image.

        Returns absolute difference, highlighting any new features (crystals).
        """
        if len(image.shape) == 3 and len(reference.shape) == 3:
            diff = cv2.absdiff(image, reference)
            # Convert to grayscale difference magnitude
            gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        elif len(image.shape) == 2 and len(reference.shape) == 2:
            gray_diff = cv2.absdiff(image, reference)
        else:
            # Mixed: convert both to gray
            g1 = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
            g2 = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY) if len(reference.shape) == 3 else reference
            gray_diff = cv2.absdiff(g1, g2)

        return gray_diff
