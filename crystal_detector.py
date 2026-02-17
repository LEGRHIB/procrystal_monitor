"""
Crystal Detector Module
=======================
Detects nucleation events and crystals in polarized microscopy images.
Uses both absolute brightness (birefringence) and reference subtraction.
"""

import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from config import PipelineConfig


@dataclass
class CrystalDetection:
    """Represents a single detected crystal or nucleation event."""
    # Unique ID for this detection in the current frame
    detection_id: int
    # Bounding box (x, y, w, h) in ROI coordinates
    bbox: Tuple[int, int, int, int]
    # Centroid (cx, cy) in ROI coordinates
    centroid: Tuple[float, float]
    # Area in pixels
    area_px: float
    # Contour of the crystal
    contour: np.ndarray
    # Mean intensity within the crystal region
    mean_intensity: float
    # Max intensity within the crystal region
    max_intensity: float
    # Whether this is classified as nucleation (small, early) vs grown crystal
    is_nucleation: bool
    # Circularity: 4*pi*area / perimeter^2
    circularity: float
    # Aspect ratio of the fitted ellipse
    aspect_ratio: float
    # Orientation angle of the fitted ellipse (degrees)
    orientation_deg: float
    # Equivalent diameter = sqrt(4*area/pi)
    equivalent_diameter_px: float
    # Solidity = area / convex_hull_area
    solidity: float
    # Perimeter
    perimeter_px: float


class CrystalDetector:
    """
    Detects crystals and nucleation events in polarized microscopy images.

    Two detection strategies are combined:
    1. Brightness-based: Under crossed polarizers, crystals appear bright
       due to birefringence. A threshold on brightness detects them.
    2. Reference subtraction: Subtract a clear/blank droplet image to
       highlight any new features that appeared (crystals).

    The union of both methods gives the best coverage.
    """

    def __init__(self, config: PipelineConfig):
        self.config = config
        self._detection_counter = 0

    def detect(self, image: np.ndarray,
               reference_diff: Optional[np.ndarray] = None) -> List[CrystalDetection]:
        """
        Run full crystal detection on a preprocessed ROI image.

        Args:
            image: Preprocessed ROI image (BGR or grayscale).
            reference_diff: Grayscale difference image from reference subtraction
                           (output of ImagePreprocessor.subtract_reference).

        Returns:
            List of CrystalDetection objects found in this frame.
        """
        gray = self._to_gray(image)
        det_cfg = self.config.detection

        # Strategy 1: Brightness-based detection
        mask_bright = self._detect_by_brightness(gray)

        # Strategy 2: Reference subtraction
        mask_diff = np.zeros_like(gray)
        if reference_diff is not None and det_cfg.use_reference_subtraction:
            mask_diff = self._detect_by_reference_diff(reference_diff)

        # Combine masks (union)
        combined_mask = cv2.bitwise_or(mask_bright, mask_diff)

        # Morphological cleanup
        combined_mask = self._morphological_cleanup(combined_mask)

        # Extract contours and build detections
        detections = self._extract_detections(combined_mask, gray, image)

        return detections

    def detect_nucleation_only(self, image: np.ndarray,
                                reference_diff: Optional[np.ndarray] = None
                                ) -> List[CrystalDetection]:
        """
        Detect only nucleation events (small, early-stage crystal spots).

        Same as detect() but filtered for small sizes consistent with nucleation.
        """
        all_detections = self.detect(image, reference_diff)
        return [d for d in all_detections if d.is_nucleation]

    def _to_gray(self, image: np.ndarray) -> np.ndarray:
        if len(image.shape) == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return image.copy()

    def _detect_by_brightness(self, gray: np.ndarray) -> np.ndarray:
        """
        Detect bright regions (birefringent crystals under crossed polarizers).

        Uses a combination of:
        - Global threshold (for very bright crystals)
        - Adaptive threshold (for crystals with varying local background)
        """
        det_cfg = self.config.detection

        # Global threshold
        _, mask_global = cv2.threshold(
            gray, det_cfg.brightness_threshold, 255, cv2.THRESH_BINARY
        )

        # Adaptive threshold (catches crystals in non-uniform illumination)
        block_size = det_cfg.adaptive_block_size
        if block_size % 2 == 0:
            block_size += 1
        mask_adaptive = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, block_size, det_cfg.adaptive_constant
        )

        # Intersection: both methods must agree to reduce false positives
        # For very bright crystals, global alone is sufficient
        mask_combined = cv2.bitwise_or(
            mask_global,
            cv2.bitwise_and(mask_global, mask_adaptive)
        )

        return mask_combined

    def _detect_by_reference_diff(self, diff_image: np.ndarray) -> np.ndarray:
        """
        Detect crystals from the difference image (vs. blank reference).

        Any region with significant difference from the clear droplet
        is likely a crystal or nucleation event.
        """
        det_cfg = self.config.detection
        _, mask = cv2.threshold(
            diff_image, det_cfg.difference_threshold, 255, cv2.THRESH_BINARY
        )
        return mask

    def _morphological_cleanup(self, mask: np.ndarray) -> np.ndarray:
        """Apply morphological operations to clean up the detection mask."""
        det_cfg = self.config.detection

        # Opening: remove small noise
        k_open = det_cfg.morph_open_kernel
        if k_open > 0:
            kernel_open = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (k_open, k_open)
            )
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open)

        # Closing: fill small holes inside crystals
        k_close = det_cfg.morph_close_kernel
        if k_close > 0:
            kernel_close = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (k_close, k_close)
            )
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)

        return mask

    def _extract_detections(self, mask: np.ndarray, gray: np.ndarray,
                            color_image: np.ndarray) -> List[CrystalDetection]:
        """
        Extract contours from mask and build CrystalDetection objects.

        Filters by area, circularity, and classifies as nucleation vs crystal.
        """
        det_cfg = self.config.detection

        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        detections = []
        for contour in contours:
            area = cv2.contourArea(contour)

            # Filter by area
            if area < det_cfg.min_crystal_area_px:
                # Check if it's a nucleation event (smaller threshold)
                if area < det_cfg.nucleation_min_area_px:
                    continue
            if area > det_cfg.max_crystal_area_px:
                continue

            # Compute shape features
            perimeter = cv2.arcLength(contour, True)
            circularity = (4 * np.pi * area / (perimeter * perimeter + 1e-6))

            # Filter by circularity if configured
            if circularity < det_cfg.min_circularity:
                continue

            # Bounding box
            x, y, w, h = cv2.boundingRect(contour)

            # Centroid from moments
            M = cv2.moments(contour)
            if M["m00"] > 0:
                cx = M["m10"] / M["m00"]
                cy = M["m01"] / M["m00"]
            else:
                cx, cy = x + w / 2, y + h / 2

            # Fit ellipse for orientation and aspect ratio
            if len(contour) >= 5:
                ellipse = cv2.fitEllipse(contour)
                (_, (minor_axis, major_axis), angle) = ellipse
                aspect_ratio = max(major_axis, 1) / max(minor_axis, 1)
                orientation = angle
            else:
                aspect_ratio = 1.0
                orientation = 0.0

            # Intensity statistics within the contour
            contour_mask = np.zeros(gray.shape, dtype=np.uint8)
            cv2.drawContours(contour_mask, [contour], -1, 255, -1)
            mean_val = cv2.mean(gray, mask=contour_mask)[0]
            _, max_val, _, _ = cv2.minMaxLoc(gray, mask=contour_mask)

            # Solidity
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            solidity = area / (hull_area + 1e-6)

            # Equivalent diameter
            eq_diameter = np.sqrt(4 * area / np.pi)

            # Classify as nucleation vs grown crystal
            is_nucleation = (
                area <= det_cfg.nucleation_max_area_px and
                area >= det_cfg.nucleation_min_area_px
            )

            self._detection_counter += 1
            detection = CrystalDetection(
                detection_id=self._detection_counter,
                bbox=(x, y, w, h),
                centroid=(cx, cy),
                area_px=area,
                contour=contour,
                mean_intensity=mean_val,
                max_intensity=max_val,
                is_nucleation=is_nucleation,
                circularity=circularity,
                aspect_ratio=aspect_ratio,
                orientation_deg=orientation,
                equivalent_diameter_px=eq_diameter,
                solidity=solidity,
                perimeter_px=perimeter
            )
            detections.append(detection)

        return detections

    def create_detection_mask(self, detections: List[CrystalDetection],
                               shape: Tuple[int, int]) -> np.ndarray:
        """
        Create a binary mask from detections (for visualization or further processing).
        """
        mask = np.zeros(shape[:2], dtype=np.uint8)
        for det in detections:
            cv2.drawContours(mask, [det.contour], -1, 255, -1)
        return mask

    def annotate_image(self, image: np.ndarray,
                       detections: List[CrystalDetection]) -> np.ndarray:
        """
        Draw detection annotations on the image for visualization.

        - Green contours for grown crystals
        - Yellow contours for nucleation events
        - Labels with area and ID
        """
        annotated = image.copy()
        if len(annotated.shape) == 2:
            annotated = cv2.cvtColor(annotated, cv2.COLOR_GRAY2BGR)

        for det in detections:
            if det.is_nucleation:
                color = (0, 255, 255)  # Yellow for nucleation
                label = f"Nuc #{det.detection_id}"
            else:
                color = (0, 255, 0)    # Green for crystal
                label = f"Crys #{det.detection_id}"

            # Draw contour
            cv2.drawContours(annotated, [det.contour], -1, color, 2)

            # Draw centroid
            cx, cy = int(det.centroid[0]), int(det.centroid[1])
            cv2.circle(annotated, (cx, cy), 3, (0, 0, 255), -1)

            # Label
            x, y, w, h = det.bbox
            label_text = f"{label} A={det.area_px:.0f}"
            cv2.putText(annotated, label_text, (x, max(y - 5, 15)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)

        # Summary text
        n_crystals = sum(1 for d in detections if not d.is_nucleation)
        n_nucleation = sum(1 for d in detections if d.is_nucleation)
        total_area = sum(d.area_px for d in detections)
        summary = f"Crystals: {n_crystals} | Nucleation: {n_nucleation} | Total area: {total_area:.0f} px"
        cv2.putText(annotated, summary, (5, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

        return annotated
