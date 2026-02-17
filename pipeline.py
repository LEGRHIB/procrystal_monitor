"""
Main Pipeline — Orchestrates the full crystal monitoring workflow.
=================================================================

Two modes of operation:

1. SINGLE DROPLET MODE (current setup):
   Analyze a time-lapse sequence of a single droplet.

2. MULTI-DROPLET MODE (with gantry):
   Process images from 130 droplets scanned by the gantry.

Usage:
    # Single droplet time-lapse
    python pipeline.py --mode single --images ./images/ --reference blank.png

    # Multi-droplet experiment
    python pipeline.py --mode multi --images ./gantry_images/ --reference ./blanks/
"""

import cv2
import numpy as np
import pandas as pd
import os
import glob
import argparse
import time
from typing import List, Tuple, Optional, Dict

from config import PipelineConfig
from preprocessing import ChannelDetector, ImagePreprocessor
from crystal_detector import CrystalDetector, CrystalDetection
from growth_tracker import GrowthTracker
from multi_droplet import MultiDropletManager
from dashboard import CrystalDashboard
from deep_learning import LabelGenerator


class SingleDropletPipeline:
    """
    Analyze crystallization in a single droplet over time.

    Workflow:
    1. Load reference (clear droplet) image
    2. For each time-lapse frame:
       a. Detect channel ROI
       b. Preprocess
       c. Subtract reference
       d. Detect crystals/nucleation
       e. Track crystals
       f. Extract features
    3. Export results (CSV + figures)
    """

    def __init__(self, config: PipelineConfig = None):
        if config is None:
            config = PipelineConfig()
        self.config = config
        self.channel_detector = ChannelDetector(config)
        self.preprocessor = ImagePreprocessor(config)
        self.crystal_detector = CrystalDetector(config)
        self.tracker = GrowthTracker(config)
        self.dashboard = None

        # Storage
        self.reference_image: Optional[np.ndarray] = None
        self.reference_roi: Optional[np.ndarray] = None
        self.roi_bbox: Optional[Tuple[int, int, int, int]] = None
        self.frame_results: List[Dict] = []
        self.all_detections: List[List[CrystalDetection]] = []
        self.processed_images: List[np.ndarray] = []
        self.timestamps: List[float] = []

    def set_reference(self, reference_image: np.ndarray):
        """Set the clear/blank droplet reference image."""
        self.reference_image = reference_image
        # Detect ROI from reference (use this consistently)
        self.roi_bbox = self.channel_detector.detect_channel_roi(reference_image)
        self.reference_roi = self.channel_detector.extract_roi(
            reference_image, self.roi_bbox
        )
        self.reference_roi = self.preprocessor.preprocess(self.reference_roi)
        print(f"  Reference set. ROI: {self.roi_bbox}")

    def process_frame(self, image: np.ndarray, timestamp: float = None,
                      frame_index: int = None) -> List[CrystalDetection]:
        """
        Process a single frame.

        Args:
            image: Raw microscope image.
            timestamp: Time in seconds (auto-increments if not provided).
            frame_index: Frame number (auto-increments if not provided).

        Returns:
            List of crystal detections for this frame.
        """
        if frame_index is None:
            frame_index = len(self.frame_results)
        if timestamp is None:
            timestamp = frame_index * self.config.multi_droplet.snapshot_interval_sec

        # Use consistent ROI if we have it from reference, otherwise detect
        if self.roi_bbox is not None:
            roi = self.channel_detector.extract_roi(image, self.roi_bbox)
        else:
            roi = self.channel_detector.extract_roi(image)

        # Preprocess
        preprocessed = self.preprocessor.preprocess(roi)

        # Reference subtraction
        ref_diff = None
        if self.reference_roi is not None:
            # Ensure same size
            if preprocessed.shape[:2] != self.reference_roi.shape[:2]:
                ref_resized = cv2.resize(
                    self.reference_roi,
                    (preprocessed.shape[1], preprocessed.shape[0])
                )
            else:
                ref_resized = self.reference_roi
            ref_diff = self.preprocessor.subtract_reference(preprocessed, ref_resized)

        # Detect crystals
        detections = self.crystal_detector.detect(preprocessed, ref_diff)

        # Update tracker
        self.tracker.update(frame_index, timestamp, detections)

        # Store results
        self.all_detections.append(detections)
        self.processed_images.append(preprocessed)
        self.timestamps.append(timestamp)

        n_nuc = sum(1 for d in detections if d.is_nucleation)
        n_crys = sum(1 for d in detections if not d.is_nucleation)
        total_area = sum(d.area_px for d in detections)

        result = {
            'frame': frame_index,
            'timestamp_sec': timestamp,
            'num_nucleation': n_nuc,
            'num_crystals': n_crys,
            'total_crystal_area_px': total_area,
            'mean_intensity': np.mean([d.mean_intensity for d in detections]) if detections else 0,
            'max_area_px': max((d.area_px for d in detections), default=0),
        }
        self.frame_results.append(result)

        return detections

    def process_timelapse(self, image_paths: List[str],
                          timestamps: Optional[List[float]] = None,
                          reference_path: Optional[str] = None,
                          verbose: bool = True) -> pd.DataFrame:
        """
        Process a full time-lapse sequence.

        Args:
            image_paths: List of paths to time-lapse images (in order).
            timestamps: Optional list of timestamps in seconds.
            reference_path: Path to clear/blank reference image.
            verbose: Print progress.

        Returns:
            DataFrame with per-frame summary statistics.
        """
        # Load and set reference
        if reference_path:
            ref = cv2.imread(reference_path)
            if ref is not None:
                self.set_reference(ref)
                if verbose:
                    print(f"  Loaded reference: {reference_path}")
            else:
                print(f"  WARNING: Could not load reference: {reference_path}")

        # Generate timestamps if not provided
        if timestamps is None:
            dt = self.config.multi_droplet.snapshot_interval_sec
            timestamps = [i * dt for i in range(len(image_paths))]

        if verbose:
            print(f"  Processing {len(image_paths)} frames...")

        for i, (path, ts) in enumerate(zip(image_paths, timestamps)):
            image = cv2.imread(path)
            if image is None:
                print(f"  WARNING: Could not load {path}, skipping.")
                continue

            detections = self.process_frame(image, timestamp=ts, frame_index=i)

            if verbose and (i % max(1, len(image_paths) // 10) == 0 or i == len(image_paths) - 1):
                n_det = len(detections)
                print(f"  Frame {i+1}/{len(image_paths)} | t={ts:.1f}s | "
                      f"Detections: {n_det}")

        # Finalize
        self.tracker.finalize()

        if verbose:
            tracks = self.tracker.get_all_tracks()
            print(f"\n  Done! {len(tracks)} crystal tracks identified.")

        return pd.DataFrame(self.frame_results)

    def export_results(self, output_dir: str = "./results", verbose: bool = True):
        """
        Export all results: CSVs, figures, and annotated images.
        """
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, "annotated"), exist_ok=True)
        figures_dir = os.path.join(output_dir, "figures")
        os.makedirs(figures_dir, exist_ok=True)

        # 1) Frame-level summary CSV
        frame_df = pd.DataFrame(self.frame_results)
        frame_path = os.path.join(output_dir, "frame_summary.csv")
        frame_df.to_csv(frame_path, index=False)
        if verbose:
            print(f"  Saved frame summary: {frame_path}")

        # 2) Track-level data CSV
        track_df = self.tracker.to_dataframe()
        if not track_df.empty:
            track_path = os.path.join(output_dir, "crystal_tracks.csv")
            track_df.to_csv(track_path, index=False)
            if verbose:
                print(f"  Saved crystal tracks: {track_path}")

        # 3) Track summary CSV
        summary_df = self.tracker.summary_dataframe()
        if not summary_df.empty:
            summary_path = os.path.join(output_dir, "track_summary.csv")
            summary_df.to_csv(summary_path, index=False)
            if verbose:
                print(f"  Saved track summary: {summary_path}")

        # 4) Growth curve plots
        self.dashboard = CrystalDashboard(self.config, output_dir=figures_dir)
        self.dashboard.plot_growth_curves(self.tracker, droplet_id=0)

        # 5) Annotated images
        if self.config.output.save_annotated_images:
            for i, (img, dets) in enumerate(zip(self.processed_images, self.all_detections)):
                annotated = self.crystal_detector.annotate_image(img, dets)
                ann_path = os.path.join(output_dir, "annotated", f"frame_{i:05d}.png")
                cv2.imwrite(ann_path, annotated)

            if verbose:
                print(f"  Saved {len(self.processed_images)} annotated images")

        # 6) Time-lapse montage
        if len(self.processed_images) > 1:
            self.dashboard.create_timelapse_montage(
                self.processed_images, self.all_detections,
                self.timestamps, cols=min(6, len(self.processed_images))
            )

        # 7) Save config
        config_path = os.path.join(output_dir, "pipeline_config.json")
        self.config.save(config_path)
        if verbose:
            print(f"  Saved config: {config_path}")

        if verbose:
            print(f"\n  All results exported to: {output_dir}")

    def generate_training_data(self, output_dir: str = "./training_data"):
        """
        Export current detections as training data for the U-Net.
        Each frame's image and detection mask become a training pair.
        """
        label_gen = LabelGenerator(output_dir)

        for i, (img, dets) in enumerate(zip(self.processed_images, self.all_detections)):
            if not dets:
                continue

            # Create mask: 0=bg, 128=nucleation, 255=crystal
            mask = np.zeros(img.shape[:2], dtype=np.uint8)
            for det in dets:
                value = 128 if det.is_nucleation else 255
                cv2.drawContours(mask, [det.contour], -1, value, -1)

            sample_id = f"frame_{i:05d}"
            label_gen.save_training_pair(
                img, mask, sample_id,
                metadata={
                    'frame': i,
                    'timestamp': self.timestamps[i] if i < len(self.timestamps) else None,
                    'num_detections': len(dets),
                }
            )

        label_gen.save_metadata()
        print(f"  Training data saved to {output_dir} ({len(self.processed_images)} pairs)")


# ============================================================
# CLI ENTRY POINT
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description='Crystal Monitor — Automated crystal detection in microfluidic droplets'
    )
    parser.add_argument('--mode', choices=['single', 'multi'], default='single',
                        help='Analysis mode: single droplet or multi-droplet')
    parser.add_argument('--images', required=True,
                        help='Directory containing time-lapse images (or glob pattern)')
    parser.add_argument('--reference', default=None,
                        help='Path to clear/blank reference image')
    parser.add_argument('--output', default='./results',
                        help='Output directory for results')
    parser.add_argument('--config', default=None,
                        help='Path to JSON config file')
    parser.add_argument('--interval', type=float, default=60.0,
                        help='Time interval between frames (seconds)')
    parser.add_argument('--generate-training', action='store_true',
                        help='Also generate U-Net training data from detections')
    parser.add_argument('--brightness-threshold', type=int, default=None,
                        help='Override brightness threshold for detection')
    parser.add_argument('--difference-threshold', type=int, default=None,
                        help='Override difference threshold for detection')
    parser.add_argument('--min-area', type=int, default=None,
                        help='Override minimum crystal area (pixels)')

    args = parser.parse_args()

    # Load or create config
    if args.config:
        config = PipelineConfig.load(args.config)
    else:
        config = PipelineConfig()

    # Apply CLI overrides
    config.multi_droplet.snapshot_interval_sec = args.interval
    if args.brightness_threshold is not None:
        config.detection.brightness_threshold = args.brightness_threshold
    if args.difference_threshold is not None:
        config.detection.difference_threshold = args.difference_threshold
    if args.min_area is not None:
        config.detection.min_crystal_area_px = args.min_area

    # Find image files
    if '*' in args.images:
        image_paths = sorted(glob.glob(args.images))
    else:
        extensions = ['*.png', '*.jpg', '*.jpeg', '*.tif', '*.tiff', '*.bmp']
        image_paths = []
        for ext in extensions:
            image_paths.extend(glob.glob(os.path.join(args.images, ext)))
        image_paths = sorted(image_paths)

    if not image_paths:
        print(f"ERROR: No images found in {args.images}")
        return

    print(f"Crystal Monitor Pipeline v1.0")
    print(f"=" * 50)
    print(f"Mode: {args.mode}")
    print(f"Images: {len(image_paths)} files")
    print(f"Reference: {args.reference or 'None (will use first frame)'}")
    print(f"Output: {args.output}")
    print(f"=" * 50)

    if args.mode == 'single':
        pipeline = SingleDropletPipeline(config)

        # Use first image as reference if none provided
        ref_path = args.reference or image_paths[0]

        # If reference is the first frame, start processing from frame 1
        start_idx = 0
        if args.reference is None:
            start_idx = 1

        frame_df = pipeline.process_timelapse(
            image_paths[start_idx:],
            reference_path=ref_path,
            verbose=True
        )

        pipeline.export_results(args.output)

        if args.generate_training:
            pipeline.generate_training_data(
                os.path.join(args.output, "training_data")
            )

    elif args.mode == 'multi':
        print("Multi-droplet mode requires gantry integration.")
        print("See MultiDropletManager class for programmatic usage.")
        print("Example workflow:")
        print("  manager = MultiDropletManager(config)")
        print("  manager.set_reference(0, blank_image)")
        print("  manager.process_image(0, image, timestamp)")
        print("  df = manager.summary_dataframe()")

    print("\nDone!")


if __name__ == '__main__':
    main()
