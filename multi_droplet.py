"""
Multi-Droplet Manager
=====================
Manages crystallization monitoring across 130 droplets (or any number).
Handles per-droplet analysis, inter-droplet statistics, and heatmap generation.
"""

import cv2
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
from config import PipelineConfig
from crystal_detector import CrystalDetector, CrystalDetection
from growth_tracker import GrowthTracker
from preprocessing import ChannelDetector, ImagePreprocessor


@dataclass
class DropletState:
    """Tracks the current state of a single droplet."""
    droplet_id: int
    # Row, col position in the grid
    grid_position: Tuple[int, int]
    # Has nucleation occurred?
    nucleation_detected: bool = False
    # Frame at which nucleation first occurred
    nucleation_frame: Optional[int] = None
    # Time (sec) at which nucleation first occurred
    nucleation_time_sec: Optional[float] = None
    # Current number of crystals
    current_crystal_count: int = 0
    # Current total crystal area (pixels)
    current_total_area_px: float = 0.0
    # Total number of nucleation events observed
    total_nucleation_events: int = 0
    # Growth tracker for this droplet
    tracker: Optional[GrowthTracker] = None
    # Reference image for this droplet
    reference_image: Optional[np.ndarray] = None
    # Number of frames processed
    frames_processed: int = 0


class MultiDropletManager:
    """
    Orchestrates crystallization analysis across multiple droplets.

    This is the main class you'll use when the gantry system scans
    all 130 droplets. It:

    - Maintains per-droplet state (reference images, trackers, detection history)
    - Routes incoming images to the correct droplet
    - Computes cross-droplet statistics
    - Generates heatmaps and statistical summaries
    """

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.droplets: Dict[int, DropletState] = {}
        self.channel_detector = ChannelDetector(config)
        self.preprocessor = ImagePreprocessor(config)
        self.crystal_detector = CrystalDetector(config)

        # Initialize droplet states
        md_cfg = config.multi_droplet
        for i in range(md_cfg.num_droplets):
            row = i // md_cfg.grid_cols
            col = i % md_cfg.grid_cols
            self.droplets[i] = DropletState(
                droplet_id=i,
                grid_position=(row, col),
                tracker=GrowthTracker(config)
            )

    def set_reference(self, droplet_id: int, reference_image: np.ndarray):
        """Set the clear/blank reference image for a specific droplet."""
        if droplet_id in self.droplets:
            self.droplets[droplet_id].reference_image = reference_image

    def set_all_references(self, reference_images: Dict[int, np.ndarray]):
        """Set reference images for all droplets at once."""
        for did, img in reference_images.items():
            self.set_reference(did, img)

    def process_image(self, droplet_id: int, image: np.ndarray,
                      timestamp: float) -> List[CrystalDetection]:
        """
        Process a new image for a specific droplet.

        This is called each time the gantry captures an image of a droplet.

        Args:
            droplet_id: Which droplet (0-129).
            image: The captured image.
            timestamp: Time in seconds since experiment start.

        Returns:
            List of detections found in this image.
        """
        if droplet_id not in self.droplets:
            raise ValueError(f"Unknown droplet ID: {droplet_id}")

        state = self.droplets[droplet_id]
        frame_idx = state.frames_processed

        # Preprocess
        roi = self.channel_detector.extract_roi(image)
        preprocessed = self.preprocessor.preprocess(roi)

        # Reference subtraction if available
        ref_diff = None
        if state.reference_image is not None:
            ref_roi = self.channel_detector.extract_roi(state.reference_image)
            ref_preprocessed = self.preprocessor.preprocess(ref_roi)
            ref_diff = self.preprocessor.subtract_reference(
                preprocessed, ref_preprocessed
            )

        # Detect crystals
        detections = self.crystal_detector.detect(preprocessed, ref_diff)

        # Update tracker
        state.tracker.update(frame_idx, timestamp, detections)

        # Update state
        state.frames_processed += 1
        state.current_crystal_count = len([d for d in detections if not d.is_nucleation])
        state.current_total_area_px = sum(d.area_px for d in detections)
        state.total_nucleation_events += sum(1 for d in detections if d.is_nucleation)

        # Check for first nucleation
        if not state.nucleation_detected and any(d.is_nucleation for d in detections):
            state.nucleation_detected = True
            state.nucleation_frame = frame_idx
            state.nucleation_time_sec = timestamp

        return detections

    # ---- Cross-droplet statistics ----

    def nucleation_probability(self) -> float:
        """Fraction of droplets that have nucleated."""
        if not self.droplets:
            return 0.0
        nucleated = sum(1 for d in self.droplets.values() if d.nucleation_detected)
        return nucleated / len(self.droplets)

    def nucleation_times(self) -> Dict[int, Optional[float]]:
        """Get nucleation time for each droplet (None if not yet nucleated)."""
        return {
            did: state.nucleation_time_sec
            for did, state in self.droplets.items()
        }

    def crystal_count_map(self) -> np.ndarray:
        """
        Generate a 2D grid of crystal counts (for heatmap visualization).
        Shape: (grid_rows, grid_cols)
        """
        md_cfg = self.config.multi_droplet
        grid = np.zeros((md_cfg.grid_rows, md_cfg.grid_cols), dtype=np.float64)
        for state in self.droplets.values():
            r, c = state.grid_position
            if r < md_cfg.grid_rows and c < md_cfg.grid_cols:
                grid[r, c] = state.current_crystal_count
        return grid

    def nucleation_time_map(self) -> np.ndarray:
        """
        2D grid of nucleation times (NaN if not nucleated).
        """
        md_cfg = self.config.multi_droplet
        grid = np.full((md_cfg.grid_rows, md_cfg.grid_cols), np.nan)
        for state in self.droplets.values():
            r, c = state.grid_position
            if r < md_cfg.grid_rows and c < md_cfg.grid_cols:
                if state.nucleation_time_sec is not None:
                    grid[r, c] = state.nucleation_time_sec
        return grid

    def total_area_map(self) -> np.ndarray:
        """2D grid of total crystal area per droplet."""
        md_cfg = self.config.multi_droplet
        grid = np.zeros((md_cfg.grid_rows, md_cfg.grid_cols), dtype=np.float64)
        for state in self.droplets.values():
            r, c = state.grid_position
            if r < md_cfg.grid_rows and c < md_cfg.grid_cols:
                grid[r, c] = state.current_total_area_px
        return grid

    def growth_rate_map(self) -> np.ndarray:
        """
        2D grid of mean growth rate per droplet.
        Growth rate = mean of all crystal track growth rates in that droplet.
        """
        md_cfg = self.config.multi_droplet
        grid = np.zeros((md_cfg.grid_rows, md_cfg.grid_cols), dtype=np.float64)
        for state in self.droplets.values():
            r, c = state.grid_position
            if r < md_cfg.grid_rows and c < md_cfg.grid_cols:
                tracks = state.tracker.get_all_tracks()
                if tracks:
                    rates = [t.growth_rate_px2_per_sec for t in tracks.values()]
                    grid[r, c] = np.mean(rates) if rates else 0.0
        return grid

    def summary_dataframe(self) -> pd.DataFrame:
        """
        Generate a summary DataFrame with one row per droplet.
        """
        records = []
        for did, state in sorted(self.droplets.items()):
            tracks = state.tracker.get_all_tracks()
            growth_rates = [t.growth_rate_px2_per_sec for t in tracks.values()] if tracks else []

            records.append({
                'droplet_id': did,
                'grid_row': state.grid_position[0],
                'grid_col': state.grid_position[1],
                'nucleation_detected': state.nucleation_detected,
                'nucleation_time_sec': state.nucleation_time_sec,
                'nucleation_frame': state.nucleation_frame,
                'current_crystal_count': state.current_crystal_count,
                'total_nucleation_events': state.total_nucleation_events,
                'current_total_area_px': state.current_total_area_px,
                'num_tracks': len(tracks),
                'mean_growth_rate': np.mean(growth_rates) if growth_rates else 0.0,
                'max_growth_rate': max(growth_rates) if growth_rates else 0.0,
                'frames_processed': state.frames_processed,
            })
        return pd.DataFrame(records)

    def detailed_tracks_dataframe(self) -> pd.DataFrame:
        """
        Generate a detailed DataFrame with all crystal tracks across all droplets.
        Each row = one crystal at one timepoint, with droplet_id added.
        """
        all_dfs = []
        for did, state in self.droplets.items():
            df = state.tracker.to_dataframe()
            if not df.empty:
                df['droplet_id'] = did
                df['grid_row'] = state.grid_position[0]
                df['grid_col'] = state.grid_position[1]
                all_dfs.append(df)
        if all_dfs:
            return pd.concat(all_dfs, ignore_index=True)
        return pd.DataFrame()

    def finalize_all(self):
        """Finalize all droplet trackers (call at end of experiment)."""
        for state in self.droplets.values():
            state.tracker.finalize()
