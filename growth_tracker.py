"""
Growth Tracker Module
=====================
Tracks individual crystals across frames and computes growth kinetics.
Implements a simple nearest-neighbor linking algorithm.
"""

import cv2
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
from crystal_detector import CrystalDetection
from config import PipelineConfig
import math


@dataclass
class CrystalTrack:
    """
    Represents a tracked crystal across multiple frames.

    Stores the full history of a crystal from nucleation to current state.
    """
    track_id: int
    # Frame indices where this crystal was observed
    frame_indices: List[int] = field(default_factory=list)
    # Timestamps (seconds) for each frame
    timestamps: List[float] = field(default_factory=list)
    # Area at each frame (pixels)
    areas: List[float] = field(default_factory=list)
    # Centroid at each frame
    centroids: List[Tuple[float, float]] = field(default_factory=list)
    # Equivalent diameters at each frame
    diameters: List[float] = field(default_factory=list)
    # Aspect ratios at each frame
    aspect_ratios: List[float] = field(default_factory=list)
    # Mean intensities at each frame
    intensities: List[float] = field(default_factory=list)
    # Circularities at each frame
    circularities: List[float] = field(default_factory=list)
    # Whether nucleation was the first event
    nucleated: bool = False
    # Frame index where nucleation was first detected
    nucleation_frame: Optional[int] = None
    # Number of consecutive frames this crystal has been missing
    _gap_counter: int = 0

    @property
    def duration_sec(self) -> float:
        """Total observed duration in seconds."""
        if len(self.timestamps) < 2:
            return 0.0
        return self.timestamps[-1] - self.timestamps[0]

    @property
    def growth_rate_px2_per_sec(self) -> float:
        """
        Average linear growth rate in area (px²/sec).
        Computed as slope of area vs time via least squares.
        """
        if len(self.areas) < 2 or self.duration_sec == 0:
            return 0.0
        t = np.array(self.timestamps) - self.timestamps[0]
        a = np.array(self.areas)
        # Simple linear fit: area = rate * t + offset
        if len(t) >= 2:
            coeffs = np.polyfit(t, a, 1)
            return coeffs[0]  # slope = growth rate
        return 0.0

    @property
    def growth_rate_diameter_per_sec(self) -> float:
        """Growth rate in equivalent diameter per second."""
        if len(self.diameters) < 2 or self.duration_sec == 0:
            return 0.0
        t = np.array(self.timestamps) - self.timestamps[0]
        d = np.array(self.diameters)
        if len(t) >= 2:
            coeffs = np.polyfit(t, d, 1)
            return coeffs[0]
        return 0.0

    @property
    def current_area(self) -> float:
        return self.areas[-1] if self.areas else 0.0

    @property
    def current_centroid(self) -> Tuple[float, float]:
        return self.centroids[-1] if self.centroids else (0.0, 0.0)


class GrowthTracker:
    """
    Links crystal detections across frames using nearest-neighbor matching.

    Algorithm:
    1. For each new frame, compute distances from existing track endpoints
       to new detections.
    2. Use a greedy assignment: assign closest pairs first (within max distance).
    3. Unmatched detections become new tracks.
    4. Unmatched tracks increment their gap counter; if exceeded, they are closed.
    """

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.active_tracks: Dict[int, CrystalTrack] = {}
        self.completed_tracks: Dict[int, CrystalTrack] = {}
        self._next_track_id = 1
        self._current_frame = -1

    def update(self, frame_index: int, timestamp: float,
               detections: List[CrystalDetection]) -> Dict[int, CrystalTrack]:
        """
        Process a new frame's detections and update tracks.

        Args:
            frame_index: Sequential frame number.
            timestamp: Time in seconds since experiment start.
            detections: List of detections from CrystalDetector.

        Returns:
            Dict of currently active tracks (track_id -> CrystalTrack).
        """
        self._current_frame = frame_index
        track_cfg = self.config.tracking

        if not self.active_tracks:
            # First frame: create a track for each detection
            for det in detections:
                self._create_track(det, frame_index, timestamp)
            return self.active_tracks

        # Compute cost matrix (distance between track endpoints and new detections)
        track_ids = list(self.active_tracks.keys())
        n_tracks = len(track_ids)
        n_dets = len(detections)

        if n_tracks == 0:
            for det in detections:
                self._create_track(det, frame_index, timestamp)
            return self.active_tracks

        if n_dets == 0:
            # No detections: increment gap counters
            self._handle_no_detections(track_cfg)
            return self.active_tracks

        # Build distance matrix
        cost_matrix = np.full((n_tracks, n_dets), np.inf)
        for i, tid in enumerate(track_ids):
            track = self.active_tracks[tid]
            tc = track.current_centroid
            for j, det in enumerate(detections):
                dc = det.centroid
                dist = math.sqrt((tc[0] - dc[0])**2 + (tc[1] - dc[1])**2)
                if dist <= track_cfg.max_linking_distance_px:
                    cost_matrix[i, j] = dist

        # Greedy assignment
        matched_tracks = set()
        matched_dets = set()
        assignments = []

        # Sort all valid pairs by distance
        valid_pairs = []
        for i in range(n_tracks):
            for j in range(n_dets):
                if cost_matrix[i, j] < np.inf:
                    valid_pairs.append((cost_matrix[i, j], i, j))
        valid_pairs.sort(key=lambda x: x[0])

        for dist, i, j in valid_pairs:
            if i not in matched_tracks and j not in matched_dets:
                assignments.append((i, j))
                matched_tracks.add(i)
                matched_dets.add(j)

        # Update matched tracks
        for i, j in assignments:
            tid = track_ids[i]
            det = detections[j]
            self._update_track(tid, det, frame_index, timestamp)

        # Handle unmatched tracks (increment gap)
        for i, tid in enumerate(track_ids):
            if i not in matched_tracks:
                track = self.active_tracks[tid]
                track._gap_counter += 1
                if track._gap_counter > track_cfg.max_gap_frames:
                    # Close this track
                    if len(track.frame_indices) >= track_cfg.min_track_length:
                        self.completed_tracks[tid] = track
                    del self.active_tracks[tid]

        # Create new tracks for unmatched detections
        for j, det in enumerate(detections):
            if j not in matched_dets:
                self._create_track(det, frame_index, timestamp)

        return self.active_tracks

    def _create_track(self, det: CrystalDetection,
                      frame_index: int, timestamp: float):
        """Create a new track from a detection."""
        track = CrystalTrack(
            track_id=self._next_track_id,
            frame_indices=[frame_index],
            timestamps=[timestamp],
            areas=[det.area_px],
            centroids=[det.centroid],
            diameters=[det.equivalent_diameter_px],
            aspect_ratios=[det.aspect_ratio],
            intensities=[det.mean_intensity],
            circularities=[det.circularity],
            nucleated=det.is_nucleation,
            nucleation_frame=frame_index if det.is_nucleation else None
        )
        self.active_tracks[self._next_track_id] = track
        self._next_track_id += 1

    def _update_track(self, track_id: int, det: CrystalDetection,
                      frame_index: int, timestamp: float):
        """Add a new detection to an existing track."""
        track = self.active_tracks[track_id]
        track.frame_indices.append(frame_index)
        track.timestamps.append(timestamp)
        track.areas.append(det.area_px)
        track.centroids.append(det.centroid)
        track.diameters.append(det.equivalent_diameter_px)
        track.aspect_ratios.append(det.aspect_ratio)
        track.intensities.append(det.mean_intensity)
        track.circularities.append(det.circularity)
        track._gap_counter = 0

        # If not already flagged as nucleated, check if this is nucleation
        if not track.nucleated and det.is_nucleation:
            track.nucleated = True
            track.nucleation_frame = frame_index

    def _handle_no_detections(self, track_cfg):
        """Handle frames with no detections."""
        to_close = []
        for tid, track in self.active_tracks.items():
            track._gap_counter += 1
            if track._gap_counter > track_cfg.max_gap_frames:
                if len(track.frame_indices) >= track_cfg.min_track_length:
                    self.completed_tracks[tid] = track
                to_close.append(tid)
        for tid in to_close:
            del self.active_tracks[tid]

    def finalize(self):
        """Move all active tracks to completed (call at end of experiment)."""
        for tid, track in self.active_tracks.items():
            if len(track.frame_indices) >= self.config.tracking.min_track_length:
                self.completed_tracks[tid] = track
        self.active_tracks.clear()

    def get_all_tracks(self) -> Dict[int, CrystalTrack]:
        """Get all tracks (both active and completed)."""
        all_tracks = {}
        all_tracks.update(self.completed_tracks)
        all_tracks.update(self.active_tracks)
        return all_tracks

    def to_dataframe(self) -> pd.DataFrame:
        """
        Export all track data to a pandas DataFrame.

        Each row is one crystal at one timepoint.
        """
        records = []
        for tid, track in self.get_all_tracks().items():
            for i in range(len(track.frame_indices)):
                records.append({
                    'track_id': track.track_id,
                    'frame': track.frame_indices[i],
                    'timestamp_sec': track.timestamps[i],
                    'area_px': track.areas[i],
                    'centroid_x': track.centroids[i][0],
                    'centroid_y': track.centroids[i][1],
                    'equiv_diameter_px': track.diameters[i],
                    'aspect_ratio': track.aspect_ratios[i],
                    'mean_intensity': track.intensities[i],
                    'circularity': track.circularities[i],
                    'is_nucleation': track.nucleated,
                    'nucleation_frame': track.nucleation_frame,
                    'growth_rate_area': track.growth_rate_px2_per_sec,
                    'growth_rate_diameter': track.growth_rate_diameter_per_sec,
                })
        return pd.DataFrame(records)

    def summary_dataframe(self) -> pd.DataFrame:
        """
        Summary DataFrame: one row per crystal track with aggregated stats.
        """
        records = []
        for tid, track in self.get_all_tracks().items():
            records.append({
                'track_id': track.track_id,
                'nucleated': track.nucleated,
                'nucleation_frame': track.nucleation_frame,
                'first_frame': track.frame_indices[0],
                'last_frame': track.frame_indices[-1],
                'duration_sec': track.duration_sec,
                'num_observations': len(track.frame_indices),
                'initial_area_px': track.areas[0],
                'final_area_px': track.areas[-1],
                'max_area_px': max(track.areas),
                'initial_diameter_px': track.diameters[0],
                'final_diameter_px': track.diameters[-1],
                'growth_rate_area_px2_s': track.growth_rate_px2_per_sec,
                'growth_rate_diam_px_s': track.growth_rate_diameter_per_sec,
                'mean_aspect_ratio': np.mean(track.aspect_ratios),
                'mean_circularity': np.mean(track.circularities),
                'mean_intensity': np.mean(track.intensities),
            })
        return pd.DataFrame(records)
