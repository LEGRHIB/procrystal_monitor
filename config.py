"""
Configuration for Crystal Monitor Pipeline
==========================================
Adjust these parameters to match your specific microscope setup,
channel geometry, and crystallization conditions.
"""

from dataclasses import dataclass, field
from typing import Tuple, Optional
import json
import os


@dataclass
class ChannelConfig:
    """Microfluidic channel geometry parameters."""
    # Approximate channel width in pixels (will be auto-detected if possible)
    channel_width_px: int = 200
    # Whether the channel runs vertically (True) or horizontally (False) in images
    vertical_channel: bool = True
    # Margin (pixels) to exclude from channel edges to avoid wall artifacts
    wall_margin_px: int = 10


@dataclass
class DetectionConfig:
    """Crystal detection parameters."""
    # --- Brightness-based detection (birefringent crystals under crossed polarizers) ---
    # Minimum brightness threshold (0-255) for detecting bright crystals
    brightness_threshold: int = 45
    # Adaptive threshold block size (must be odd)
    adaptive_block_size: int = 31
    # Adaptive threshold constant subtracted from mean
    adaptive_constant: int = -8

    # --- Difference-based detection (comparing to reference/blank) ---
    # Minimum pixel difference from reference to flag as crystal
    difference_threshold: int = 30
    # Use reference subtraction method (requires a blank/clear image)
    use_reference_subtraction: bool = True

    # --- Morphological filtering ---
    # Minimum crystal area in pixels (filter out noise)
    min_crystal_area_px: int = 50
    # Maximum crystal area in pixels (filter out large artifacts)
    max_crystal_area_px: int = 50000
    # Minimum circularity (0-1) to keep a detection (0 = any shape)
    min_circularity: float = 0.0
    # Morphological opening kernel size (removes small noise)
    morph_open_kernel: int = 3
    # Morphological closing kernel size (fills small gaps)
    morph_close_kernel: int = 5

    # --- Nucleation detection ---
    # Minimum area (px) for a spot to count as nucleation event
    nucleation_min_area_px: int = 15
    # Maximum area (px) for a spot to still be "nucleation" vs "crystal"
    nucleation_max_area_px: int = 300
    # Intensity z-score above local mean to flag as nucleation
    nucleation_sensitivity: float = 3.0


@dataclass
class TrackingConfig:
    """Crystal growth tracking parameters."""
    # Maximum distance (px) between frames to link same crystal
    max_linking_distance_px: int = 50
    # Number of frames a crystal can disappear before losing track
    max_gap_frames: int = 3
    # Minimum track length (frames) to be considered valid
    min_track_length: int = 3


@dataclass
class MultiDropletConfig:
    """Configuration for multi-droplet (130 wells) experiments."""
    # Total number of droplets/wells
    num_droplets: int = 130
    # Grid layout (rows x cols) if applicable
    grid_rows: int = 10
    grid_cols: int = 13
    # Time interval between snapshots of same droplet (seconds)
    snapshot_interval_sec: float = 60.0
    # Time to complete one full scan of all droplets (seconds)
    gantry_cycle_time_sec: float = 300.0


@dataclass
class OutputConfig:
    """Output and export configuration."""
    # Directory for CSV/Excel exports
    export_dir: str = "./results"
    # Directory for annotated images
    annotated_dir: str = "./results/annotated"
    # Save annotated images with detections overlaid
    save_annotated_images: bool = True
    # Export format
    export_format: str = "csv"  # "csv" or "xlsx"
    # Image format for annotated outputs
    image_format: str = "png"


@dataclass
class PipelineConfig:
    """Master configuration combining all sub-configs."""
    channel: ChannelConfig = field(default_factory=ChannelConfig)
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    multi_droplet: MultiDropletConfig = field(default_factory=MultiDropletConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    def save(self, path: str):
        """Save configuration to JSON."""
        import dataclasses
        def _to_dict(obj):
            if dataclasses.is_dataclass(obj):
                return {k: _to_dict(v) for k, v in dataclasses.asdict(obj).items()}
            return obj
        with open(path, 'w') as f:
            json.dump(_to_dict(self), f, indent=2)

    @classmethod
    def load(cls, path: str) -> 'PipelineConfig':
        """Load configuration from JSON."""
        with open(path, 'r') as f:
            data = json.load(f)
        config = cls()
        config.channel = ChannelConfig(**data.get('channel', {}))
        config.detection = DetectionConfig(**data.get('detection', {}))
        config.tracking = TrackingConfig(**data.get('tracking', {}))
        config.multi_droplet = MultiDropletConfig(**data.get('multi_droplet', {}))
        config.output = OutputConfig(**data.get('output', {}))
        return config


# Default configuration instance
DEFAULT_CONFIG = PipelineConfig()
