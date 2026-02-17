"""
Crystal Monitor — Automated Nucleation & Crystal Growth Detection
=================================================================
A Python pipeline for analyzing polarized microscopy images of
biomolecule crystallization in microfluidic droplets.

Modules:
    config          — Configuration parameters
    preprocessing   — Image preprocessing, ROI extraction, channel segmentation
    crystal_detector — Crystal and nucleation detection
    feature_extractor — Morphological feature extraction
    growth_tracker  — Time-series crystal tracking
    multi_droplet   — Multi-droplet experiment manager
    statistics      — Statistical analysis and kinetics
    dashboard       — Real-time visualization
    deep_learning   — U-Net segmentation (requires PyTorch)
    pipeline        — Main orchestrator
"""

__version__ = "1.0.0"
__author__ = "Crystal Monitor Pipeline"
