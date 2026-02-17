# Crystal Monitor

**Automated nucleation and crystal growth detection for biomolecule crystallization in microfluidic droplets using polarized microscopy.**

---

## Overview

Crystal Monitor is a Python-based image analysis pipeline that automatically detects, tracks, and quantifies crystal nucleation and growth in time-lapse microscopy images. Designed for high-throughput crystallization screening in microfluidic devices with support for up to 130 parallel droplets.

### Key Features

- ✅ **Automatic droplet detection** — Isolates droplet region within microfluidic channels
- ✅ **Dual-mode crystal detection** — Combines brightness-based (birefringence) and reference subtraction methods
- ✅ **Growth tracking** — Links crystals across frames, computes growth rates
- ✅ **Multi-droplet support** — Parallel analysis of 130 droplets with statistical mapping
- ✅ **Real-time capture** — Automated image acquisition with dual local/cloud storage
- ✅ **Rich visualization** — Growth curves, time-lapse montages, heatmaps
- ✅ **Deep learning ready** — U-Net segmentation stub for future training

---

## Quick Start

### Installation

```bash
# Clone or copy the crystal_monitor folder
cd crystal_monitor

# Install dependencies
pip install opencv-python numpy pandas matplotlib scipy

# Test the pipeline
python test_pipeline.py
```

### Basic Usage

#### 1. Capture images
```bash
# Single droplet experiment
python autocapture.py --mode single --interval 1800 --name my_experiment
```

#### 2. Analyze experiment
```bash
# Analyze captured images
python analyze_experiment.py --experiment ~/CrystalMonitor/my_experiment
```

#### 3. View results
Results are saved to `<experiment>/results/`:
- `frame_summary.csv` — Per-frame crystal counts
- `crystal_tracks.csv` — Full growth trajectories
- `track_summary.csv` — Growth rates per crystal
- `figures/` — Growth curves and visualizations
- `annotated/` — Images with detected crystals overlaid

---

## Architecture

### Pipeline Overview

```
┌──────────────────────────────────────────────────────────────┐
│                   CRYSTAL MONITOR PIPELINE                    │
└──────────────────────────────────────────────────────────────┘

INPUT: Time-lapse microscopy images
  │
  ├─► [1] PREPROCESSING (preprocessing.py)
  │       ├─ Channel/tube ROI detection
  │       ├─ Droplet boundary detection
  │       ├─ Denoising & contrast enhancement
  │       └─ Reference image subtraction
  │
  ├─► [2] CRYSTAL DETECTION (crystal_detector.py)
  │       ├─ Brightness-based detection (birefringence)
  │       ├─ Reference difference detection
  │       ├─ Morphological filtering
  │       └─ Feature extraction (area, shape, intensity)
  │
  ├─► [3] GROWTH TRACKING (growth_tracker.py)
  │       ├─ Nearest-neighbor linking
  │       ├─ Track management (gap handling)
  │       └─ Growth rate computation
  │
  ├─► [4] MULTI-DROPLET MANAGER (multi_droplet.py)
  │       ├─ Per-droplet state tracking
  │       ├─ Cross-droplet statistics
  │       └─ Nucleation probability analysis
  │
  └─► [5] VISUALIZATION (dashboard.py)
          ├─ Growth curves
          ├─ Time-lapse montages
          ├─ Multi-droplet heatmaps
          └─ Statistical summaries

OUTPUT: CSV data + Figures + Annotated images
```

### Module Responsibilities

| Module | Purpose | Key Functions |
|--------|---------|---------------|
| `config.py` | Configuration management | Parameter definitions, JSON I/O |
| `preprocessing.py` | Image preparation | ROI detection, denoising, CLAHE, reference subtraction |
| `crystal_detector.py` | Crystal detection | Threshold-based + diff-based detection, feature extraction |
| `growth_tracker.py` | Time-series tracking | Crystal linking, track management, growth rate fitting |
| `multi_droplet.py` | Multi-well experiments | Droplet state management, statistical aggregation |
| `dashboard.py` | Visualization | Growth plots, montages, heatmaps |
| `deep_learning.py` | U-Net segmentation | Model architecture, training loop (requires PyTorch) |
| `pipeline.py` | Main orchestrator | CLI, single-droplet workflow |
| `autocapture.py` | Image acquisition | Camera control, dual storage, experiment metadata |

---

## Configuration

All detection parameters are defined in `config.py`. Key tuning parameters:

```python
# Detection sensitivity
brightness_threshold = 60      # Higher = less sensitive
difference_threshold = 40      # Higher = ignore small changes
min_crystal_area_px = 100      # Minimum crystal size

# Channel geometry
vertical_channel = True        # Channel orientation
wall_margin_px = 30           # Crop from tube edges

# Tracking
max_linking_distance_px = 50  # Max distance to link crystals between frames
max_gap_frames = 3            # Frames a crystal can disappear before losing track
```

**Tuning tips:**
- Too many false positives? → Increase thresholds
- Missing real crystals? → Decrease thresholds
- IDs jumping? → Increase `max_linking_distance_px`

---

## Data Format

### Folder Structure (Created by `autocapture.py`)
```
experiment_YYYYMMDD_HHMMSS/
├── experiment.json              # Metadata
├── capture_log.csv              # Capture log
├── references/                  # Clear droplet images
│   └── droplet_NNN.png
└── captures/                    # Time-lapse images
    └── droplet_NNN/
        └── cycle_CCCC_tTTTTTTs.png
```

### Naming Convention
- **Droplets**: `droplet_000` to `droplet_129` (3-digit zero-padded)
- **Frames**: `cycle_CCCC_tTTTTTTs.png`
  - `CCCC` = cycle number (gantry sweep or frame index)
  - `TTTTTT` = elapsed seconds since experiment start

### CSV Outputs

**frame_summary.csv**
```csv
frame,timestamp_sec,num_nucleation,num_crystals,total_crystal_area_px,...
```

**crystal_tracks.csv**
```csv
track_id,frame,timestamp_sec,area_px,centroid_x,centroid_y,equiv_diameter_px,...
```

**track_summary.csv**
```csv
track_id,nucleated,nucleation_frame,duration_sec,growth_rate_area_px2_s,...
```

---

## Multi-Droplet Workflow

### Setup (130 droplets with gantry)

```python
from multi_droplet import MultiDropletManager
from config import PipelineConfig

# Configure
config = PipelineConfig()
config.multi_droplet.num_droplets = 130
config.multi_droplet.grid_rows = 10
config.multi_droplet.grid_cols = 13

# Initialize manager
manager = MultiDropletManager(config)

# Phase 1: Capture reference images (once)
for droplet_id in range(130):
    ref_image = capture_clear_droplet(droplet_id)  # Your gantry function
    manager.set_reference(droplet_id, ref_image)

# Phase 2: Monitor crystallization (timed sweeps)
for cycle in range(num_cycles):
    for droplet_id in range(130):
        image = capture_image(droplet_id)  # Your gantry function
        timestamp = time.time() - experiment_start
        detections = manager.process_image(droplet_id, image, timestamp)

# Phase 3: Generate reports
manager.finalize_all()
summary_df = manager.summary_dataframe()
summary_df.to_csv("multi_droplet_summary.csv")

# Generate heatmaps
from dashboard import CrystalDashboard
dashboard = CrystalDashboard(config, output_dir="./figures")
dashboard.plot_multi_droplet_heatmaps(manager)
dashboard.plot_nucleation_statistics(manager)
```

---

## Extending the Pipeline

### Adding a New Detection Method

1. **Create detector class** in `crystal_detector.py`:
```python
class MyCustomDetector:
    def detect(self, image: np.ndarray) -> List[CrystalDetection]:
        # Your detection logic
        pass
```

2. **Integrate in pipeline**:
```python
# In SingleDropletPipeline.process_frame()
custom_detections = my_detector.detect(preprocessed)
all_detections = combine(classic_detections, custom_detections)
```

3. **Add config parameters** in `config.py`:
```python
@dataclass
class DetectionConfig:
    # ... existing params ...
    custom_detector_threshold: float = 0.5
```

### Adding New Features

**New morphological features**:
- Edit `crystal_detector.py` → `_extract_detections()`
- Add to `CrystalDetection` dataclass
- Feature automatically exported to CSV

**New visualization**:
- Add method to `CrystalDashboard` class
- Use matplotlib for plotting
- Save to `output_dir`

**New tracking metric**:
- Edit `GrowthTracker` class
- Add property to `CrystalTrack` dataclass
- Include in `to_dataframe()` export

### Deep Learning Integration

**When to switch to U-Net:**
- You have 100+ labeled images
- Complex crystal morphologies
- Classical methods miss crystals

**Training workflow:**
```python
# 1. Generate training data from classical detections
pipeline.generate_training_data("./training_data")

# 2. Manually review/correct labels (visual inspection)

# 3. Train U-Net
from deep_learning import UNet, CrystalDataset, UNetTrainer

model = UNet(in_channels=1, num_classes=3)
train_data = CrystalDataset("./training_data/images", "./training_data/masks")
trainer = UNetTrainer(model, train_data)
trainer.train(epochs=50)
trainer.save_model("crystal_unet.pth")

# 4. Use U-Net in pipeline
model.load_model("crystal_unet.pth")
predictions = model.predict(image)  # Returns class map
```

---

## Performance Considerations

### Computational Complexity

**Per-frame processing time** (typical):
- Preprocessing: ~50ms
- Crystal detection: ~100ms
- Tracking: ~10ms
- **Total**: ~160ms per frame

**Bottlenecks:**
- CLAHE (contrast enhancement) — most expensive
- Connected components extraction — scales with crystal count
- Tracking distance matrix — O(n × m) where n=tracks, m=detections

**Optimization strategies:**
1. **Reduce image size** — Downsample if resolution is excessive
2. **Skip preprocessing** — If images are high quality
3. **Parallel processing** — Batch process multiple droplets
4. **GPU acceleration** — Use OpenCV GPU functions or switch to U-Net

### Memory Usage

- **Single frame**: ~1-5 MB (640×480 RGB image)
- **Time-lapse (100 frames)**: ~100-500 MB in memory
- **Multi-droplet (130 wells, 100 frames)**: ~13-65 GB total

**For large experiments:**
- Process droplets sequentially (one at a time)
- Stream images from disk rather than loading all
- Use generators instead of lists

---

## Troubleshooting

### Common Issues

**Q: Getting false detections on tube edges**
- A: Increase `wall_margin_px` or use droplet detection (`detect_droplet.py`)

**Q: Missing small nucleation events**
- A: Decrease `nucleation_min_area_px` and `brightness_threshold`

**Q: Crystal IDs jump between frames**
- A: Increase `max_linking_distance_px` and `max_gap_frames`

**Q: Images look too dark/noisy**
- A: Check camera exposure settings, enable denoising in preprocessing

**Q: Analysis is slow**
- A: Reduce image resolution, disable CLAHE, or skip annotated image saving

---

## Development Roadmap

### Near-term (Next 1-3 months)
- [ ] Gantry integration for 130-droplet scanning
- [ ] Real-time dashboard (live monitoring during experiments)
- [ ] Automated parameter tuning (grid search on validation set)
- [ ] Export to HDF5 for large datasets

### Mid-term (3-6 months)
- [ ] U-Net training pipeline with active learning
- [ ] Multi-condition statistical analysis (pH, temp, concentration)
- [ ] 3D crystal reconstruction from multiple angles
- [ ] Integration with LIMS (Lab Information Management System)

### Long-term (6-12 months)
- [ ] Predictive modeling (nucleation probability from time-series)
- [ ] Federated learning across multiple labs
- [ ] Mobile app for remote monitoring
- [ ] Automated crystallization condition optimization (closed-loop)

---

## Citation

If you use Crystal Monitor in your research, please cite:

```bibtex
@software{crystal_monitor_2026,
  title = {Crystal Monitor: Automated Crystal Nucleation and Growth Detection},
  author = {Your Name},
  year = {2026},
  url = {https://github.com/yourusername/crystal_monitor}
}
```

---

## License

MIT License — see `LICENSE` file for details.

---

## Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

---

## Support

**Issues**: Report bugs or request features via GitHub Issues

**Contact**: youcef.leghrib@kuleuven.be

**Documentation**: See `SESSION_NOTES_FEB16_2026.md` for detailed setup guide

---

## Acknowledgments

Developed at KU Leuven for high-throughput biomolecule crystallization screening.

Built with: OpenCV, NumPy, Pandas, Matplotlib, SciPy

---

**Status**: Active Development | **Version**: 1.0.0 | **Last Updated**: February 2026
