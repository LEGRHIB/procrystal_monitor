# ProCrystal Monitor

Automated nucleation and crystal growth detection for biomolecule crystallization in microfluidic droplets, using polarized microscopy.

---

## Overview

ProCrystal Monitor is a Python toolkit for studying protein crystallization inside microfluidic capillary tubes. It has two main parts:

1. **Crystal Annotator** — a Flask web app for manually reviewing time-lapse experiments, measuring crystal dimensions, marking nucleation events, calibrating pixel scale, and exporting annotated datasets (COCO JSON + cropped images).
2. **Automated pipeline** — classical computer-vision modules that detect crystals by brightness / reference-subtraction, track them across frames, and export growth statistics. A U-Net architecture is included for training on the annotated data produced by the app.

The current workflow is: **annotate manually in the app ➜ export a dataset ➜ train the U-Net**.

---

## Repository Structure

```
procrystal_monitor/
│
├── annotator/                # Flask annotation web app
│   ├── app.py                #   Server (run this)
│   ├── run.sh                #   One-command launcher (creates venv, installs deps)
│   └── templates/
│       └── index.html        #   Single-page UI
│
├── config.py                 # Pipeline configuration dataclasses
├── preprocessing.py          # Channel ROI detection, denoising, CLAHE, reference subtraction
├── crystal_detector.py       # Brightness + difference-based crystal detection
├── growth_tracker.py         # Nearest-neighbour crystal tracking across frames
├── multi_droplet.py          # Multi-droplet (130-well) experiment manager
├── dashboard.py              # matplotlib visualisations (growth curves, heatmaps, montages)
├── deep_learning.py          # U-Net model, dataset loader, trainer (requires PyTorch)
├── pipeline.py               # Single-droplet orchestrator CLI
├── autocapture.py            # Timed image acquisition with dual local/OneDrive storage
├── analyze_experiment.py     # Batch analysis of captured experiments
├── quickstart.py             # Guided examples
├── train.py                  # Training entry point
├── test_pipeline.py          # Integration test with synthetic images
│
├── scripts/                  # Utility scripts
│   ├── analyze_current.py
│   ├── analyze_droplet_only.py
│   ├── annotate_droplet.py
│   └── detect_droplet.py
│
├── requirements.txt
└── README.md
```

---

## Installation

```bash
cd procrystal_monitor
pip install -r requirements.txt
```

**Core dependencies** (always required):

- `opencv-python`, `numpy`, `pandas`, `matplotlib`, `Pillow`

**Annotator app** (web UI):

- `flask`, `openpyxl`

**Deep learning** (optional — uncomment in requirements.txt):

- `torch`, `torchvision`

---

## Crystal Annotator (Web App)

### Quick start

```bash
# Option A — use the helper script (creates its own venv)
cd annotator
bash run.sh

# Option B — run directly (if deps are already installed)
cd annotator
python app.py
```

Open **http://localhost:5050** in your browser.

### Data root

The app scans for experiments under:

```
~/OneDrive - KU Leuven/DATA/experiments/<experiment_id>/raw_images/
```

Override with the `CRYSTAL_DATA_ROOT` environment variable:

```bash
export CRYSTAL_DATA_ROOT="/path/to/your/experiments"
```

Each experiment folder should contain a `raw_images/` directory with PNG files named `img_YYYYMMDD_HHMMSS.png`.

### What you can do in the app

| Feature | How |
|---|---|
| Browse frames | Arrow keys or Prev / Next buttons |
| Create / delete crystals | **New Crystal** button in sidebar |
| Mark nucleation | Select a crystal, press **N** on the frame where it first appears |
| Measure horizontal size | Select a crystal, press **H**, click two points |
| Measure vertical size | Select a crystal, press **V**, click two points |
| Calibrate scale | Press **T**, click across the tube width (known = 1.5 mm) |
| Measure droplet height | Press **D**, click top and bottom of droplet |
| Auto-crop to droplet | Press **A** |
| Temporal enhancement | Press **E** (subtracts reference frame to highlight changes) |
| Set reference frame | Press **R** |
| Export dataset | **Export Dataset** button → writes crops, enhanced images, COCO JSON |
| Export metrics Excel | **Excel** button → downloads growth metrics spreadsheet |

### Exported dataset layout

Clicking **Export Dataset** generates:

```
<experiment>/dataset/
├── crops/               # Raw droplet crops per frame
├── enhanced/            # Background-subtracted + contrast-stretched crops
├── coco_droplets.json   # Droplet bounding boxes (one per frame)
└── coco_crystals.json   # Crystal measurements from manual annotations
```

These files are the input for training.

---

## Automated Pipeline (Classical CV)

The automated pipeline can process a time-lapse without the web app.

### Single-droplet analysis

```bash
# From captured images
python pipeline.py --mode single --images ./images/ --reference blank.png --output ./results

# From an autocapture experiment folder
python analyze_experiment.py --experiment ~/CrystalMonitor/experiment_20260216_140000
```

### Image capture

```bash
# Single droplet, 30-min interval
python autocapture.py --mode single --interval 1800 --name my_experiment

# Multi-droplet with gantry (130 wells)
python autocapture.py --mode multi --num-droplets 130 --interval 1800
```

### Pipeline steps

1. **Preprocessing** (`preprocessing.py`) — Channel ROI detection, bilateral denoising, CLAHE contrast enhancement, reference subtraction.
2. **Detection** (`crystal_detector.py`) — Brightness thresholding (birefringence) + difference-from-reference, morphological cleanup, feature extraction.
3. **Tracking** (`growth_tracker.py`) — Nearest-neighbour linking across frames with gap handling and growth-rate fitting.
4. **Multi-droplet** (`multi_droplet.py`) — Per-droplet state tracking, cross-droplet statistics, nucleation probability.
5. **Visualisation** (`dashboard.py`) — Growth curves, time-lapse montages, multi-droplet heatmaps.

### Output files

| File | Contents |
|---|---|
| `frame_summary.csv` | Per-frame crystal counts and total area |
| `crystal_tracks.csv` | Full trajectory of every tracked crystal |
| `track_summary.csv` | Growth rate and duration per crystal |
| `figures/` | Growth curves, montages, heatmaps |
| `annotated/` | Images with detection overlays |

---

## Deep Learning (U-Net)

A U-Net segmentation model is included in `deep_learning.py`. It predicts three classes: background (0), nucleation (1), crystal (2).

### Training workflow

1. Annotate experiments in the web app.
2. Click **Export Dataset** to generate crops and COCO JSON.
3. Train the U-Net on the exported data.

```python
from deep_learning import UNet, CrystalDataset, UNetTrainer

model   = UNet(in_channels=1, num_classes=3)
train   = CrystalDataset("dataset/crops", "dataset/masks", image_size=(256, 256))
trainer = UNetTrainer(model, train)
trainer.train(epochs=50)
trainer.save_model("crystal_unet.pth")
```

Once a trained model exists at `models/crystal_unet.pth`, the pipeline loads it automatically and uses it instead of classical detection.

> **Note:** PyTorch is required. Install with `pip install torch torchvision`.

---

## Configuration

All detection parameters live in `config.py` as dataclasses. Key knobs:

```python
# Detection sensitivity
brightness_threshold = 45      # Higher → fewer false positives
difference_threshold = 30      # Higher → ignore small changes
min_crystal_area_px  = 50      # Minimum crystal size in pixels

# Tracking
max_linking_distance_px = 50   # Max movement between frames
max_gap_frames          = 3    # Frames a crystal can vanish before losing track
```

Save / load configs as JSON:

```python
config = PipelineConfig()
config.save("my_config.json")
config = PipelineConfig.load("my_config.json")
```

---

## Testing

```bash
python test_pipeline.py
```

Generates synthetic polarized-microscopy images and runs the full detection + tracking + export pipeline on them.

---

## contact

youcef.leghrib@kuleuven.be


