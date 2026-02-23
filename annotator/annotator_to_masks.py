#!/usr/bin/env python3
"""
annotator_to_masks.py
─────────────────────
Converts Crystal Annotator manual annotations into PNG segmentation masks
for training the U-Net in procrystal_monitor/deep_learning.py.

Prerequisites:
  1. Run "Export Dataset" in Crystal Annotator for the experiment first.
     This creates  <exp>/dataset/crops/  and  <exp>/dataset/coco_droplets.json.
  2. Make sure you have measured crystals (H/V lines) and marked nucleation
     frames (N key) in Crystal Annotator.

Usage:
  python annotator_to_masks.py <experiment_path>

  # example:
  python annotator_to_masks.py \
    ~/OneDrive\ -\ KU\ Leuven/DATA/experiments/exp001

Output:
  <experiment>/dataset/masks/   ← one mask PNG per annotated frame

Mask convention (matches deep_learning.CrystalDataset thresholds):
  0   = background
  128 = crystal at its nucleation frame  (rare class — up-weighted in loss)
  255 = grown crystal body

How crystal position is estimated:
  Crystal Annotator stores manual line measurements (H and V through the
  crystal centre).  From these two lines the centre and ellipse axes are
  computed:
    - centre x  = midpoint of the H measurement (horizontal line)
    - centre y  = midpoint of the V measurement (vertical line)
    - semi-axis rx = h_px / 2
    - semi-axis ry = v_px / 2
  If only one direction is measured, the missing axis is set equal to the
  known one (square approximation).  A minimum radius of 5 px is enforced.
"""

import json
import sys
import cv2
import numpy as np
from collections import defaultdict
from pathlib import Path


def build_masks(exp_path: Path):
    exp_path    = Path(exp_path).expanduser()
    ann_dir     = exp_path / "annotations"
    dataset_dir = exp_path / "dataset"
    crops_dir   = dataset_dir / "crops"
    masks_dir   = dataset_dir / "masks"

    # ── sanity checks ─────────────────────────────────────────────────────────
    if not exp_path.is_dir():
        print(f"ERROR: experiment folder not found: {exp_path}")
        sys.exit(1)

    lpath = ann_dir / "lengths.jsonl"
    if not lpath.exists():
        print(f"ERROR: no lengths.jsonl in {ann_dir}  — annotate the experiment first.")
        sys.exit(1)

    dpath = dataset_dir / "coco_droplets.json"
    if not dpath.exists():
        print(f"ERROR: no coco_droplets.json in {dataset_dir}  — run Export Dataset first.")
        sys.exit(1)

    if not crops_dir.is_dir():
        print(f"ERROR: crops/ folder missing in {dataset_dir}  — run Export Dataset first.")
        sys.exit(1)

    masks_dir.mkdir(parents=True, exist_ok=True)

    # ── load annotations ──────────────────────────────────────────────────────
    measurements = []
    for line in lpath.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        m = json.loads(line)
        # skip calibration / droplet-height special entries
        if m.get("crystal_id", "").startswith("__"):
            continue
        measurements.append(m)

    # nucleation_map: crystal_id -> nucleation_frame_idx (or None)
    nucleation_map = {}
    cpath = ann_dir / "crystals.json"
    if cpath.exists():
        for c in json.loads(cpath.read_text()).get("crystals", []):
            nucleation_map[c["id"]] = c.get("nucleation_frame_idx")

    # ── load crop offsets from coco_droplets.json ─────────────────────────────
    # Each image entry has: id, file_name, width, height, crop_x, crop_y
    frame_crop_map = {}
    for entry in json.loads(dpath.read_text())["images"]:
        frame_crop_map[entry["id"]] = entry

    # ── group measurements by frame ───────────────────────────────────────────
    frame_meas: dict = defaultdict(list)
    for m in measurements:
        frame_meas[m["frame_idx"]].append(m)

    generated = 0
    skipped   = 0

    for frame_idx, meas_list in sorted(frame_meas.items()):
        crop_info = frame_crop_map.get(frame_idx)
        if crop_info is None:
            skipped += 1
            continue  # frame not exported → no crop image to pair with

        cw     = crop_info["width"]
        ch     = crop_info["height"]
        crop_x = crop_info.get("crop_x", 0)
        crop_y = crop_info.get("crop_y", 0)

        mask = np.zeros((ch, cw), dtype=np.uint8)

        # group by crystal within this frame
        crystal_meas: dict = defaultdict(list)
        for m in meas_list:
            crystal_meas[m["crystal_id"]].append(m)

        for crystal_id, c_meas in crystal_meas.items():
            h_m = next((m for m in c_meas if m.get("direction") == "h"), None)
            v_m = next((m for m in c_meas if m.get("direction") == "v"), None)

            # ── estimate centre (original image coords) and axes ──────────────
            if h_m and v_m:
                # centre = intersection of the two measurement lines
                cx_orig = (h_m["p1"]["x"] + h_m["p2"]["x"]) / 2.0
                cy_orig = (v_m["p1"]["y"] + v_m["p2"]["y"]) / 2.0
                rx = max(5, h_m["h_px"] // 2)
                ry = max(5, v_m["v_px"] // 2)
            elif h_m:
                cx_orig = (h_m["p1"]["x"] + h_m["p2"]["x"]) / 2.0
                cy_orig = (h_m["p1"]["y"] + h_m["p2"]["y"]) / 2.0
                rx = max(5, h_m["h_px"] // 2)
                ry = rx   # assume square
            elif v_m:
                cx_orig = (v_m["p1"]["x"] + v_m["p2"]["x"]) / 2.0
                cy_orig = (v_m["p1"]["y"] + v_m["p2"]["y"]) / 2.0
                ry = max(5, v_m["v_px"] // 2)
                rx = ry   # assume square
            else:
                continue

            # ── convert to crop-relative coordinates ──────────────────────────
            cx_crop = int(cx_orig - crop_x)
            cy_crop = int(cy_orig - crop_y)

            # skip if the crystal centre falls outside the crop
            if not (0 <= cx_crop < cw and 0 <= cy_crop < ch):
                continue

            # ── mask intensity: 128 = nucleation frame, 255 = grown crystal ──
            nuc_frame = nucleation_map.get(crystal_id)
            intensity = 128 if (nuc_frame is not None and nuc_frame == frame_idx) else 255

            cv2.ellipse(mask,
                        center=(cx_crop, cy_crop),
                        axes=(rx, ry),
                        angle=0, startAngle=0, endAngle=360,
                        color=intensity, thickness=-1)

        stem = crop_info["file_name"].replace(".png", "")
        out_path = masks_dir / f"{stem}.png"
        cv2.imwrite(str(out_path), mask)
        generated += 1

    print(f"\n  Experiment : {exp_path.name}")
    print(f"  Masks      : {generated} generated → {masks_dir}")
    if skipped:
        print(f"  Skipped    : {skipped} frames (not in Export Dataset)")
    print()
    print("  Next step:")
    print("    from deep_learning import CrystalDataset, UNetTrainer")
    print(f"    ds = CrystalDataset('{dataset_dir / 'crops'}',")
    print(f"                        '{masks_dir}')")
    print("    trainer = UNetTrainer(num_classes=3)")
    print("    trainer.train(ds, epochs=50)")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python annotator_to_masks.py <experiment_path>")
        print()
        print("  experiment_path — full path to the experiment folder, e.g.:")
        print("  ~/OneDrive - KU Leuven/DATA/experiments/exp001")
        sys.exit(1)

    build_masks(Path(sys.argv[1]))
