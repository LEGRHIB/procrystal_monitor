"""Analyze current experiment using droplet ROI"""
import sys
sys.path.insert(0, '.')

from pathlib import Path
from pipeline import SingleDropletPipeline
from config import PipelineConfig
from datetime import datetime
import glob
import json
import cv2

# Load the droplet ROI
with open('droplet_roi.json') as f:
    droplet_roi = json.load(f)

print(f"Droplet ROI: {droplet_roi}")

# Your OneDrive folder
IMAGES_PATH = Path.home() / "OneDrive - KU Leuven" / "CameraCaptures"
image_files = sorted(glob.glob(str(IMAGES_PATH / "img_*.png")))

print(f"Found {len(image_files)} images")

# Parse timestamps
def parse_timestamp(filename):
    name = Path(filename).stem
    datestr = name.split('_')[1] + name.split('_')[2]
    dt = datetime.strptime(datestr, '%Y%m%d%H%M%S')
    return dt

times = [parse_timestamp(f) for f in image_files]
start_time = times[0]
timestamps = [(t - start_time).total_seconds() for t in times]

# Crop all images to droplet region and save temporarily
temp_dir = Path('./temp_droplet_crops')
temp_dir.mkdir(exist_ok=True)

print("\nCropping images to droplet region...")
cropped_paths = []
for i, img_path in enumerate(image_files):
    img = cv2.imread(img_path)
    # Crop to droplet
    cropped = img[
        droplet_roi['droplet_y']:droplet_roi['droplet_y']+droplet_roi['droplet_h'],
        droplet_roi['tube_x']:droplet_roi['tube_x']+droplet_roi['tube_w']
    ]
    crop_path = temp_dir / f"crop_{i:03d}.png"
    cv2.imwrite(str(crop_path), cropped)
    cropped_paths.append(str(crop_path))
    if i % 3 == 0:
        print(f"  {i+1}/{len(image_files)}")

# Configure
config = PipelineConfig()
config.detection.brightness_threshold = 60
config.detection.difference_threshold = 40
config.detection.min_crystal_area_px = 100
config.channel.wall_margin_px = 5  # Less margin needed since we cropped

# Run analysis on cropped images
print("\nRunning analysis on droplet region...")
pipeline = SingleDropletPipeline(config)
pipeline.process_timelapse(
    cropped_paths[1:],
    timestamps=timestamps[1:],
    reference_path=cropped_paths[0],
    verbose=True
)

# Export
output_dir = "./droplet_analysis_results"
pipeline.export_results(output_dir)

print(f"\n✅ Done! Results in: {output_dir}")
print("\nCheck the annotated images - false detections should be gone!")