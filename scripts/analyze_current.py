"""Quick analysis for old-format images"""
import sys
sys.path.insert(0, '.')

from pathlib import Path
from pipeline import SingleDropletPipeline
from config import PipelineConfig
from datetime import datetime
import glob

# Your OneDrive folder
IMAGES_PATH = Path.home() / "OneDrive - KU Leuven" / "CameraCaptures"

# Get all images
image_files = sorted(glob.glob(str(IMAGES_PATH / "img_*.png")))

print(f"Found {len(image_files)} images")
print(f"First: {Path(image_files[0]).name}")
print(f"Last:  {Path(image_files[-1]).name}")

# Use first image as reference
reference_path = image_files[0]
capture_paths = image_files[1:]  # Rest are captures

# Parse timestamps from filenames
def parse_timestamp(filename):
    # img_20260216_104622.png -> extract time
    name = Path(filename).stem
    datestr = name.split('_')[1] + name.split('_')[2]  # 20260216104622
    dt = datetime.strptime(datestr, '%Y%m%d%H%M%S')
    return dt

times = [parse_timestamp(f) for f in image_files]
start_time = times[0]
timestamps = [(t - start_time).total_seconds() for t in times]

print(f"\nDuration: {timestamps[-1]/3600:.1f} hours")
print(f"Interval: ~{timestamps[1] - timestamps[0]:.0f} seconds\n")

# Configure detection
config = PipelineConfig()
config.detection.brightness_threshold = 45
config.detection.difference_threshold = 30
config.detection.min_crystal_area_px = 50
# Stricter detection to ignore tube edges
config.detection.brightness_threshold = 60    # Higher = less sensitive
config.detection.difference_threshold = 40    # Higher = ignore small changes
config.detection.min_crystal_area_px = 100    # Larger minimum size
config.channel.wall_margin_px = 30            # Crop more from edges
# Run analysis
print("Running analysis...")
pipeline = SingleDropletPipeline(config)
pipeline.process_timelapse(
    capture_paths,
    timestamps=timestamps[1:],
    reference_path=reference_path,
    verbose=True
)

# Export results
output_dir = "./current_experiment_results"
pipeline.export_results(output_dir)

print(f"\n✅ Done! Results in: {output_dir}")