"""
Quick Start Script — Get running in minutes
============================================

This script shows how to use the crystal_monitor pipeline
with YOUR actual microscope images. Just change the paths below!

Three usage examples:
  1. Analyze a single time-lapse folder
  2. Analyze a single image pair (reference + sample)
  3. Programmatic usage for integration with your gantry
"""

import os
import sys
import glob

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import PipelineConfig
from pipeline import SingleDropletPipeline
from multi_droplet import MultiDropletManager
from dashboard import CrystalDashboard
import cv2


# ============================================================
# EXAMPLE 1: Analyze a time-lapse folder
# ============================================================

def analyze_timelapse():
    """
    Point this at your folder of time-lapse images.

    Expected folder structure:
        my_images/
            blank.png          <-- clear droplet (reference)
            frame_001.png      <-- first time point
            frame_002.png
            ...
    """

    # ---- CHANGE THESE PATHS ----
    IMAGE_FOLDER = "./my_images"        # Your images folder
    REFERENCE_IMAGE = "./my_images/blank.png"  # Clear droplet
    OUTPUT_DIR = "./my_results"         # Where to save results
    TIME_INTERVAL = 60.0                # Seconds between frames
    # ----------------------------

    # Configure detection parameters for YOUR setup
    config = PipelineConfig()

    # Key parameters to tune for your microscope:
    config.detection.brightness_threshold = 45      # Increase if too many false positives
    config.detection.difference_threshold = 30      # Increase if noisy background
    config.detection.min_crystal_area_px = 50       # Minimum crystal size to detect
    config.detection.nucleation_min_area_px = 15    # Minimum nucleation spot size
    config.detection.nucleation_max_area_px = 300   # Max size to still call "nucleation"
    config.channel.vertical_channel = True          # Set False if channel is horizontal
    config.channel.wall_margin_px = 10              # Pixels to crop from channel walls
    config.multi_droplet.snapshot_interval_sec = TIME_INTERVAL

    # Run pipeline
    pipeline = SingleDropletPipeline(config)

    # Find images (sorted alphabetically — make sure naming gives correct order)
    image_paths = sorted(glob.glob(os.path.join(IMAGE_FOLDER, "*.png")))
    # Also check for .tif, .jpg
    for ext in ["*.tif", "*.tiff", "*.jpg", "*.jpeg", "*.bmp"]:
        image_paths.extend(sorted(glob.glob(os.path.join(IMAGE_FOLDER, ext))))
    image_paths = sorted(set(image_paths))

    # Remove reference from the list if it's in there
    ref_path = os.path.abspath(REFERENCE_IMAGE)
    image_paths = [p for p in image_paths if os.path.abspath(p) != ref_path]

    print(f"Found {len(image_paths)} images")
    print(f"Reference: {REFERENCE_IMAGE}")

    # Run!
    frame_df = pipeline.process_timelapse(
        image_paths,
        reference_path=REFERENCE_IMAGE,
        verbose=True
    )

    # Export everything
    pipeline.export_results(OUTPUT_DIR)

    # Also generate training data for future U-Net training
    pipeline.generate_training_data(os.path.join(OUTPUT_DIR, "training_data"))

    print(f"\nResults saved to {OUTPUT_DIR}")
    print(f"Open {OUTPUT_DIR}/figures/ to see growth curves and montages")


# ============================================================
# EXAMPLE 2: Analyze a single image
# ============================================================

def analyze_single_image():
    """
    Quick check: does my detection work on this image?
    """
    REFERENCE = "./blank.png"
    SAMPLE = "./crystal_sample.png"

    config = PipelineConfig()
    config.detection.brightness_threshold = 45

    pipeline = SingleDropletPipeline(config)

    ref = cv2.imread(REFERENCE)
    sample = cv2.imread(SAMPLE)

    if ref is not None:
        pipeline.set_reference(ref)

    detections = pipeline.process_frame(sample, timestamp=0.0)

    print(f"Found {len(detections)} objects:")
    for d in detections:
        label = "NUCLEATION" if d.is_nucleation else "CRYSTAL"
        print(f"  [{label}] Area={d.area_px:.0f}px², "
              f"Circularity={d.circularity:.2f}, "
              f"Aspect={d.aspect_ratio:.2f}")

    # Save annotated image
    from crystal_detector import CrystalDetector
    det = CrystalDetector(config)
    annotated = det.annotate_image(sample, detections)
    cv2.imwrite("annotated_output.png", annotated)
    print("\nSaved: annotated_output.png")


# ============================================================
# EXAMPLE 3: Gantry integration (130 droplets)
# ============================================================

def gantry_loop_example():
    """
    Example of how to integrate with your gantry system.

    Call manager.process_image() each time the gantry captures
    a new droplet image. At any point, you can query statistics.
    """

    config = PipelineConfig()
    config.multi_droplet.num_droplets = 130
    config.multi_droplet.grid_rows = 10
    config.multi_droplet.grid_cols = 13

    manager = MultiDropletManager(config)

    # --- Phase 1: Collect reference images (before crystallization) ---
    print("Phase 1: Collecting reference images...")
    for droplet_id in range(130):
        # ref_image = capture_image(droplet_id)  # <-- Your gantry function
        ref_image = None  # Placeholder
        if ref_image is not None:
            manager.set_reference(droplet_id, ref_image)

    # --- Phase 2: Monitor crystallization ---
    print("Phase 2: Monitoring crystallization...")
    experiment_start = 0  # time.time()

    # Simulated gantry loop:
    # for cycle in range(num_cycles):
    #     for droplet_id in range(130):
    #         image = capture_image(droplet_id)
    #         timestamp = time.time() - experiment_start
    #         detections = manager.process_image(droplet_id, image, timestamp)
    #
    #         # Real-time status
    #         if detections:
    #             print(f"Droplet {droplet_id}: {len(detections)} crystals detected")

    # --- Phase 3: Generate reports ---
    print("Phase 3: Generating reports...")
    manager.finalize_all()

    # Summary statistics
    summary = manager.summary_dataframe()
    summary.to_csv("multi_droplet_summary.csv", index=False)

    # Detailed track data
    tracks = manager.detailed_tracks_dataframe()
    tracks.to_csv("all_crystal_tracks.csv", index=False)

    # Heatmap visualization
    dashboard = CrystalDashboard(config, output_dir="./multi_droplet_figures")
    dashboard.plot_multi_droplet_heatmaps(manager)
    dashboard.plot_nucleation_statistics(manager)

    # Quick stats
    print(f"\nNucleation probability: {manager.nucleation_probability()*100:.1f}%")
    print(f"Crystal count map shape: {manager.crystal_count_map().shape}")


# ============================================================
# Parameter Tuning Guide
# ============================================================

def print_tuning_guide():
    """
    Print a guide on how to tune parameters for your specific setup.
    """
    guide = """
    ╔══════════════════════════════════════════════════════════════╗
    ║             PARAMETER TUNING GUIDE                          ║
    ╠══════════════════════════════════════════════════════════════╣
    ║                                                              ║
    ║  If you get TOO MANY false detections:                       ║
    ║    → Increase brightness_threshold (try 60, 80, 100)         ║
    ║    → Increase difference_threshold (try 40, 50)              ║
    ║    → Increase min_crystal_area_px (try 100, 200)             ║
    ║    → Increase morph_open_kernel (try 5, 7)                   ║
    ║                                                              ║
    ║  If you MISS crystals:                                       ║
    ║    → Decrease brightness_threshold (try 30, 20)              ║
    ║    → Decrease difference_threshold (try 20, 15)              ║
    ║    → Decrease min_crystal_area_px (try 20, 10)               ║
    ║                                                              ║
    ║  If NUCLEATION events are missed:                            ║
    ║    → Decrease nucleation_min_area_px (try 5, 3)              ║
    ║    → Increase nucleation_max_area_px (try 500)               ║
    ║    → Decrease brightness_threshold                           ║
    ║                                                              ║
    ║  If TRACKING is poor (IDs jumping):                          ║
    ║    → Increase max_linking_distance_px (try 80, 100)          ║
    ║    → Increase max_gap_frames (try 5, 10)                     ║
    ║                                                              ║
    ║  For the CHANNEL detection:                                  ║
    ║    → Set vertical_channel=True/False for your orientation    ║
    ║    → Increase wall_margin_px if detecting wall artifacts     ║
    ║                                                              ║
    ║  CALIBRATION TIP:                                            ║
    ║    Run analyze_single_image() on one frame first,            ║
    ║    tune parameters until detections look right,              ║
    ║    then run the full time-lapse.                             ║
    ║                                                              ║
    ╚══════════════════════════════════════════════════════════════╝
    """
    print(guide)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Crystal Monitor Quick Start")
    parser.add_argument('--mode', choices=['timelapse', 'single', 'gantry', 'guide'],
                       default='guide',
                       help='Which example to run')
    args = parser.parse_args()

    if args.mode == 'timelapse':
        analyze_timelapse()
    elif args.mode == 'single':
        analyze_single_image()
    elif args.mode == 'gantry':
        gantry_loop_example()
    elif args.mode == 'guide':
        print_tuning_guide()
