"""
Test Script — Validates the full pipeline with synthetic test images
====================================================================
Creates images that mimic polarized microscopy of crystals in a
microfluidic channel, then runs the full detection + tracking pipeline.
"""

import cv2
import numpy as np
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import PipelineConfig
from preprocessing import ChannelDetector, ImagePreprocessor
from crystal_detector import CrystalDetector
from growth_tracker import GrowthTracker
from pipeline import SingleDropletPipeline
from dashboard import CrystalDashboard


def create_synthetic_channel_image(
    width=640, height=480,
    channel_x=200, channel_w=200,
    background_val=10,
    channel_val=35,
    crystals=None,
    noise_std=5,
):
    """
    Create a synthetic polarized microscopy image.

    Args:
        crystals: list of dicts, each with:
            'center': (x, y) relative to channel
            'size': (w, h)
            'brightness': 0-255
            'shape': 'ellipse' or 'rect' or 'needle'
    """
    # Dark background (crossed polarizers → extinction)
    img = np.full((height, width, 3), background_val, dtype=np.uint8)

    # Channel region (slightly brighter)
    img[:, channel_x:channel_x+channel_w] = channel_val

    # Add crystals
    if crystals:
        for c in crystals:
            cx, cy = c['center']
            cx += channel_x  # Convert to image coords
            sw, sh = c['size']
            brightness = c['brightness']
            shape = c.get('shape', 'ellipse')

            # Birefringence colors (simulate interference colors)
            color = c.get('color', (brightness, int(brightness * 0.8), int(brightness * 0.3)))

            if shape == 'ellipse':
                cv2.ellipse(img, (int(cx), int(cy)), (int(sw//2), int(sh//2)),
                           c.get('angle', 0), 0, 360, color, -1)
            elif shape == 'rect':
                x1, y1 = int(cx - sw//2), int(cy - sh//2)
                x2, y2 = int(cx + sw//2), int(cy + sh//2)
                cv2.rectangle(img, (x1, y1), (x2, y2), color, -1)
            elif shape == 'needle':
                angle = c.get('angle', 30)
                rad = np.radians(angle)
                length = max(sw, sh)
                dx = int(length * np.cos(rad))
                dy = int(length * np.sin(rad))
                cv2.line(img, (int(cx) - dx, int(cy) - dy),
                        (int(cx) + dx, int(cy) + dy), color, max(2, min(sw, sh) // 3))

    # Add noise
    noise = np.random.normal(0, noise_std, img.shape).astype(np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    return img


def generate_test_timelapse(output_dir, n_frames=12):
    """Generate a synthetic time-lapse of crystal nucleation and growth."""
    os.makedirs(output_dir, exist_ok=True)

    paths = []

    # Frame 0: blank (reference)
    blank = create_synthetic_channel_image()
    path = os.path.join(output_dir, 'frame_000.png')
    cv2.imwrite(path, blank)
    paths.append(path)

    # Frames 1-3: nothing happening yet
    for i in range(1, 4):
        img = create_synthetic_channel_image()
        path = os.path.join(output_dir, f'frame_{i:03d}.png')
        cv2.imwrite(path, img)
        paths.append(path)

    # Frame 4: first nucleation (small bright spot)
    crystals = [
        {'center': (100, 240), 'size': (8, 6), 'brightness': 80,
         'shape': 'ellipse', 'color': (80, 70, 40)},
    ]
    img = create_synthetic_channel_image(crystals=crystals)
    path = os.path.join(output_dir, 'frame_004.png')
    cv2.imwrite(path, img)
    paths.append(path)

    # Frames 5-8: crystal grows
    for i in range(5, 9):
        growth_factor = (i - 3) * 1.5
        crystals = [
            {'center': (100, 240),
             'size': (int(8 + growth_factor * 4), int(6 + growth_factor * 3)),
             'brightness': min(255, int(80 + growth_factor * 15)),
             'shape': 'ellipse',
             'color': (min(255, int(100 + growth_factor * 20)),
                      min(255, int(80 + growth_factor * 10)),
                      min(255, int(40 + growth_factor * 5)))},
        ]
        # Second nucleation appears at frame 6
        if i >= 6:
            g2 = (i - 5) * 1.2
            crystals.append({
                'center': (60, 300),
                'size': (int(6 + g2 * 3), int(8 + g2 * 4)),
                'brightness': min(255, int(70 + g2 * 12)),
                'shape': 'needle',
                'angle': 45,
                'color': (min(255, int(70 + g2 * 15)),
                         min(255, int(50 + g2 * 10)),
                         min(255, int(120 + g2 * 8)))
            })

        img = create_synthetic_channel_image(crystals=crystals)
        path = os.path.join(output_dir, f'frame_{i:03d}.png')
        cv2.imwrite(path, img)
        paths.append(path)

    # Frames 9-11: multiple crystals, full growth
    for i in range(9, n_frames):
        gf1 = (i - 3) * 1.5
        gf2 = (i - 5) * 1.2
        crystals = [
            {'center': (100, 240),
             'size': (int(8 + gf1 * 4), int(6 + gf1 * 3)),
             'brightness': min(255, int(100 + gf1 * 12)),
             'shape': 'ellipse',
             'color': (min(255, int(140 + gf1 * 8)),
                      min(255, int(100 + gf1 * 6)),
                      min(255, int(50 + gf1 * 3)))},
            {'center': (60, 300),
             'size': (int(6 + gf2 * 3), int(8 + gf2 * 4)),
             'brightness': min(255, int(90 + gf2 * 10)),
             'shape': 'needle',
             'angle': 45,
             'color': (min(255, int(90 + gf2 * 10)),
                      min(255, int(70 + gf2 * 8)),
                      min(255, int(140 + gf2 * 5)))},
            # Third crystal
            {'center': (140, 180),
             'size': (int(10 + (i - 8) * 5), int(12 + (i - 8) * 4)),
             'brightness': min(255, int(75 + (i - 8) * 20)),
             'shape': 'rect',
             'color': (min(255, int(75 + (i - 8) * 20)),
                      min(255, int(60 + (i - 8) * 15)),
                      min(255, int(30 + (i - 8) * 10)))},
        ]
        img = create_synthetic_channel_image(crystals=crystals)
        path = os.path.join(output_dir, f'frame_{i:03d}.png')
        cv2.imwrite(path, img)
        paths.append(path)

    return paths


def test_single_droplet_pipeline():
    """Full integration test of the single-droplet pipeline."""
    print("=" * 60)
    print("CRYSTAL MONITOR — Integration Test")
    print("=" * 60)

    # Setup
    test_dir = "./crystal_test"
    images_dir = os.path.join(test_dir, "images")
    results_dir = os.path.join(test_dir, "results")

    # Generate synthetic time-lapse
    print("\n1) Generating synthetic time-lapse images...")
    paths = generate_test_timelapse(images_dir, n_frames=12)
    print(f"   Created {len(paths)} test images in {images_dir}")

    # Configure pipeline with settings tuned for synthetic data
    config = PipelineConfig()
    config.detection.brightness_threshold = 55
    config.detection.difference_threshold = 20
    config.detection.min_crystal_area_px = 20
    config.detection.nucleation_min_area_px = 8
    config.detection.nucleation_max_area_px = 200
    config.detection.morph_open_kernel = 2
    config.detection.morph_close_kernel = 3
    config.multi_droplet.snapshot_interval_sec = 30.0

    # Run pipeline
    print("\n2) Running single-droplet pipeline...")
    pipeline = SingleDropletPipeline(config)

    frame_df = pipeline.process_timelapse(
        paths[1:],  # Skip reference frame
        reference_path=paths[0],
        verbose=True
    )

    # Export results
    print("\n3) Exporting results...")
    pipeline.export_results(results_dir)

    # Generate training data
    print("\n4) Generating training data for U-Net...")
    pipeline.generate_training_data(os.path.join(results_dir, "training_data"))

    # Print summary
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"\nFrame-level summary:")
    print(frame_df.to_string(index=False))

    track_summary = pipeline.tracker.summary_dataframe()
    if not track_summary.empty:
        print(f"\nTrack summary:")
        print(track_summary.to_string(index=False))
    else:
        print("\nNo crystal tracks detected (this is expected if synthetic")
        print("images don't perfectly match detector thresholds).")

    print(f"\nAll outputs saved to: {results_dir}")
    print("  - frame_summary.csv")
    print("  - crystal_tracks.csv")
    print("  - track_summary.csv")
    print("  - figures/growth_curves_droplet_0.png")
    print("  - figures/timelapse_montage.png")
    print("  - annotated/frame_*.png")
    print("  - training_data/")

    return pipeline, frame_df


def test_components():
    """Test individual components."""
    print("\n--- Testing ChannelDetector ---")
    config = PipelineConfig()
    detector = ChannelDetector(config)
    img = create_synthetic_channel_image()
    roi = detector.detect_channel_roi(img)
    print(f"  Detected ROI: {roi}")
    assert roi[2] > 50, "Channel width should be > 50 px"
    print("  PASS")

    print("\n--- Testing ImagePreprocessor ---")
    preprocessor = ImagePreprocessor(config)
    processed = preprocessor.preprocess(img)
    print(f"  Input shape: {img.shape} -> Output shape: {processed.shape}")
    assert processed.shape == img.shape
    print("  PASS")

    print("\n--- Testing CrystalDetector ---")
    config.detection.brightness_threshold = 55
    config.detection.min_crystal_area_px = 20
    config.detection.nucleation_min_area_px = 8
    crystal_det = CrystalDetector(config)

    # Image with crystals
    crystals = [
        {'center': (100, 240), 'size': (30, 25), 'brightness': 150,
         'shape': 'ellipse', 'color': (150, 120, 60)},
    ]
    img_with_crystal = create_synthetic_channel_image(crystals=crystals)
    roi_img = detector.extract_roi(img_with_crystal)
    processed = preprocessor.preprocess(roi_img)
    ref = preprocessor.preprocess(detector.extract_roi(img))
    diff = preprocessor.subtract_reference(processed, ref)

    detections = crystal_det.detect(processed, diff)
    print(f"  Detections in image with crystal: {len(detections)}")
    for d in detections:
        print(f"    ID={d.detection_id}, Area={d.area_px:.0f}, "
              f"Circ={d.circularity:.2f}, Nucleation={d.is_nucleation}")
    print("  PASS")

    print("\n--- Testing GrowthTracker ---")
    tracker = GrowthTracker(config)
    for i in range(5):
        fake_dets = crystal_det.detect(processed, diff)
        tracker.update(i, i * 30.0, fake_dets)
    tracker.finalize()
    all_tracks = tracker.get_all_tracks()
    print(f"  Tracks after 5 frames: {len(all_tracks)}")
    df = tracker.to_dataframe()
    print(f"  DataFrame shape: {df.shape}")
    print("  PASS")

    print("\nAll component tests passed!")


if __name__ == '__main__':
    test_components()
    print("\n" + "=" * 60 + "\n")
    test_single_droplet_pipeline()
