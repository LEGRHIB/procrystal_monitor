"""Automatic droplet detection within the tube"""
import cv2
import numpy as np
from pathlib import Path
import json
import argparse
import csv
import sys
import os

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

def detect_droplet_boundaries(image):
    """Find top and bottom meniscus lines of the droplet"""
    # Convert to grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    h, w = gray.shape
    
    # Horizontal intensity profile (average across width)
    row_profile = np.mean(gray, axis=1)
    
    # Simple smoothing with moving average
    kernel_size = 11
    kernel = np.ones(kernel_size) / kernel_size
    smoothed = np.convolve(row_profile, kernel, mode='same')
    
    # Find the droplet region (brighter than surroundings)
    threshold = np.mean(smoothed)
    bright_region = smoothed > threshold
    
    # Find first and last bright row (droplet boundaries)
    bright_rows = np.where(bright_region)[0]
    
    if len(bright_rows) < 2:
        # Fallback: use middle 70% of image
        margin = int(h * 0.15)
        return margin, h - margin
    
    top = bright_rows[0]
    bottom = bright_rows[-1]
    
    # Add small margin
    margin = 10
    top = max(0, top - margin)
    bottom = min(h, bottom + margin)
    
    return top, bottom


def process_image(image_path, config, channel_detector):
    """Process a single image and return droplet detection results"""
    img = cv2.imread(str(image_path))
    
    if img is None:
        return None, f"ERROR: Could not load image {image_path}"
    
    try:
        # Get the tube ROI
        roi_bbox = channel_detector.detect_channel_roi(img)
        x, y, w, h_tube = roi_bbox
        
        # Extract tube region
        tube_roi = img[y:y+h_tube, x:x+w]
        
        # Detect droplet within tube
        top, bottom = detect_droplet_boundaries(tube_roi)
        
        # Compile results
        results = {
            'image_name': image_path.name,
            'image_path': str(image_path),
            'tube_x': int(x),
            'tube_y': int(y),
            'tube_w': int(w),
            'tube_h': int(h_tube),
            'droplet_top': int(top),
            'droplet_bottom': int(bottom),
            'droplet_y': int(y + top),
            'droplet_h': int(bottom - top),
            'status': 'success'
        }
        
        return results, None
    except Exception as e:
        return None, f"ERROR processing {image_path.name}: {str(e)}"


def main():
    parser = argparse.ArgumentParser(description="Detect droplets in batch images")
    parser.add_argument('--images', type=str, required=True, help='Directory containing images to process')
    parser.add_argument('--output', type=str, required=True, help='Output directory for results')
    
    args = parser.parse_args()
    
    images_dir = Path(args.images).expanduser()
    output_dir = Path(args.output)
    
    if not images_dir.exists():
        print(f"ERROR: Images directory not found: {images_dir}")
        sys.exit(1)
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / 'visualizations').mkdir(exist_ok=True)
    
    # Initialize detection tools
    from preprocessing import ChannelDetector
    from config import PipelineConfig
    
    config = PipelineConfig()
    channel_detector = ChannelDetector(config)
    
    # Find all image files
    image_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']
    image_files = []
    for ext in image_extensions:
        image_files.extend(sorted(images_dir.glob(f'*{ext}')))
        image_files.extend(sorted(images_dir.glob(f'*{ext.upper()}')))
    
    image_files = sorted(set(image_files))  # Remove duplicates and sort
    
    if not image_files:
        print(f"ERROR: No images found in {images_dir}")
        sys.exit(1)
    
    print(f"Found {len(image_files)} images to process")
    
    # Process all images
    results_list = []
    errors = []
    
    for idx, image_path in enumerate(image_files, 1):
        print(f"[{idx}/{len(image_files)}] Processing {image_path.name}...", end='')
        
        results, error = process_image(image_path, config, channel_detector)
        
        if error:
            print(f" ❌ {error}")
            errors.append(error)
            continue
        
        results_list.append(results)
        
        # Create visualization
        img = cv2.imread(str(image_path))
        vis = img.copy()
        x, y, w, h_tube = results['tube_x'], results['tube_y'], results['tube_w'], results['tube_h']
        top, bottom = results['droplet_top'], results['droplet_bottom']
        
        # Draw tube in blue
        cv2.rectangle(vis, (x, y), (x+w, y+h_tube), (255, 0, 0), 2)
        # Draw droplet in green
        cv2.rectangle(vis, (x, y+top), (x+w, y+bottom), (0, 255, 0), 3)
        cv2.putText(vis, "TUBE", (x+5, y+20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        cv2.putText(vis, "DROPLET", (x+5, y+top+25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Save visualization
        vis_path = output_dir / 'visualizations' / f"{image_path.stem}_detected.png"
        cv2.imwrite(str(vis_path), vis)
        
        print(f" ✅ (droplet height: {results['droplet_h']} px)")
    
    # Save summary CSV
    if results_list:
        csv_path = output_dir / 'droplet_detections.csv'
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=results_list[0].keys())
            writer.writeheader()
            writer.writerows(results_list)
        print(f"\n✅ Results saved to {csv_path}")
    
    print(f"\n📊 Summary:")
    print(f"  ✅ Processed: {len(results_list)}/{len(image_files)}")
    print(f"  ❌ Failed: {len(errors)}")
    print(f"  📁 Output: {output_dir}")
    
    if errors:
        print(f"\nErrors encountered:")
        for error in errors:
            print(f"  - {error}")


if __name__ == '__main__':
    main()