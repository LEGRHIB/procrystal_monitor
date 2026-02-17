"""Automatic droplet detection within the tube"""
import cv2
import numpy as np
from pathlib import Path
import json

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


# Test on your reference image
image_path = Path.home() / "OneDrive - KU Leuven" / "CameraCaptures" / "img_20260216_104622.png"
img = cv2.imread(str(image_path))

if img is None:
    print("ERROR: Could not load image")
    exit(1)

# First get the tube ROI (existing channel detection)
from preprocessing import ChannelDetector
from config import PipelineConfig

config = PipelineConfig()
channel_detector = ChannelDetector(config)
roi_bbox = channel_detector.detect_channel_roi(img)
x, y, w, h_tube = roi_bbox

# Extract tube region
tube_roi = img[y:y+h_tube, x:x+w]

# Detect droplet within tube
top, bottom = detect_droplet_boundaries(tube_roi)

print(f"Tube ROI: x={x}, y={y}, w={w}, h={h_tube}")
print(f"Droplet boundaries: top={top}, bottom={bottom}, height={bottom-top}")

# Save the droplet coordinates
droplet_roi = {
    'tube_x': int(x),
    'tube_y': int(y),
    'tube_w': int(w),
    'tube_h': int(h_tube),
    'droplet_top': int(top),
    'droplet_bottom': int(bottom),
    'droplet_y': int(y + top),
    'droplet_h': int(bottom - top)
}

with open('droplet_roi.json', 'w') as f:
    json.dump(droplet_roi, f, indent=2)

print(f"\n✅ Saved to droplet_roi.json")

# Visualize
vis = img.copy()
# Draw tube in blue
cv2.rectangle(vis, (x, y), (x+w, y+h_tube), (255, 0, 0), 2)
# Draw droplet in green
cv2.rectangle(vis, (x, y+top), (x+w, y+bottom), (0, 255, 0), 3)
cv2.putText(vis, "TUBE", (x+5, y+20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
cv2.putText(vis, "DROPLET", (x+5, y+top+25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

cv2.imwrite('droplet_detection.png', vis)
print("Saved visualization: droplet_detection.png")
print("\nOpen 'droplet_detection.png' to verify the detection!")