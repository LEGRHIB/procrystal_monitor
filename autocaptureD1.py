# 📸 Automatically capture images from a USB camera at regular intervals
import cv2
import time
import os
from datetime import datetime
from pathlib import Path

# 🔧 Settings
CAMERA_INDEX = 0          # try 0, 1, 2... to find your USB camera
INTERVAL_SECONDS = 1800   # time between pictures

# 🧪 Ask user for experiment reference
experiment_ref = input("Enter experiment reference: ").strip()
if not experiment_ref:
    print("❌ Experiment reference cannot be empty.")
    exit(1)

# ☁️ OneDrive folder - images will automatically sync to the cloud
EXPERIMENTS_BASE = Path("/Users/youcef/Library/CloudStorage/OneDrive-KULeuven/DATA/experiments")
OUTPUT_FOLDER = EXPERIMENTS_BASE / experiment_ref / "raw_images"

# 📁 Make output folder if it doesn't exist
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
print(f"📂 Saving images to: {OUTPUT_FOLDER}")

# 🎥 Open camera
cap = cv2.VideoCapture(CAMERA_INDEX)
if not cap.isOpened():
    print(f"❌ Could not open camera with index {CAMERA_INDEX}")
    exit(1)

print("✅ Camera opened. Press Ctrl+C in the terminal to stop.")

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("⚠️ Failed to grab frame")
            break

        # 🕒 Use timestamp in filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(OUTPUT_FOLDER, f"img_{timestamp}.png")

        # 💾 Save the image
        cv2.imwrite(filename, frame)
        print(f"📸 Saved {filename}")

        time.sleep(INTERVAL_SECONDS)

except KeyboardInterrupt:
    print("\n⏹ Stopped by user.")

finally:
    cap.release()
    print("🔓 Camera released.")
