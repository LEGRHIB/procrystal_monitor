"""Manual droplet annotation tool - click to mark boundaries"""
import cv2
import json
from pathlib import Path

# Load your reference image
image_path = Path.home() / "OneDrive - KU Leuven" / "CameraCaptures" / "img_20260216_104622.png"
img = cv2.imread(str(image_path))

if img is None:
    print("ERROR: Could not load image")
    exit(1)

# Make a copy for drawing
display = img.copy()
points = []

def click_handler(event, x, y, flags, param):
    global points, display
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        cv2.circle(display, (x, y), 5, (0, 255, 0), -1)
        cv2.putText(display, str(len(points)), (x+10, y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        if len(points) == 4:
            # Draw the droplet box
            cv2.rectangle(display, points[0], points[2], (0, 255, 0), 2)
        
        cv2.imshow('Annotate Droplet', display)

print("=== Droplet Annotation Tool ===")
print("\nINSTRUCTIONS:")
print("1. Click TOP-LEFT corner of droplet")
print("2. Click TOP-RIGHT corner of droplet")
print("3. Click BOTTOM-RIGHT corner of droplet")
print("4. Click BOTTOM-LEFT corner of droplet")
print("5. Press 's' to SAVE")
print("6. Press 'r' to RESET and start over")
print("7. Press 'q' to QUIT without saving\n")

cv2.namedWindow('Annotate Droplet')
cv2.setMouseCallback('Annotate Droplet', click_handler)
cv2.imshow('Annotate Droplet', display)

while True:
    key = cv2.waitKey(1) & 0xFF
    
    if key == ord('s') and len(points) == 4:
        # Save coordinates
        droplet_roi = {
            'x': points[0][0],
            'y': points[0][1],
            'width': points[2][0] - points[0][0],
            'height': points[2][1] - points[0][1]
        }
        with open('droplet_roi.json', 'w') as f:
            json.dump(droplet_roi, f, indent=2)
        print(f"\n✅ Saved: {droplet_roi}")
        break
    
    elif key == ord('r'):
        points = []
        display = img.copy()
        cv2.imshow('Annotate Droplet', display)
        print("Reset - click 4 corners again")
    
    elif key == ord('q'):
        print("Cancelled")
        break

cv2.destroyAllWindows()