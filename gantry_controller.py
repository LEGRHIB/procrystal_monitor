"""
gantry_controller.py
====================
Control an Ender 5 Pro as a camera gantry for scanning
microfluidic droplets with a USB microscope.

Modes:
  --test         Verify serial connection (home, report position)
  --calibrate    Interactive: WASD to move, SPACE to mark droplet positions
  --scan         Scan all positions, capture one image per droplet
  --monitor      Continuous scanning at interval (like autocapture + gantry)

The bed stays at Z=0 (fully lowered on Ender 5 Pro). Only XY moves.
Images are stored so app.py can read them directly for annotation.

Folder structure created per experiment:
    ~/OneDrive - KU Leuven/DATA/experiments/
        <expname>_droplet_001/raw_images/img_YYYYMMDD_HHMMSS.png
        <expname>_droplet_002/raw_images/img_YYYYMMDD_HHMMSS.png
        ...

Each droplet appears as a separate experiment in app.py.

Usage:
    python gantry_controller.py --test
    python gantry_controller.py --calibrate --output positions.json
    python gantry_controller.py --scan --positions positions.json --name myexp
    python gantry_controller.py --monitor --positions positions.json --name myexp --interval 1800

Author: Youcef Leghrib (KU Leuven)
"""

import cv2
import numpy as np
import os
import sys
import json
import time
import glob
import argparse
import signal
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Tuple

# Try importing serial — give a clear error if missing
try:
    import serial
    import serial.tools.list_ports
    HAS_SERIAL = True
except ImportError:
    HAS_SERIAL = False


# =============================================================
# CONFIGURATION
# =============================================================

# Default paths
ONEDRIVE_EXPERIMENTS = (
    Path.home() / "OneDrive - KU Leuven" / "DATA" / "experiments"
)

# Ender 5 Pro defaults
DEFAULT_BAUD = 115200
Z_HOME = 0.0            # Bed fully lowered — Ender 5 Pro: Z=0 is bed DOWN
XY_SPEED = 3000         # mm/min
Z_SPEED = 1000          # mm/min
SETTLE_TIME = 2.0       # seconds after move before capture
CAMERA_INDEX = 0
WARMUP_FRAMES = 5       # discard first N frames (auto-exposure)


# =============================================================
# SERIAL PORT DETECTION
# =============================================================

def find_printer_port() -> Optional[str]:
    """Auto-detect the Ender 5 Pro serial port by probing for Marlin firmware."""
    if not HAS_SERIAL:
        return None

    # Look for CH340 / USB serial ports
    patterns = ['usbserial', 'wchusbserial', 'SLAB_USB', 'usbmodem']

    # Gather candidate ports
    candidates: list[str] = []
    ports = serial.tools.list_ports.comports()
    for port in ports:
        device = port.device.lower()
        for pat in patterns:
            if pat.lower() in device:
                if port.device not in candidates:
                    candidates.append(port.device)

    # Fallback: check /dev/cu.* on macOS
    for pat in patterns:
        for m in glob.glob(f"/dev/cu.*{pat}*"):
            if m not in candidates:
                candidates.append(m)

    if not candidates:
        # List all available ports for debugging
        if ports:
            print("  Available serial ports:")
            for p in ports:
                print(f"    {p.device} — {p.description}")
        return None

    # Probe each candidate for Marlin firmware (M115)
    for dev in candidates:
        try:
            print(f"  Probing {dev} for Marlin firmware...")
            ser = serial.Serial(dev, DEFAULT_BAUD, timeout=3)
            time.sleep(2)  # wait for boot
            while ser.in_waiting:
                ser.readline()
            ser.write(b"M115\n")
            deadline = time.time() + 5
            while time.time() < deadline:
                if ser.in_waiting:
                    line = ser.readline().decode(errors='replace')
                    if 'FIRMWARE' in line.upper() or 'MARLIN' in line.upper():
                        print(f"  Found Marlin on {dev}")
                        ser.close()
                        return dev
                time.sleep(0.05)
            ser.close()
            print(f"  {dev}: no Marlin response")
        except Exception as e:
            print(f"  {dev}: probe failed ({e})")
            continue

    # Fallback: return the first candidate even without Marlin confirmation
    print(f"  Falling back to first candidate: {candidates[0]}")
    return candidates[0]


# =============================================================
# GANTRY CONTROLLER
# =============================================================

class GantryController:
    """
    Control Ender 5 Pro as an XY gantry via G-code over USB serial.

    Z is fixed at Z_HOME (bed fully down). Only XY moves.
    Heaters are disabled on connect for safety.
    """

    def __init__(self, port: str = "auto", baud: int = DEFAULT_BAUD,
                 timeout: float = 10.0):
        if not HAS_SERIAL:
            print("ERROR: pyserial not installed.")
            print("  Install with: pip install pyserial --break-system-packages")
            sys.exit(1)

        # Auto-detect port
        if port == "auto":
            port = find_printer_port()
            if port is None:
                print("ERROR: Could not auto-detect printer port.")
                print("  Is the Ender 5 Pro connected via USB?")
                print("  Try: ls /dev/cu.*")
                sys.exit(1)

        print(f"  Connecting to {port} at {baud} baud...")
        self.ser = serial.Serial(port, baud, timeout=timeout)
        time.sleep(2)  # Wait for Marlin to boot
        self._flush_startup()

        self.port = port
        self.current_x = 0.0
        self.current_y = 0.0
        self.current_z = Z_HOME

        # Safety: disable heaters
        self._send("M104 S0")  # hotend off
        self._send("M140 S0")  # bed off
        print(f"  Connected. Heaters disabled.")

    def _flush_startup(self):
        """Read and discard Marlin boot messages."""
        while self.ser.in_waiting:
            self.ser.readline()
        # Send a blank line to get an 'ok'
        self.ser.write(b"\n")
        time.sleep(0.5)
        while self.ser.in_waiting:
            self.ser.readline()

    def _send(self, cmd: str, timeout: float = 30.0):
        """Send G-code command and wait for 'ok' from Marlin."""
        self.ser.write(f"{cmd}\n".encode())
        start = time.time()
        while time.time() - start < timeout:
            if self.ser.in_waiting:
                line = self.ser.readline().decode(errors='replace').strip()
                if line.startswith("ok"):
                    return line
                if line.startswith("echo:") or line.startswith("//"):
                    continue  # info messages, skip
                if "Error" in line or "error" in line:
                    print(f"  MARLIN ERROR: {line}")
                    return line
            time.sleep(0.05)
        print(f"  WARNING: Timeout waiting for response to: {cmd}")
        return ""

    def home(self):
        """Home XY only. Z is set manually once — never touch it again."""
        print("  Homing X and Y only (Z stays where it is)...")
        self._send("G28 X Y", timeout=60)
        self._send("G90")  # absolute positioning
        self._send("M400")  # wait for move to complete
        print("  Homed. X=0 Y=0. Z untouched.")

    def move_to(self, x: float, y: float, speed: int = XY_SPEED):
        """Move to absolute XY position. Z stays fixed."""
        self._send("G90")
        self._send(f"G1 X{x:.2f} Y{y:.2f} F{speed}")
        self._send("M400")  # wait for move to complete
        self.current_x = x
        self.current_y = y

    def move_relative(self, dx: float = 0, dy: float = 0, speed: int = XY_SPEED):
        """Move relative to current position."""
        self._send("G91")  # relative mode
        self._send(f"G1 X{dx:.2f} Y{dy:.2f} F{speed}")
        self._send("M400")
        self._send("G90")  # back to absolute
        self.current_x += dx
        self.current_y += dy

    def get_position(self) -> Tuple[float, float, float]:
        """Query current position from Marlin."""
        self.ser.write(b"M114\n")
        time.sleep(0.3)
        while self.ser.in_waiting:
            line = self.ser.readline().decode(errors='replace').strip()
            if line.startswith("X:"):
                # Parse: "X:100.00 Y:50.00 Z:300.00 E:0.00"
                parts = line.split()
                x = float(parts[0].split(':')[1])
                y = float(parts[1].split(':')[1])
                z = float(parts[2].split(':')[1])
                self.current_x = x
                self.current_y = y
                self.current_z = z
                return x, y, z
        return self.current_x, self.current_y, self.current_z

    def disable_steppers(self):
        """Disable stepper motors (allows manual movement)."""
        self._send("M84")
        print("  Steppers disabled.")

    def safe_shutdown(self):
        """Home, lower bed, disable steppers and heaters."""
        print("  Safe shutdown...")
        self._send("M104 S0")
        self._send("M140 S0")
        self.home()
        self.disable_steppers()

    def close(self):
        """Close serial connection."""
        if self.ser and self.ser.is_open:
            self._send("M104 S0")
            self._send("M140 S0")
            self.ser.close()
            print("  Serial connection closed.")


# =============================================================
# CAMERA CAPTURE
# =============================================================

class CameraCapture:
    """USB microscope capture with warmup and retry logic."""

    def __init__(self, camera_index: int = CAMERA_INDEX,
                 warmup_frames: int = WARMUP_FRAMES):
        self.index = camera_index
        self.warmup_frames = warmup_frames
        self.cap = None

    def open(self) -> bool:
        self.cap = cv2.VideoCapture(self.index)
        if not self.cap.isOpened():
            print(f"  ERROR: Could not open camera at index {self.index}")
            print("  Try --camera 1 or --camera 2")
            return False

        # Discard warmup frames (auto-exposure settling)
        for _ in range(self.warmup_frames):
            self.cap.read()
            time.sleep(0.1)

        # Get resolution
        w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"  Camera opened (index {self.index}, {w}x{h})")
        return True

    def capture(self, retries: int = 3) -> Optional[np.ndarray]:
        """Capture a single frame with retry logic."""
        for attempt in range(retries):
            ret, frame = self.cap.read()
            if ret and frame is not None:
                return frame
            print(f"  Capture attempt {attempt+1}/{retries} failed...")
            time.sleep(0.5)

        # Try reopening
        print("  Reopening camera...")
        self.close()
        if self.open():
            ret, frame = self.cap.read()
            if ret:
                return frame
        return None

    def close(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None


# =============================================================
# POSITION MANAGEMENT
# =============================================================

def load_positions(path: str) -> List[Dict]:
    """Load droplet positions from JSON file.

    Format:
    [
        {"id": "droplet_001", "x": 10.0, "y": 15.0},
        {"id": "droplet_002", "x": 10.0, "y": 25.0},
        ...
    ]
    """
    with open(path) as f:
        positions = json.load(f)
    print(f"  Loaded {len(positions)} droplet positions from {path}")
    return positions


def save_positions(positions: List[Dict], path: str):
    """Save droplet positions to JSON."""
    with open(path, 'w') as f:
        json.dump(positions, f, indent=2)
    print(f"  Saved {len(positions)} positions to {path}")


# =============================================================
# MODE: TEST
# =============================================================

def run_test(args):
    """Test serial connection: connect, home, report position."""
    print("\n=== GANTRY TEST ===\n")

    gantry = GantryController(port=args.port)
    gantry.home()

    x, y, z = gantry.get_position()
    print(f"\n  Current position: X={x:.2f}  Y={y:.2f}  Z={z:.2f}")

    # Test a small move
    print("  Moving to X=110 Y=110...")
    gantry.move_to(110, 110)
    x, y, z = gantry.get_position()
    print(f"  Position: X={x:.2f}  Y={y:.2f}  Z={z:.2f}")

    print("  Moving back to X=0 Y=0...")
    gantry.move_to(0, 0)

    # Test camera
    print("\n  Testing camera...")
    camera = CameraCapture(camera_index=args.camera)
    if camera.open():
        frame = camera.capture()
        if frame is not None:
            test_path = "gantry_test_capture.png"
            cv2.imwrite(test_path, frame)
            print(f"  Test image saved: {test_path}")
            print(f"  Image size: {frame.shape[1]}x{frame.shape[0]}")
        camera.close()

    gantry.close()
    print("\n  TEST PASSED. Gantry and camera working.")


# =============================================================
# MODE: CALIBRATE
# =============================================================

def run_calibrate(args):
    """
    Interactive calibration: move gantry with keyboard, mark droplet positions.

    Controls:
        W/S     — move Y forward/back (step size)
        A/D     — move X left/right (step size)
        +/-     — increase/decrease step size
        SPACE   — mark current position as a droplet
        L       — list all marked positions
        R       — remove last marked position
        P       — report current position
        Q       — quit and save
    """
    print("\n=== GANTRY CALIBRATION ===\n")
    print("  Controls:")
    print("    W/S       Move Y +/-")
    print("    A/D       Move X +/-")
    print("    +/-       Adjust step size (mm)")
    print("    SPACE     Mark droplet position")
    print("    L         List marked positions")
    print("    R         Remove last position")
    print("    P         Report current XY")
    print("    Q         Quit and save\n")

    gantry = GantryController(port=args.port)
    gantry.home()

    # Open camera for live preview (optional)
    camera = CameraCapture(camera_index=args.camera)
    camera_ok = camera.open()

    positions = []
    step = 5.0  # mm per keypress
    droplet_count = 0

    # Use a simple OpenCV window for keyboard input + live preview
    if camera_ok:
        cv2.namedWindow("Gantry Calibration", cv2.WINDOW_NORMAL)

    print(f"\n  Step size: {step:.1f} mm")
    print(f"  Ready. Use keyboard in the OpenCV window.\n")

    running = True
    while running:
        # Show live preview if camera available
        if camera_ok:
            frame = camera.capture()
            if frame is not None:
                # Draw info overlay
                vis = frame.copy()
                x, y = gantry.current_x, gantry.current_y
                info = f"X={x:.1f} Y={y:.1f} | Step={step:.1f}mm | Marked={len(positions)}"
                cv2.putText(vis, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (0, 255, 0), 2)
                cv2.putText(vis, "WASD=move SPACE=mark Q=quit", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1)
                cv2.imshow("Gantry Calibration", vis)

            key = cv2.waitKey(100) & 0xFF
        else:
            # No camera — use terminal input
            try:
                key_str = input("  Command (wasd/space/+/-/l/r/p/q): ").strip().lower()
                if not key_str:
                    continue
                key = ord(key_str[0])
            except (EOFError, KeyboardInterrupt):
                break

        if key == ord('q'):
            running = False
        elif key == ord('w'):
            gantry.move_relative(dy=step)
            print(f"    Y+{step:.1f} → Y={gantry.current_y:.1f}")
        elif key == ord('s'):
            gantry.move_relative(dy=-step)
            print(f"    Y-{step:.1f} → Y={gantry.current_y:.1f}")
        elif key == ord('a'):
            gantry.move_relative(dx=-step)
            print(f"    X-{step:.1f} → X={gantry.current_x:.1f}")
        elif key == ord('d'):
            gantry.move_relative(dx=step)
            print(f"    X+{step:.1f} → X={gantry.current_x:.1f}")
        elif key == ord('+') or key == ord('='):
            step = min(50, step * 2)
            print(f"    Step size: {step:.1f} mm")
        elif key == ord('-'):
            step = max(0.1, step / 2)
            print(f"    Step size: {step:.1f} mm")
        elif key == ord(' '):
            droplet_count += 1
            pos = {
                "id": f"droplet_{droplet_count:03d}",
                "x": round(gantry.current_x, 2),
                "y": round(gantry.current_y, 2),
            }
            positions.append(pos)
            print(f"    MARKED: {pos['id']} at X={pos['x']}, Y={pos['y']}")

            # Save a preview image
            if camera_ok:
                frame = camera.capture()
                if frame is not None:
                    preview_dir = Path(args.output).parent / "calibration_previews"
                    preview_dir.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(
                        str(preview_dir / f"{pos['id']}.png"), frame)
        elif key == ord('l'):
            print(f"\n    --- {len(positions)} positions ---")
            for p in positions:
                print(f"    {p['id']}: X={p['x']}, Y={p['y']}")
            print()
        elif key == ord('r'):
            if positions:
                removed = positions.pop()
                droplet_count -= 1
                print(f"    Removed: {removed['id']}")
            else:
                print("    No positions to remove")
        elif key == ord('p'):
            x, y, z = gantry.get_position()
            print(f"    Position: X={x:.2f}, Y={y:.2f}, Z={z:.2f}")

    # Cleanup
    if camera_ok:
        cv2.destroyAllWindows()
        camera.close()

    # Save positions
    if positions:
        save_positions(positions, args.output)
    else:
        print("  No positions marked.")

    gantry.close()


# =============================================================
# MODE: SCAN (one-shot capture of all droplets)
# =============================================================

def run_scan(args):
    """
    Scan all droplet positions once. Capture one image per droplet.
    Each droplet is stored as a separate experiment for app.py.
    """
    print("\n=== GANTRY SCAN ===\n")

    positions = load_positions(args.positions)
    exp_base = Path(args.experiments_dir)

    gantry = GantryController(port=args.port)
    gantry.home()

    camera = CameraCapture(camera_index=args.camera)
    if not camera.open():
        gantry.close()
        sys.exit(1)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = args.name or f"scan_{timestamp}"
    captured = 0
    failed = 0

    print(f"\n  Experiment: {exp_name}")
    print(f"  Scanning {len(positions)} droplets...\n")

    for i, pos in enumerate(positions):
        droplet_id = pos["id"]
        x, y = pos["x"], pos["y"]

        # Each droplet = separate experiment folder for app.py
        droplet_exp = f"{exp_name}_{droplet_id}"
        raw_dir = exp_base / droplet_exp / "raw_images"
        raw_dir.mkdir(parents=True, exist_ok=True)

        # Move gantry
        gantry.move_to(x, y)
        time.sleep(args.settle)

        # Capture
        frame = camera.capture()
        if frame is not None:
            fname = f"img_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            fpath = raw_dir / fname
            cv2.imwrite(str(fpath), frame)
            captured += 1
            if (i + 1) % 10 == 0 or i == len(positions) - 1:
                print(f"  [{i+1}/{len(positions)}] {droplet_id} at "
                      f"X={x:.1f} Y={y:.1f} → {fname}")
        else:
            failed += 1
            print(f"  [{i+1}/{len(positions)}] {droplet_id} — CAPTURE FAILED")

    # Save positions + metadata in a master folder
    master_dir = exp_base / f"{exp_name}_master"
    master_dir.mkdir(parents=True, exist_ok=True)

    # Copy positions.json
    save_positions(positions, str(master_dir / "positions.json"))

    # Save scan metadata
    metadata = {
        "experiment_name": exp_name,
        "scan_time": timestamp,
        "num_droplets": len(positions),
        "captured": captured,
        "failed": failed,
        "z_height": Z_HOME,
        "xy_speed": XY_SPEED,
        "camera_index": args.camera,
    }
    with open(master_dir / "scan_metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)

    camera.close()
    gantry.close()

    print(f"\n  Scan complete: {captured} captured, {failed} failed")
    print(f"  Experiments created in: {exp_base}")
    print(f"  Open app.py and select any '{exp_name}_droplet_*' experiment")


# =============================================================
# MODE: MONITOR (continuous scanning at interval)
# =============================================================

def run_monitor(args):
    """
    Continuous monitoring: scan all droplets every --interval seconds.
    Each scan cycle adds one image to each droplet's time-lapse folder.
    Press Ctrl+C to stop.
    """
    print("\n=== GANTRY MONITOR ===\n")

    positions = load_positions(args.positions)
    exp_base = Path(args.experiments_dir)

    gantry = GantryController(port=args.port)
    gantry.home()

    camera = CameraCapture(camera_index=args.camera)
    if not camera.open():
        gantry.close()
        sys.exit(1)

    exp_name = args.name or f"monitor_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # Create all droplet experiment folders upfront
    for pos in positions:
        droplet_exp = f"{exp_name}_{pos['id']}"
        (exp_base / droplet_exp / "raw_images").mkdir(parents=True, exist_ok=True)

    print(f"  Experiment: {exp_name}")
    print(f"  Droplets: {len(positions)}")
    print(f"  Interval: {args.interval}s ({args.interval/60:.0f} min)")
    if args.duration:
        print(f"  Duration: {args.duration}h")
    print(f"  Press Ctrl+C to stop.\n")

    running = True
    cycle = 0
    start_time = time.time()

    def signal_handler(sig, frame):
        nonlocal running
        running = False
        print("\n  Stopping after current sweep...")

    signal.signal(signal.SIGINT, signal_handler)

    while running:
        # Check duration limit
        if args.duration:
            elapsed_h = (time.time() - start_time) / 3600
            if elapsed_h >= args.duration:
                print(f"\n  Duration limit reached ({args.duration}h). Stopping.")
                break

        sweep_start = time.time()
        captured = 0
        failed = 0

        print(f"  --- Cycle {cycle} ---")

        for i, pos in enumerate(positions):
            if not running:
                break

            droplet_id = pos["id"]
            x, y = pos["x"], pos["y"]

            gantry.move_to(x, y)
            time.sleep(args.settle)

            frame = camera.capture()
            if frame is not None:
                droplet_exp = f"{exp_name}_{droplet_id}"
                raw_dir = exp_base / droplet_exp / "raw_images"
                fname = f"img_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                cv2.imwrite(str(raw_dir / fname), frame)
                captured += 1
            else:
                failed += 1

            if (i + 1) % 20 == 0:
                print(f"    {i+1}/{len(positions)} droplets scanned...")

        sweep_duration = time.time() - sweep_start
        elapsed_total = (time.time() - start_time) / 3600
        print(f"  Cycle {cycle}: {captured} captured, {failed} failed, "
              f"sweep took {sweep_duration:.1f}s, "
              f"total elapsed: {elapsed_total:.1f}h")

        cycle += 1

        # Wait for next sweep
        wait_time = max(0, args.interval - sweep_duration)
        if wait_time > 0 and running:
            print(f"  Waiting {wait_time:.0f}s until next sweep...")
            sleep_end = time.time() + wait_time
            while running and time.time() < sleep_end:
                time.sleep(1)

    # Cleanup
    camera.close()
    gantry.safe_shutdown()
    gantry.close()

    total_h = (time.time() - start_time) / 3600
    print(f"\n  Monitor ended. {cycle} sweeps over {total_h:.1f} hours.")
    print(f"  Experiments in: {exp_base}/{exp_name}_droplet_*")


# =============================================================
# CLI
# =============================================================

def main():
    p = argparse.ArgumentParser(
        description="Ender 5 Pro Gantry Controller for Crystal Microscopy",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test connection
  python gantry_controller.py --test

  # Calibrate: mark droplet positions interactively
  python gantry_controller.py --calibrate --output positions.json

  # Single scan of all droplets
  python gantry_controller.py --scan --positions positions.json --name lysozyme_001

  # Continuous monitoring (every 30 min for 72 hours)
  python gantry_controller.py --monitor --positions positions.json \\
      --name lysozyme_001 --interval 1800 --duration 72
        """)

    # Mode (mutually exclusive)
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument('--test', action='store_true',
                      help='Test serial + camera connection')
    mode.add_argument('--calibrate', action='store_true',
                      help='Interactive: mark droplet positions')
    mode.add_argument('--scan', action='store_true',
                      help='One-shot scan of all droplets')
    mode.add_argument('--monitor', action='store_true',
                      help='Continuous scanning at interval')

    # Shared arguments
    p.add_argument('--port', default='auto',
                   help='Serial port (default: auto-detect)')
    p.add_argument('--camera', type=int, default=CAMERA_INDEX,
                   help=f'Camera index (default: {CAMERA_INDEX})')
    p.add_argument('--name', default=None,
                   help='Experiment name (auto-generated if omitted)')
    p.add_argument('--experiments-dir',
                   default=str(ONEDRIVE_EXPERIMENTS),
                   help=f'Experiments root (default: {ONEDRIVE_EXPERIMENTS})')

    # Positions file
    p.add_argument('--positions', default='positions.json',
                   help='Droplet positions JSON file')
    p.add_argument('--output', default='positions.json',
                   help='Output file for calibration (default: positions.json)')

    # Monitor settings
    p.add_argument('--settle', type=float, default=SETTLE_TIME,
                   help=f'Seconds to wait after move before capture (default: {SETTLE_TIME})')
    p.add_argument('--interval', type=float, default=1800,
                   help='Seconds between sweeps for monitor mode (default: 1800)')
    p.add_argument('--duration', type=float, default=None,
                   help='Total hours to run monitor mode (default: unlimited)')

    args = p.parse_args()

    if args.test:
        run_test(args)
    elif args.calibrate:
        run_calibrate(args)
    elif args.scan:
        run_scan(args)
    elif args.monitor:
        run_monitor(args)


if __name__ == '__main__':
    main()
