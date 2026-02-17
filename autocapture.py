"""
autocapture.py — Image Capture for Crystal Monitor Pipeline
============================================================

Two modes:
  1. SINGLE DROPLET  — Your current setup. One camera, one droplet, timed captures.
  2. MULTI DROPLET   — Gantry moves camera to 130 droplet positions in sequence.

Folder structure (designed to plug directly into the analysis pipeline):

    experiment_YYYYMMDD_HHMMSS/
    |
    |-- experiment.json            <-- metadata (start time, interval, camera, etc.)
    |-- config.json                <-- pipeline detection config (for reproducibility)
    |
    |-- references/                <-- clear/blank droplet images (captured once at start)
    |   |-- droplet_000.png
    |   |-- droplet_001.png
    |   |-- ...
    |
    |-- captures/                  <-- all raw time-lapse images
    |   |-- droplet_000/           <-- one subfolder per droplet
    |   |   |-- cycle_0000_t000000s.png
    |   |   |-- cycle_0001_t001800s.png
    |   |   |-- ...
    |   |-- droplet_001/
    |   |   |-- cycle_0000_t000005s.png
    |   |   |-- ...
    |   ...
    |
    |-- capture_log.csv            <-- machine-readable log of every capture

Naming convention:
    cycle_CCCC_tTTTTTTs.png
      CCCC = gantry cycle number (0000, 0001, ...) — for single droplet this is just frame number
      TTTTTT = elapsed seconds since experiment start (integer, zero-padded)

This naming guarantees:
  - Alphabetical sort = chronological order
  - The pipeline can parse droplet_id + timestamp directly from the path
  - Easy to correlate across droplets (same cycle number = same gantry sweep)

Usage:
    # Single droplet (your current setup)
    python autocapture.py --mode single --interval 1800

    # Multi-droplet with gantry
    python autocapture.py --mode multi --num-droplets 130 --interval 1800

    # Capture references only (run once before experiment)
    python autocapture.py --mode reference --num-droplets 130
"""

import cv2
import time
import os
import sys
import csv
import json
import shutil
import argparse
import signal
from datetime import datetime
from pathlib import Path
from typing import Optional


# =============================================================
# CONFIGURATION
# =============================================================

# Default paths
ONEDRIVE_BASE = Path.home() / "OneDrive - KU Leuven"
LOCAL_BASE = Path.home() / "CrystalMonitor"

# Camera
DEFAULT_CAMERA_INDEX = 0
DEFAULT_INTERVAL = 1800  # 30 minutes


# =============================================================
# EXPERIMENT SESSION
# =============================================================

class ExperimentSession:
    """
    Manages the folder structure, naming, logging, and dual storage
    for one crystallization experiment.
    """

    def __init__(self, experiment_name: Optional[str] = None,
                 local_base: Path = LOCAL_BASE,
                 onedrive_base: Path = ONEDRIVE_BASE,
                 num_droplets: int = 1,
                 interval_sec: float = 1800,
                 camera_index: int = 0):

        # Generate experiment name from timestamp if not provided
        if experiment_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            experiment_name = f"experiment_{timestamp}"

        self.experiment_name = experiment_name
        self.num_droplets = num_droplets
        self.interval_sec = interval_sec
        self.camera_index = camera_index
        self.start_time = time.time()
        self.cycle_count = 0

        # --- Local storage (fast, primary) ---
        self.local_root = local_base / experiment_name
        self.local_references = self.local_root / "references"
        self.local_captures = self.local_root / "captures"

        # --- OneDrive mirror (synced backup) ---
        self.onedrive_root = onedrive_base / "CrystalMonitor" / experiment_name
        self.onedrive_references = self.onedrive_root / "references"
        self.onedrive_captures = self.onedrive_root / "captures"

        # Create folder structure
        self._create_folders()

        # CSV log
        self.log_path = self.local_root / "capture_log.csv"
        self._init_log()

        # Save experiment metadata
        self._save_metadata()

        print(f"Experiment: {experiment_name}")
        print(f"  Local:    {self.local_root}")
        print(f"  OneDrive: {self.onedrive_root}")
        print(f"  Droplets: {num_droplets}")
        print(f"  Interval: {interval_sec}s ({interval_sec/60:.0f} min)")

    def _create_folders(self):
        """Create the full directory tree for both local and OneDrive."""
        for base_captures, base_refs in [
            (self.local_captures, self.local_references),
            (self.onedrive_captures, self.onedrive_references),
        ]:
            base_refs.mkdir(parents=True, exist_ok=True)
            for i in range(self.num_droplets):
                (base_captures / f"droplet_{i:03d}").mkdir(parents=True, exist_ok=True)

    def _init_log(self):
        """Initialize the CSV capture log."""
        if not self.log_path.exists():
            with open(self.log_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'timestamp_iso', 'elapsed_sec', 'cycle', 'droplet_id',
                    'filename', 'image_type', 'local_path', 'onedrive_path',
                    'frame_width', 'frame_height', 'capture_ok'
                ])

    def _save_metadata(self):
        """Save experiment metadata JSON."""
        metadata = {
            'experiment_name': self.experiment_name,
            'start_time_iso': datetime.now().isoformat(),
            'start_time_unix': self.start_time,
            'num_droplets': self.num_droplets,
            'interval_sec': self.interval_sec,
            'camera_index': self.camera_index,
            'local_root': str(self.local_root),
            'onedrive_root': str(self.onedrive_root),
            'folder_structure': {
                'references': 'references/droplet_NNN.png',
                'captures': 'captures/droplet_NNN/cycle_CCCC_tTTTTTTs.png',
            },
            'naming_convention': {
                'droplet_id': '3-digit zero-padded (000-129)',
                'cycle': '4-digit zero-padded gantry cycle / frame number',
                'timestamp': '6-digit zero-padded elapsed seconds',
            }
        }
        for root in [self.local_root, self.onedrive_root]:
            meta_path = root / "experiment.json"
            with open(meta_path, 'w') as f:
                json.dump(metadata, f, indent=2)

    def _log_capture(self, elapsed_sec: float, cycle: int, droplet_id: int,
                     filename: str, image_type: str, local_path: str,
                     onedrive_path: str, width: int, height: int, ok: bool):
        """Append one row to the capture log."""
        with open(self.log_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                datetime.now().isoformat(),
                f"{elapsed_sec:.1f}",
                cycle,
                droplet_id,
                filename,
                image_type,
                local_path,
                onedrive_path,
                width,
                height,
                ok
            ])

    def build_filename(self, cycle: int, elapsed_sec: float) -> str:
        """
        Build the standardized filename.
        Example: cycle_0003_t005400s.png
        """
        return f"cycle_{cycle:04d}_t{int(elapsed_sec):06d}s.png"

    def save_reference(self, droplet_id: int, frame: 'numpy.ndarray') -> bool:
        """
        Save a reference (clear/blank) image for a droplet.
        Saves to both local and OneDrive.
        """
        filename = f"droplet_{droplet_id:03d}.png"
        h, w = frame.shape[:2]

        # Local
        local_path = self.local_references / filename
        cv2.imwrite(str(local_path), frame)

        # OneDrive copy
        onedrive_path = self.onedrive_references / filename
        try:
            shutil.copy2(str(local_path), str(onedrive_path))
        except Exception as e:
            print(f"  OneDrive copy failed (will sync later): {e}")

        elapsed = time.time() - self.start_time
        self._log_capture(elapsed, 0, droplet_id, filename, 'reference',
                          str(local_path), str(onedrive_path), w, h, True)
        return True

    def save_capture(self, droplet_id: int, cycle: int,
                     frame: 'numpy.ndarray') -> bool:
        """
        Save a time-lapse capture image.
        Saves to both local and OneDrive.
        """
        elapsed = time.time() - self.start_time
        filename = self.build_filename(cycle, elapsed)
        h, w = frame.shape[:2]

        droplet_folder = f"droplet_{droplet_id:03d}"

        # Local
        local_path = self.local_captures / droplet_folder / filename
        ok = cv2.imwrite(str(local_path), frame)

        # OneDrive copy
        onedrive_path = self.onedrive_captures / droplet_folder / filename
        try:
            shutil.copy2(str(local_path), str(onedrive_path))
        except Exception as e:
            # OneDrive folder might not be mounted or sync might be slow
            # The file will sync automatically once OneDrive catches up
            print(f"  OneDrive copy note: {e}")

        self._log_capture(elapsed, cycle, droplet_id, filename, 'capture',
                          str(local_path), str(onedrive_path), w, h, ok)

        return ok

    def get_droplet_images(self, droplet_id: int) -> list:
        """
        Get sorted list of all capture paths for a droplet.
        (For feeding into the analysis pipeline.)
        """
        folder = self.local_captures / f"droplet_{droplet_id:03d}"
        return sorted(folder.glob("cycle_*.png"))

    def get_reference_path(self, droplet_id: int) -> Optional[Path]:
        """Get the reference image path for a droplet."""
        ref = self.local_references / f"droplet_{droplet_id:03d}.png"
        return ref if ref.exists() else None

    def copy_log_to_onedrive(self):
        """Copy the capture log to OneDrive."""
        try:
            onedrive_log = self.onedrive_root / "capture_log.csv"
            shutil.copy2(str(self.log_path), str(onedrive_log))
        except Exception:
            pass


# =============================================================
# CAMERA WRAPPER
# =============================================================

class Camera:
    """Thin wrapper around cv2.VideoCapture with retry logic."""

    def __init__(self, index: int = 0, warmup_frames: int = 5):
        self.index = index
        self.cap = None
        self.warmup_frames = warmup_frames

    def open(self) -> bool:
        self.cap = cv2.VideoCapture(self.index)
        if not self.cap.isOpened():
            print(f"Could not open camera at index {self.index}")
            return False

        # Discard warmup frames (auto-exposure settling)
        for _ in range(self.warmup_frames):
            self.cap.read()
            time.sleep(0.1)

        print(f"Camera opened (index {self.index})")
        return True

    def capture(self, retries: int = 3) -> Optional['numpy.ndarray']:
        """Capture a single frame with retry logic."""
        for attempt in range(retries):
            ret, frame = self.cap.read()
            if ret and frame is not None:
                return frame
            print(f"  Capture attempt {attempt+1}/{retries} failed, retrying...")
            time.sleep(0.5)

        # If all retries failed, try reopening the camera
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
            print("Camera released.")


# =============================================================
# CAPTURE MODES
# =============================================================

def run_single_droplet(args):
    """
    Single droplet mode — captures one droplet at regular intervals.
    This is equivalent to your original autocapture.py but with
    proper naming and dual storage.
    """
    session = ExperimentSession(
        experiment_name=args.name,
        local_base=Path(args.local_dir),
        onedrive_base=Path(args.onedrive_dir),
        num_droplets=1,
        interval_sec=args.interval,
        camera_index=args.camera,
    )

    camera = Camera(index=args.camera)
    if not camera.open():
        sys.exit(1)

    # --- Phase 1: Capture reference ---
    if not args.skip_reference:
        print("\n--- REFERENCE CAPTURE ---")
        input("  Load CLEAR solution, then press Enter to capture reference...")
        frame = camera.capture()
        if frame is not None:
            session.save_reference(0, frame)
            print(f"  Reference saved for droplet 0")
        else:
            print("  WARNING: Failed to capture reference")

    # --- Phase 2: Time-lapse ---
    print(f"\n--- TIME-LAPSE CAPTURE ---")
    print(f"  Interval: {args.interval}s ({args.interval/60:.0f} min)")
    print(f"  Press Ctrl+C to stop.\n")

    cycle = 0
    running = True

    def signal_handler(sig, frame_):
        nonlocal running
        running = False
        print("\n  Stopping...")

    signal.signal(signal.SIGINT, signal_handler)

    while running:
        frame = camera.capture()
        if frame is None:
            print(f"  Cycle {cycle}: FAILED to capture")
            time.sleep(10)
            continue

        elapsed = time.time() - session.start_time
        ok = session.save_capture(droplet_id=0, cycle=cycle, frame=frame)

        filename = session.build_filename(cycle, elapsed)
        print(f"  Cycle {cycle:4d} | t={elapsed:8.0f}s ({elapsed/3600:.1f}h) | {filename} | {'OK' if ok else 'FAIL'}")

        # Periodically sync the log
        if cycle % 10 == 0:
            session.copy_log_to_onedrive()

        cycle += 1
        session.cycle_count = cycle

        # Sleep in small intervals so Ctrl+C is responsive
        sleep_end = time.time() + args.interval
        while running and time.time() < sleep_end:
            time.sleep(1)

    camera.close()
    session.copy_log_to_onedrive()
    print(f"\nExperiment ended. {cycle} frames captured.")
    print(f"  Local:    {session.local_root}")
    print(f"  OneDrive: {session.onedrive_root}")


def run_multi_droplet(args):
    """
    Multi-droplet mode — gantry moves camera to each droplet position.

    For now this is a TEMPLATE. You'll integrate your gantry control
    code where indicated with # GANTRY: comments.
    """
    session = ExperimentSession(
        experiment_name=args.name,
        local_base=Path(args.local_dir),
        onedrive_base=Path(args.onedrive_dir),
        num_droplets=args.num_droplets,
        interval_sec=args.interval,
        camera_index=args.camera,
    )

    camera = Camera(index=args.camera)
    if not camera.open():
        sys.exit(1)

    # --- Phase 1: Capture references for all droplets ---
    if not args.skip_reference:
        print("\n--- REFERENCE CAPTURE (all droplets) ---")
        input(f"  Load CLEAR solution in all {args.num_droplets} droplets.\n"
              f"  Press Enter to start reference scan...")

        for droplet_id in range(args.num_droplets):
            # GANTRY: Move camera to droplet position
            # gantry.move_to_droplet(droplet_id)
            # time.sleep(0.5)  # Wait for settling

            frame = camera.capture()
            if frame is not None:
                session.save_reference(droplet_id, frame)
                if droplet_id % 10 == 0:
                    print(f"  Reference {droplet_id+1}/{args.num_droplets}")
            else:
                print(f"  WARNING: Failed reference for droplet {droplet_id}")

        print(f"  All {args.num_droplets} references captured.")

    # --- Phase 2: Timed gantry sweeps ---
    print(f"\n--- MULTI-DROPLET TIME-LAPSE ---")
    print(f"  Droplets: {args.num_droplets}")
    print(f"  Interval between full sweeps: {args.interval}s")
    print(f"  Press Ctrl+C to stop.\n")

    cycle = 0
    running = True

    def signal_handler(sig, frame_):
        nonlocal running
        running = False
        print("\n  Stopping after current sweep...")

    signal.signal(signal.SIGINT, signal_handler)

    while running:
        sweep_start = time.time()
        print(f"  --- Cycle {cycle} ---")

        for droplet_id in range(args.num_droplets):
            if not running:
                break

            # GANTRY: Move camera to droplet position
            # gantry.move_to_droplet(droplet_id)
            # time.sleep(0.3)  # settling time

            frame = camera.capture()
            if frame is not None:
                session.save_capture(droplet_id, cycle, frame)
            else:
                print(f"    Droplet {droplet_id}: capture FAILED")

            if droplet_id % 20 == 0:
                elapsed = time.time() - session.start_time
                print(f"    Droplet {droplet_id:3d}/{args.num_droplets} | "
                      f"t={elapsed:.0f}s")

        sweep_duration = time.time() - sweep_start
        print(f"  Sweep {cycle} done in {sweep_duration:.1f}s")

        session.copy_log_to_onedrive()
        cycle += 1
        session.cycle_count = cycle

        # Wait for next sweep interval
        wait_time = max(0, args.interval - sweep_duration)
        if wait_time > 0:
            print(f"  Waiting {wait_time:.0f}s until next sweep...")
            sleep_end = time.time() + wait_time
            while running and time.time() < sleep_end:
                time.sleep(1)

    camera.close()
    session.copy_log_to_onedrive()
    print(f"\nExperiment ended. {cycle} sweeps completed.")
    print(f"  Local:    {session.local_root}")
    print(f"  OneDrive: {session.onedrive_root}")


def run_reference_only(args):
    """Capture reference images only (for all droplets)."""
    session = ExperimentSession(
        experiment_name=args.name or f"references_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        local_base=Path(args.local_dir),
        onedrive_base=Path(args.onedrive_dir),
        num_droplets=args.num_droplets,
        interval_sec=0,
        camera_index=args.camera,
    )

    camera = Camera(index=args.camera)
    if not camera.open():
        sys.exit(1)

    print(f"\n--- Capturing references for {args.num_droplets} droplets ---")
    input("  Load clear solution, then press Enter...")

    for droplet_id in range(args.num_droplets):
        # GANTRY: Move to position
        # gantry.move_to_droplet(droplet_id)
        # time.sleep(0.3)

        frame = camera.capture()
        if frame is not None:
            session.save_reference(droplet_id, frame)
        if droplet_id % 10 == 0:
            print(f"  {droplet_id+1}/{args.num_droplets}")

    camera.close()
    print(f"\nDone. References saved to:")
    print(f"  {session.local_references}")


# =============================================================
# CLI
# =============================================================

def main():
    parser = argparse.ArgumentParser(
        description='Crystal Monitor — Image Capture',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single droplet, 30-min interval
  python autocapture.py --mode single --interval 1800

  # Multi-droplet with gantry, 130 wells, 30-min sweeps
  python autocapture.py --mode multi --num-droplets 130 --interval 1800

  # Just capture references
  python autocapture.py --mode reference --num-droplets 130

  # Custom experiment name + camera index
  python autocapture.py --mode single --name lysozyme_run_01 --camera 1
        """
    )
    parser.add_argument('--mode', choices=['single', 'multi', 'reference'],
                        default='single', help='Capture mode')
    parser.add_argument('--name', default=None,
                        help='Experiment name (auto-generated if omitted)')
    parser.add_argument('--interval', type=float, default=DEFAULT_INTERVAL,
                        help=f'Seconds between captures (default: {DEFAULT_INTERVAL})')
    parser.add_argument('--camera', type=int, default=DEFAULT_CAMERA_INDEX,
                        help=f'Camera index (default: {DEFAULT_CAMERA_INDEX})')
    parser.add_argument('--num-droplets', type=int, default=130,
                        help='Number of droplets for multi mode (default: 130)')
    parser.add_argument('--local-dir', default=str(LOCAL_BASE),
                        help=f'Local storage root (default: {LOCAL_BASE})')
    parser.add_argument('--onedrive-dir', default=str(ONEDRIVE_BASE),
                        help=f'OneDrive root (default: {ONEDRIVE_BASE})')
    parser.add_argument('--skip-reference', action='store_true',
                        help='Skip reference capture (use existing references)')

    args = parser.parse_args()

    if args.mode == 'single':
        run_single_droplet(args)
    elif args.mode == 'multi':
        run_multi_droplet(args)
    elif args.mode == 'reference':
        run_reference_only(args)


if __name__ == '__main__':
    main()
