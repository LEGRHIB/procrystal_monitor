#!/usr/bin/env python3
"""
crystal_annotator/app.py
────────────────────────
Minimal Flask web app for manual crystallization annotation.

Run:
    python app.py
    open http://localhost:5050

Expects experiments at:
    ~/OneDrive - KU Leuven/DATA/experiments/<ID>/raw_images/img_YYYYMMDD_HHMMSS.png
"""

import json
import os
import re
import math
from datetime import datetime
from pathlib import Path
import io

try:
    from openpyxl import Workbook
    from openpyxl.styles import Font, PatternFill, Alignment
    HAS_OPENPYXL = True
except ImportError:
    HAS_OPENPYXL = False

try:
    import cv2
    import numpy as np
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

from flask import Flask, jsonify, render_template, request, send_file, abort

# ─── configuration ───────────────────────────────────────────────────────────

_default_root = Path.home() / "OneDrive - KU Leuven" / "DATA" / "experiments"
DATA_ROOT = Path(os.environ.get("CRYSTAL_DATA_ROOT", str(_default_root)))
FNAME_RE = re.compile(r"^img_(\d{8})_(\d{6})\.png$")

app = Flask(__name__)


# ─── droplet detection ───────────────────────────────────────────────────────

def detect_droplet_boundaries(image):
    """Find top and bottom meniscus lines of the droplet using intensity profile."""
    if not HAS_CV2:
        return None, None
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image.copy()
    h = gray.shape[0]
    row_profile = np.mean(gray, axis=1)
    kernel = np.ones(11) / 11
    smoothed = np.convolve(row_profile, kernel, mode='same')
    bright_rows = np.where(smoothed > np.mean(smoothed))[0]
    if len(bright_rows) < 2:
        margin = int(h * 0.15)
        return margin, h - margin
    top = max(0, int(bright_rows[0]) - 10)
    bottom = min(h, int(bright_rows[-1]) + 10)
    return top, bottom


def apply_enhancement(image):
    """
    Denoise → background-subtract → percentile-stretch pipeline.
    Works better than CLAHE for low-SNR tube images with uneven illumination:
      1. bilateralFilter  — remove grain while keeping crystal/meniscus edges
      2. large GaussianBlur → background estimate (slow illumination gradient)
      3. subtract background + 128 offset → flat mid-gray field, crystals visible
      4. percentile stretch (2–98) — use full dynamic range without noise boost
    """
    # 1. Denoise — edge-preserving
    denoised = cv2.bilateralFilter(image, d=9, sigmaColor=50, sigmaSpace=50)

    # 2. Background estimate via very large blur
    bg = cv2.GaussianBlur(denoised.astype(np.float32), (0, 0), sigmaX=40)

    # 3. Background subtraction centred on mid-gray
    corrected = denoised.astype(np.float32) - bg + 128.0
    corrected  = np.clip(corrected, 0, 255)

    # 4. Gentle contrast stretch
    p2, p98 = np.percentile(corrected, 2), np.percentile(corrected, 98)
    if p98 > p2:
        corrected = (corrected - p2) / (p98 - p2) * 255.0
        corrected = np.clip(corrected, 0, 255)

    return corrected.astype(np.uint8)


# In-memory cache: {exp_name: (ref_filename, np.ndarray)}
_ref_cache: dict = {}


def align_to_reference(ref_gray_f32: "np.ndarray", target_img: "np.ndarray") -> "np.ndarray":
    """Phase-correlation drift correction (translation only, clamped to ±50 px).
    Returns target_img warped so it aligns with ref_gray_f32."""
    tgt_gray = cv2.cvtColor(target_img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    (dx, dy), _ = cv2.phaseCorrelate(ref_gray_f32, tgt_gray)
    dx = max(-50.0, min(50.0, dx))
    dy = max(-50.0, min(50.0, dy))
    M = np.float32([[1, 0, -dx], [0, 1, -dy]])
    h, w = target_img.shape[:2]
    return cv2.warpAffine(target_img, M, (w, h),
                          flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REFLECT_101)


def apply_temporal_enhancement(ref_img: "np.ndarray", current_img: "np.ndarray") -> "np.ndarray":
    """Align current frame to reference, subtract background, percentile-stretch.
    Highlights changes (crystals growing) against the static tube background."""
    ref_gray_f32 = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    aligned = align_to_reference(ref_gray_f32, current_img)
    diff = aligned.astype(np.float32) - ref_img.astype(np.float32) + 128.0
    diff = np.clip(diff, 0, 255)
    p1, p99 = np.percentile(diff, 1), np.percentile(diff, 99)
    if p99 > p1:
        diff = (diff - p1) / (p99 - p1) * 255.0
        diff = np.clip(diff, 0, 255)
    return diff.astype(np.uint8)


def _build_coco_crystals(exp_name: str, dataset_dir: Path, frames: list, cal: dict):
    """Build coco_crystals.json from lengths.jsonl measurements."""
    lpath = annotations_dir(exp_name) / "lengths.jsonl"
    if not lpath.exists():
        return
    pixels_per_mm = cal.get("pixels_per_mm")
    coco = {
        "info": {"description": f"Crystal measurements — {exp_name}", "version": "1.0"},
        "categories": [{"id": 1, "name": "crystal", "supercategory": "none"}],
        "images": [{"id": f["frame_idx"], "file_name": f["filename"]} for f in frames],
        "annotations": [],
    }
    ann_id = 1
    for line in lpath.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        m = json.loads(line)
        cid = m.get("crystal_id", "")
        if cid.startswith("__"):
            continue
        p1, p2 = m.get("p1", {}), m.get("p2", {})
        if not p1 or not p2:
            continue
        x_min = min(p1["x"], p2["x"])
        y_min = min(p1["y"], p2["y"])
        bw = abs(p2["x"] - p1["x"])
        bh = abs(p2["y"] - p1["y"])
        ann = {
            "id": ann_id,
            "image_id": m.get("frame_idx"),
            "category_id": 1,
            "crystal_id": cid,
            "direction": m.get("direction"),
            "bbox": [x_min, y_min, bw, bh],
            "h_px": m.get("h_px"),
            "v_px": m.get("v_px"),
        }
        if pixels_per_mm:
            if m.get("h_px"):
                ann["h_um"] = round(m["h_px"] / pixels_per_mm * 1000, 2)
            if m.get("v_px"):
                ann["v_um"] = round(m["v_px"] / pixels_per_mm * 1000, 2)
        coco["annotations"].append(ann)
        ann_id += 1
    (dataset_dir / "coco_crystals.json").write_text(json.dumps(coco, indent=2))


# ─── helpers ─────────────────────────────────────────────────────────────────

def parse_timestamp(fname: str):
    """Return datetime from 'img_YYYYMMDD_HHMMSS.png', or None."""
    m = FNAME_RE.match(fname)
    if m is None:
        return None
    try:
        return datetime.strptime(f"{m.group(1)}_{m.group(2)}", "%Y%m%d_%H%M%S")
    except ValueError:
        return None


def scan_experiment(exp_path: Path) -> list[dict]:
    """Return sorted list of {filename, timestamp_iso, t_sec, frame_idx}."""
    raw = exp_path / "raw_images"
    if not raw.is_dir():
        return []

    frames = []
    for f in sorted(raw.iterdir()):
        ts = parse_timestamp(f.name)
        if ts:
            frames.append({"filename": f.name, "timestamp": ts})

    frames.sort(key=lambda r: r["timestamp"])
    if not frames:
        return []

    t0 = frames[0]["timestamp"]
    result = []
    for i, f in enumerate(frames):
        dt = (f["timestamp"] - t0).total_seconds()
        result.append({
            "frame_idx": i,
            "filename": f["filename"],
            "timestamp_iso": f["timestamp"].isoformat(),
            "t_sec": dt,
            "t_min": round(dt / 60.0, 4),
        })
    return result


def list_experiments() -> list[str]:
    """Return folder names that contain a raw_images/ subdirectory."""
    if not DATA_ROOT.is_dir():
        return []
    exps = []
    for d in sorted(DATA_ROOT.iterdir()):
        if d.is_dir() and (d / "raw_images").is_dir():
            exps.append(d.name)
    return exps


def annotations_dir(exp_name: str) -> Path:
    p = DATA_ROOT / exp_name / "annotations"
    p.mkdir(parents=True, exist_ok=True)
    return p


def results_dir(exp_name: str) -> Path:
    p = DATA_ROOT / exp_name / "results"
    p.mkdir(parents=True, exist_ok=True)
    return p


def get_tube_width_calibration(exp_name: str) -> dict:
    """Extract calibration from __TUBE_WIDTH__ measurement in lengths.jsonl.
    Tube width is known to be 1.5 mm.
    Returns {pixels_per_mm: float} or empty dict if not calibrated."""
    lpath = annotations_dir(exp_name) / "lengths.jsonl"
    if not lpath.exists():
        return {}

    for line in lpath.read_text().splitlines():
        line = line.strip()
        if line:
            obj = json.loads(line)
            if obj.get("crystal_id") == "__TUBE_WIDTH__":
                # Found calibration measurement
                tube_width_px = obj.get("h_px") or obj.get("v_px")
                if tube_width_px:
                    pixels_per_mm = tube_width_px / 1.5
                    p1 = obj.get("p1", {})
                    p2 = obj.get("p2", {})
                    result = {
                        "pixels_per_mm": pixels_per_mm,
                        "tube_width_px": tube_width_px,
                        "tube_width_mm": 1.5,
                    }
                    if p1 and p2:
                        result["tube_x1"] = int(min(p1.get("x", 0), p2.get("x", 0)))
                        result["tube_x2"] = int(max(p1.get("x", 0), p2.get("x", 0)))
                    return result
    return {}


def get_droplet_height(exp_name: str) -> dict:
    """
    Extract droplet height from lengths.jsonl.
    Returns {height_px, height_um, volume_ul} or empty dict.
    Assumes droplet diameter = 1.2 mm (cylindrical approximation).
    """
    lpath = annotations_dir(exp_name) / "lengths.jsonl"
    if not lpath.exists():
        return {}

    # First get calibration
    calibration = get_tube_width_calibration(exp_name)
    if not calibration:
        return {}

    pixels_per_mm = calibration["pixels_per_mm"]

    # Find __DROPLET_HEIGHT__ measurement
    for line in lpath.read_text().splitlines():
        line = line.strip()
        if line:
            obj = json.loads(line)
            if obj.get("crystal_id") == "__DROPLET_HEIGHT__":
                height_px = obj.get("v_px", 0)
                if height_px > 0:
                    height_mm = height_px / pixels_per_mm
                    height_um = height_mm * 1000
                    # Volume = π * r² * h, where r = 0.6 mm (diameter 1.2mm)
                    # Result in µL (mm³)
                    radius_mm = 0.6
                    volume_ul = 3.14159 * (radius_mm ** 2) * height_mm
                    return {
                        "height_px": height_px,
                        "height_mm": round(height_mm, 3),
                        "height_um": round(height_um, 2),
                        "volume_ul": round(volume_ul, 3),
                        "droplet_diameter_mm": 1.2
                    }

    return {}


def export_metrics_excel(exp_name: str) -> bytes:
    """Generate Excel file with dashboard metrics (matches JS dashboard exactly)."""
    if not HAS_OPENPYXL:
        return None

    frames = scan_experiment(DATA_ROOT / exp_name)
    crystals_data = load_crystals(exp_name)
    crystals = crystals_data.get("crystals", [])
    lpath = annotations_dir(exp_name) / "lengths.jsonl"

    # Load measurements
    all_meas = []
    if lpath.exists():
        for line in lpath.read_text().splitlines():
            if line.strip():
                obj = json.loads(line)
                # Apply µm conversion if calibrated
                calibration = get_tube_width_calibration(exp_name)
                pixels_per_mm = calibration.get("pixels_per_mm")
                if pixels_per_mm:
                    if obj.get("h_px"):
                        obj["h_um"] = round(obj["h_px"] / pixels_per_mm * 1000, 2)
                    if obj.get("v_px"):
                        obj["v_um"] = round(obj["v_px"] / pixels_per_mm * 1000, 2)
                all_meas.append(obj)

    calibration = get_tube_width_calibration(exp_name)
    droplet = get_droplet_height(exp_name)
    pixels_per_mm = calibration.get("pixels_per_mm")
    use_um = bool(pixels_per_mm)

    wb = Workbook()
    ws = wb.active
    ws.title = "Summary"

    # Summary sheet
    header_fill = PatternFill(start_color="366092", end_color="366092", fill_type="solid")
    header_font = Font(bold=True, color="FFFFFF")

    ws["A1"] = "Experiment Summary"
    ws["A1"].font = Font(bold=True, size=14)

    row = 3
    ws[f"A{row}"] = "Total Time (hr)"
    ws[f"B{row}"] = round((frames[-1]["t_min"] / 60) if frames else 0, 3)

    row += 1
    ws[f"A{row}"] = "Nucleation Events"
    ws[f"B{row}"] = len([c for c in crystals if c.get("nucleation_frame_idx") is not None])

    row += 1
    ws[f"A{row}"] = "Droplet Volume (µL)"
    ws[f"B{row}"] = droplet.get("volume_ul", "—")

    row += 1
    ws[f"A{row}"] = "Nucleation Rate (events/hr/µL)"
    if droplet.get("volume_ul") and frames:
        total_hr = (frames[-1]["t_min"] / 60) if frames else 0
        nuc_count = len([c for c in crystals if c.get("nucleation_frame_idx") is not None])
        nuc_rate = (nuc_count / (total_hr * droplet["volume_ul"])) if total_hr > 0 else 0
        ws[f"B{row}"] = round(nuc_rate, 6)
    else:
        ws[f"B{row}"] = "—"

    row += 1
    ws[f"A{row}"] = "Calibration (px/mm)"
    ws[f"B{row}"] = round(pixels_per_mm, 2) if pixels_per_mm else "—"

    # Metrics sheet
    ws2 = wb.create_sheet("Growth Metrics")

    # Header
    h_unit = "µm" if use_um else "px"
    v_unit = "µm" if use_um else "px"
    area_unit = "µm²" if use_um else "px²"
    rate_unit = "µm/hr" if use_um else "px/hr"

    headers = [
        "Crystal",
        f"H size ({h_unit})",
        f"V size ({v_unit})",
        f"H rate ({rate_unit})",
        f"V rate ({rate_unit})",
        f"Area ({area_unit})",
        "Aspect Ratio",
        "Induction Time (hr)"
    ]
    for col, header in enumerate(headers, 1):
        cell = ws2.cell(row=1, column=col)
        cell.value = header
        cell.fill = header_fill
        cell.font = header_font

    # Helper functions matching JS
    def linear_slope(points):
        if len(points) < 2:
            return None
        n = len(points)
        sx = sum(p[0] for p in points)
        sy = sum(p[1] for p in points)
        sxx = sum(p[0]**2 for p in points)
        sxy = sum(p[0]*p[1] for p in points)
        denom = n*sxx - sx*sx
        if denom == 0:
            return None
        return (n*sxy - sx*sy) / denom

    def latest_value(meas, key, frame_list):
        """Find latest value by timestamp (not file order)."""
        best = None
        for m in meas:
            if m.get(key) is None:
                continue
            frame_idx = m.get("frame_idx")
            if frame_idx is None or frame_idx >= len(frame_list):
                continue
            t_min = frame_list[frame_idx].get("t_min", 0)
            if best is None or t_min > best["t"]:
                best = {"t": t_min, "v": m[key]}
        return best["v"] if best else None

    # Data for each crystal
    for row_idx, crystal in enumerate(crystals, 2):
        cid = crystal["id"]
        c_meas = [m for m in all_meas if m.get("crystal_id") == cid]

        ws2.cell(row=row_idx, column=1).value = cid

        # Latest H and V (by timestamp, using same logic as JS)
        h_key = "h_um" if use_um else "h_px"
        v_key = "v_um" if use_um else "v_px"

        h_latest = latest_value(c_meas, h_key, frames)
        v_latest = latest_value(c_meas, v_key, frames)

        ws2.cell(row=row_idx, column=2).value = round(h_latest, 2) if h_latest else "—"
        ws2.cell(row=row_idx, column=3).value = round(v_latest, 2) if v_latest else "—"

        # Growth rates (slope) - match JS logic
        h_points = []
        v_points = []
        for m in c_meas:
            frame_idx = m.get("frame_idx")
            if frame_idx is None or frame_idx >= len(frames):
                continue
            t_min = frames[frame_idx].get("t_min", 0)
            t_hr = t_min / 60

            h_val = m.get(h_key)
            v_val = m.get(v_key)

            if h_val is not None:
                h_points.append((t_hr, h_val))
            if v_val is not None:
                v_points.append((t_hr, v_val))

        h_slope = linear_slope(h_points)
        v_slope = linear_slope(v_points)

        ws2.cell(row=row_idx, column=4).value = round(h_slope, 2) if h_slope else "—"
        ws2.cell(row=row_idx, column=5).value = round(v_slope, 2) if v_slope else "—"

        # Area (ellipse) - use math.pi to match JS Math.PI
        if h_latest and v_latest:
            area = math.pi * (h_latest / 2) * (v_latest / 2)
            ws2.cell(row=row_idx, column=6).value = round(area, 2)
        else:
            ws2.cell(row=row_idx, column=6).value = "—"

        # Aspect ratio
        if h_latest and v_latest:
            ws2.cell(row=row_idx, column=7).value = round(h_latest / v_latest, 3)
        else:
            ws2.cell(row=row_idx, column=7).value = "—"

        # Induction time
        if crystal.get("nucleation_frame_idx") is not None:
            frame_idx = crystal["nucleation_frame_idx"]
            if frame_idx < len(frames):
                t_min = frames[frame_idx]["t_min"]
                t_hr = t_min / 60
                ws2.cell(row=row_idx, column=8).value = round(t_hr, 3)
        if crystal.get("nucleation_frame_idx") is None:
            ws2.cell(row=row_idx, column=8).value = "Not marked"

    # Auto-size columns
    for ws_sheet in [ws, ws2]:
        for column in ws_sheet.columns:
            max_length = 0
            column_letter = column[0].column_letter
            for cell in column:
                try:
                    max_length = max(max_length, len(str(cell.value)))
                except:
                    pass
            ws_sheet.column_dimensions[column_letter].width = max_length + 2

    # Save to bytes
    output = io.BytesIO()
    wb.save(output)
    output.seek(0)
    return output.getvalue()


# ─── routes: pages ───────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


# ─── routes: API ─────────────────────────────────────────────────────────────

@app.get("/api/experiments")
def api_experiments():
    return jsonify(list_experiments())


@app.get("/api/frames/<exp_name>")
def api_frames(exp_name: str):
    exp_path = DATA_ROOT / exp_name
    if not exp_path.is_dir():
        abort(404)
    frames = scan_experiment(exp_path)
    return jsonify(frames)


@app.get("/api/image/<exp_name>/<filename>")
def api_image(exp_name: str, filename: str):
    img_path = DATA_ROOT / exp_name / "raw_images" / filename
    if not img_path.is_file():
        abort(404)
    return send_file(img_path, mimetype="image/png")


@app.get("/api/calibration/<exp_name>")
def api_get_calibration(exp_name: str):
    """Get calibration (pixels_per_mm) extracted from __TUBE_WIDTH__ measurement."""
    cal = get_tube_width_calibration(exp_name)
    return jsonify(cal)


@app.get("/api/droplet/<exp_name>")
def api_get_droplet(exp_name: str):
    """Get droplet height and volume data."""
    droplet = get_droplet_height(exp_name)
    return jsonify(droplet)


@app.get("/api/detect_droplet/<exp_name>/<filename>")
def api_detect_droplet(exp_name: str, filename: str):
    """Auto-detect droplet bounding box. Returns {x, y, w, h} in original image pixels."""
    if not HAS_CV2:
        return jsonify({"error": "opencv-python not installed — run: pip install opencv-python numpy"}), 501

    img_path = DATA_ROOT / exp_name / "raw_images" / filename
    if not img_path.is_file():
        abort(404)

    img = cv2.imread(str(img_path))
    if img is None:
        abort(404)

    img_w = img.shape[1]
    cal = get_tube_width_calibration(exp_name)

    # Use tube x-coords from calibration if available, otherwise centre 60%
    x1 = cal.get("tube_x1", int(img_w * 0.2))
    x2 = cal.get("tube_x2", int(img_w * 0.8))
    x1 = max(0, min(x1, img_w - 1))
    x2 = max(x1 + 1, min(x2, img_w))

    top, bottom = detect_droplet_boundaries(img[:, x1:x2])

    pad_x = max(10, int((x2 - x1) * 0.1))
    return jsonify({
        "x": max(0, x1 - pad_x),
        "y": top,
        "w": min(img_w, x2 + pad_x) - max(0, x1 - pad_x),
        "h": bottom - top,
    })


@app.get("/api/enhanced_crop/<exp_name>/<filename>")
def api_enhanced_crop(exp_name: str, filename: str):
    """Return a temporally-enhanced PNG for the given crop region.
    Query params: x, y, w, h (in original image pixels); ref=<filename> (reference frame)."""
    if not HAS_CV2:
        return jsonify({"error": "opencv-python not installed"}), 501

    img_path = DATA_ROOT / exp_name / "raw_images" / filename
    if not img_path.is_file():
        abort(404)

    img = cv2.imread(str(img_path))
    if img is None:
        abort(404)

    img_h_full, img_w_full = img.shape[:2]
    try:
        x = int(request.args.get("x", 0))
        y = int(request.args.get("y", 0))
        w = int(request.args.get("w", img_w_full))
        h = int(request.args.get("h", img_h_full))
    except (ValueError, TypeError):
        x, y, w, h = 0, 0, img_w_full, img_h_full

    x  = max(0, min(x, img_w_full - 1))
    y  = max(0, min(y, img_h_full - 1))
    x2 = max(x + 1, min(x + w, img_w_full))
    y2 = max(y + 1, min(y + h, img_h_full))

    # Load reference frame (default: frame 0 of this experiment)
    ref_filename = request.args.get("ref", None)
    if not ref_filename:
        exp_frames = scan_experiment(DATA_ROOT / exp_name)
        ref_filename = exp_frames[0]["filename"] if exp_frames else filename

    cached = _ref_cache.get(exp_name)
    if cached is None or cached[0] != ref_filename:
        ref_path = DATA_ROOT / exp_name / "raw_images" / ref_filename
        ref_img = cv2.imread(str(ref_path))
        if ref_img is not None:
            _ref_cache[exp_name] = (ref_filename, ref_img)
    else:
        ref_img = cached[1]

    crop = img[y:y2, x:x2]
    if ref_img is not None and filename != ref_filename:
        ref_crop = ref_img[y:y2, x:x2]
        enhanced = apply_temporal_enhancement(ref_crop, crop)
    else:
        enhanced = apply_enhancement(crop)  # fallback when viewing the ref frame itself

    ok, buf = cv2.imencode(".png", enhanced)
    if not ok:
        abort(500)
    return send_file(io.BytesIO(buf.tobytes()), mimetype="image/png")


@app.post("/api/export/<exp_name>/dataset")
def api_export_dataset(exp_name: str):
    """Batch-export crops, CLAHE-enhanced images, and COCO JSON into <exp>/dataset/."""
    if not HAS_CV2:
        return jsonify({"error": "opencv-python not installed — run: pip install opencv-python numpy"}), 501

    exp_path = DATA_ROOT / exp_name
    if not exp_path.is_dir():
        abort(404)

    frames = scan_experiment(exp_path)
    if not frames:
        return jsonify({"error": "No frames found in experiment"}), 400

    dataset_dir  = exp_path / "dataset"
    crops_dir    = dataset_dir / "crops"
    enhanced_dir = dataset_dir / "enhanced"
    crops_dir.mkdir(parents=True, exist_ok=True)
    enhanced_dir.mkdir(parents=True, exist_ok=True)

    cal = get_tube_width_calibration(exp_name)

    # Load frame 0 as the temporal reference for background subtraction
    ref_img = cv2.imread(str(exp_path / "raw_images" / frames[0]["filename"]))

    coco_droplets = {
        "info": {"description": f"Droplet bounding boxes — {exp_name}", "version": "1.0"},
        "categories": [{"id": 1, "name": "droplet", "supercategory": "none"}],
        "images": [],
        "annotations": [],
    }
    ann_id    = 1
    processed = 0
    errors    = []

    for frame in frames:
        img_path = exp_path / "raw_images" / frame["filename"]
        img = cv2.imread(str(img_path))
        if img is None:
            errors.append(frame["filename"])
            continue

        img_w_px = img.shape[1]

        x1_t = cal.get("tube_x1", int(img_w_px * 0.2))
        x2_t = cal.get("tube_x2", int(img_w_px * 0.8))
        x1_t = max(0, min(x1_t, img_w_px - 1))
        x2_t = max(x1_t + 1, min(x2_t, img_w_px))

        top, bottom = detect_droplet_boundaries(img[:, x1_t:x2_t])
        pad_x = max(10, int((x2_t - x1_t) * 0.1))

        cx = max(0, x1_t - pad_x)
        cy = top
        cw = min(img_w_px, x2_t + pad_x) - cx
        ch = bottom - cy

        crop = img[cy:cy + ch, cx:cx + cw]
        if ref_img is not None and frame["filename"] != frames[0]["filename"]:
            ref_crop = ref_img[cy:cy + ch, cx:cx + cw]
            enhanced = apply_temporal_enhancement(ref_crop, crop)
        else:
            enhanced = apply_enhancement(crop)

        stem = frame["filename"].replace(".png", "")
        cv2.imwrite(str(crops_dir    / f"{stem}.png"), crop)
        cv2.imwrite(str(enhanced_dir / f"{stem}.png"), enhanced)

        coco_droplets["images"].append({
            "id":             frame["frame_idx"],
            "file_name":      f"{stem}.png",
            "width":          cw,
            "height":         ch,
            "crop_x":         cx,
            "crop_y":         cy,
            "timestamp_iso":  frame["timestamp_iso"],
            "t_min":          frame["t_min"],
        })
        coco_droplets["annotations"].append({
            "id":          ann_id,
            "image_id":    frame["frame_idx"],
            "category_id": 1,
            "bbox":        [0, 0, cw, ch],
            "area":        cw * ch,
            "iscrowd":     0,
        })
        ann_id    += 1
        processed += 1

    (dataset_dir / "coco_droplets.json").write_text(json.dumps(coco_droplets, indent=2))
    _build_coco_crystals(exp_name, dataset_dir, frames, cal)

    return jsonify({
        "ok":         True,
        "processed":  processed,
        "errors":     errors,
        "dataset_dir": str(dataset_dir),
    })


@app.get("/api/export/<exp_name>/metrics")
def api_export_metrics(exp_name: str):
    """Export dashboard metrics as Excel file and save a copy to results/."""
    if not HAS_OPENPYXL:
        return jsonify({"error": "openpyxl not installed"}), 500

    exp_path = DATA_ROOT / exp_name
    if not exp_path.is_dir():
        abort(404)

    excel_data = export_metrics_excel(exp_name)
    if not excel_data:
        return jsonify({"error": "Failed to generate Excel"}), 500

    # Auto-save a copy to <experiment>/results/
    fname = f"{exp_name}_metrics.xlsx"
    save_path = results_dir(exp_name) / fname
    save_path.write_bytes(excel_data)

    # Send file from disk
    return send_file(
        str(save_path),
        mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        as_attachment=True,
    )


# ── crystals ──────────────────────────────────────────────────────────────────

def crystals_file(exp_name: str) -> Path:
    """Return path to crystals.json for an experiment."""
    return annotations_dir(exp_name) / "crystals.json"


def load_crystals(exp_name: str) -> dict:
    """Load crystals data, return {crystals: [...]} or empty structure."""
    cf = crystals_file(exp_name)
    if cf.exists():
        return json.loads(cf.read_text())
    return {"crystals": []}


def save_crystals(exp_name: str, data: dict):
    """Save crystals data."""
    cf = crystals_file(exp_name)
    cf.write_text(json.dumps(data, indent=2))


@app.get("/api/crystals/<exp_name>")
def api_get_crystals(exp_name: str):
    """Return list of all crystals in an experiment."""
    data = load_crystals(exp_name)
    return jsonify(data.get("crystals", []))


@app.post("/api/crystals/<exp_name>")
def api_create_crystal(exp_name: str):
    """Create a new crystal. Body: {notes: "optional notes"}"""
    data = request.get_json(force=True) or {}
    crystals = load_crystals(exp_name)

    # Generate crystal ID
    existing_ids = [c.get("id") for c in crystals["crystals"]]
    crystal_id = f"crystal_{len(existing_ids) + 1}"

    new_crystal = {
        "id": crystal_id,
        "nucleation_frame_idx": None,
        "nucleation_timestamp_iso": None,
        "notes": data.get("notes", ""),
    }
    crystals["crystals"].append(new_crystal)
    save_crystals(exp_name, crystals)
    return jsonify(new_crystal), 201


@app.delete("/api/crystals/<exp_name>/<crystal_id>")
def api_delete_crystal(exp_name: str, crystal_id: str):
    """Delete a crystal and its associated measurements."""
    crystals = load_crystals(exp_name)
    crystals["crystals"] = [c for c in crystals["crystals"] if c["id"] != crystal_id]
    save_crystals(exp_name, crystals)

    # Also delete measurements for this crystal
    lpath = annotations_dir(exp_name) / "lengths.jsonl"
    if lpath.exists():
        lines = [l for l in lpath.read_text().splitlines() if l.strip()]
        lines = [l for l in lines if json.loads(l).get("crystal_id") != crystal_id]
        lpath.write_text("\n".join(lines) + ("\n" if lines else ""))

    return jsonify({"ok": True})


@app.put("/api/crystals/<exp_name>/<crystal_id>/nucleation")
def api_set_crystal_nucleation(exp_name: str, crystal_id: str):
    """Set nucleation time for a crystal. Body: {nucleation_frame_idx: N, nucleation_timestamp_iso: "..."}"""
    data = request.get_json(force=True)
    crystals = load_crystals(exp_name)

    for crystal in crystals["crystals"]:
        if crystal["id"] == crystal_id:
            crystal["nucleation_frame_idx"] = data.get("nucleation_frame_idx")
            crystal["nucleation_timestamp_iso"] = data.get("nucleation_timestamp_iso")
            break

    save_crystals(exp_name, crystals)
    return jsonify({"ok": True})


# ── measurements ──────────────────────────────────────────────────────────────

@app.get("/api/lengths/<exp_name>")
def api_get_lengths(exp_name: str):
    """Return all length measurements as a JSON array, optionally filtered by crystal_id.
    Only shows measurements for crystals that are defined in crystals.json.
    Automatically includes µm conversion if tube width is calibrated."""
    crystal_id = request.args.get("crystal_id")
    lpath = annotations_dir(exp_name) / "lengths.jsonl"
    if not lpath.exists():
        return jsonify([])

    # Load valid crystal IDs from crystals.json
    crystals = load_crystals(exp_name)
    valid_crystal_ids = {c["id"] for c in crystals.get("crystals", [])}
    # Also allow special measurements
    valid_crystal_ids.add("__TUBE_WIDTH__")
    valid_crystal_ids.add("__DROPLET_HEIGHT__")

    # Get calibration if available
    calibration = get_tube_width_calibration(exp_name)
    pixels_per_mm = calibration.get("pixels_per_mm")

    lines = []
    for line in lpath.read_text().splitlines():
        line = line.strip()
        if line:
            obj = json.loads(line)
            # Skip measurements for crystals that don't exist in crystals.json
            if obj.get("crystal_id") not in valid_crystal_ids:
                continue

            # Add µm conversion if calibrated
            if pixels_per_mm:
                if obj.get("h_px"):
                    obj["h_um"] = round(obj["h_px"] / pixels_per_mm * 1000, 2)
                if obj.get("v_px"):
                    obj["v_um"] = round(obj["v_px"] / pixels_per_mm * 1000, 2)

            if crystal_id is None or obj.get("crystal_id") == crystal_id:
                lines.append(obj)
    return jsonify(lines)


@app.post("/api/lengths/<exp_name>")
def api_add_length(exp_name: str):
    """Append one measurement line to lengths.jsonl.
    Body: {crystal_id: "crystal_1", frame_idx: N, length: L, ...}"""
    data = request.get_json(force=True)
    lpath = annotations_dir(exp_name) / "lengths.jsonl"
    with lpath.open("a") as fh:
        fh.write(json.dumps(data) + "\n")
    return jsonify({"ok": True})


@app.delete("/api/lengths/<exp_name>/<crystal_id>/<int:line_idx>")
def api_delete_length(exp_name: str, crystal_id: str, line_idx: int):
    """Delete a single measurement by its 0-based line index within a crystal."""
    lpath = annotations_dir(exp_name) / "lengths.jsonl"
    if not lpath.exists():
        abort(404)

    # Filter only lines for this crystal
    all_lines = [l for l in lpath.read_text().splitlines() if l.strip()]
    crystal_lines = [l for l in all_lines if json.loads(l).get("crystal_id") == crystal_id]

    if line_idx < 0 or line_idx >= len(crystal_lines):
        abort(404)

    # Remove the specific measurement
    to_delete = crystal_lines[line_idx]
    all_lines.remove(to_delete)
    lpath.write_text("\n".join(all_lines) + ("\n" if all_lines else ""))
    return jsonify({"ok": True})


# ─── main ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"\n  Data root:  {DATA_ROOT}")
    print(f"  Experiments found: {list_experiments() or '(none yet)'}\n")
    print("  API Endpoints:")
    print("    GET  /api/experiments              - List all experiments")
    print("    GET  /api/frames/<exp_name>        - List frames for experiment")
    print("    GET  /api/crystals/<exp_name>      - List all crystals")
    print("    POST /api/crystals/<exp_name>      - Create a new crystal")
    print("    PUT  /api/crystals/<exp_name>/<id>/nucleation - Set nucleation time")
    print("    GET  /api/lengths/<exp_name>?crystal_id=<id> - Get measurements")
    print()
    print("  Open  http://localhost:5050  in your browser.\n")
    app.run(host="127.0.0.1", port=5050, debug=False)
