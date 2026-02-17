"""
analyze_experiment.py — Run analysis on a captured experiment folder
====================================================================

Reads the folder structure created by autocapture.py and runs the
full crystal detection + tracking + statistics pipeline.

Usage:
    # Analyze a single-droplet experiment
    python analyze_experiment.py --experiment ~/CrystalMonitor/experiment_20260216_140000

    # Analyze specific droplets from a multi-droplet experiment
    python analyze_experiment.py --experiment ~/CrystalMonitor/experiment_20260216_140000 --droplets 0-9

    # Analyze all 130 droplets and generate heatmaps
    python analyze_experiment.py --experiment ~/CrystalMonitor/experiment_20260216_140000 --all
"""

import os
import sys
import json
import glob
import argparse
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import PipelineConfig
from pipeline import SingleDropletPipeline
from multi_droplet import MultiDropletManager
from dashboard import CrystalDashboard
import cv2
import pandas as pd


def load_experiment_metadata(experiment_dir: Path) -> dict:
    """Load experiment.json from an experiment folder."""
    meta_path = experiment_dir / "experiment.json"
    if not meta_path.exists():
        print(f"ERROR: No experiment.json found in {experiment_dir}")
        print("  Make sure you're pointing to a folder created by autocapture.py")
        sys.exit(1)
    with open(meta_path) as f:
        return json.load(f)


def parse_droplet_range(range_str: str, max_droplets: int) -> list:
    """Parse '0-9' or '0,5,10' or '0-9,20-29' into a list of droplet IDs."""
    ids = set()
    for part in range_str.split(','):
        part = part.strip()
        if '-' in part:
            start, end = part.split('-', 1)
            ids.update(range(int(start), int(end) + 1))
        else:
            ids.add(int(part))
    return sorted(i for i in ids if 0 <= i < max_droplets)


def get_droplet_images(experiment_dir: Path, droplet_id: int) -> list:
    """Get sorted list of capture image paths for a droplet."""
    folder = experiment_dir / "captures" / f"droplet_{droplet_id:03d}"
    if not folder.exists():
        return []
    return sorted(folder.glob("cycle_*.png"))


def get_reference_path(experiment_dir: Path, droplet_id: int) -> str:
    """Get the reference image path for a droplet."""
    ref = experiment_dir / "references" / f"droplet_{droplet_id:03d}.png"
    return str(ref) if ref.exists() else None


def parse_timestamp_from_filename(filename: str) -> float:
    """
    Extract elapsed seconds from filename like 'cycle_0003_t005400s.png'
    Returns the timestamp in seconds.
    """
    name = Path(filename).stem  # 'cycle_0003_t005400s'
    # Find the t...s part
    parts = name.split('_')
    for part in parts:
        if part.startswith('t') and part.endswith('s'):
            try:
                return float(part[1:-1])
            except ValueError:
                pass
    return 0.0


def analyze_single_droplet(experiment_dir: Path, droplet_id: int,
                           config: PipelineConfig, results_dir: Path):
    """Run full analysis on one droplet."""
    image_paths = get_droplet_images(experiment_dir, droplet_id)
    ref_path = get_reference_path(experiment_dir, droplet_id)

    if not image_paths:
        print(f"  Droplet {droplet_id}: no images found, skipping.")
        return None

    # Parse timestamps from filenames
    timestamps = [parse_timestamp_from_filename(str(p)) for p in image_paths]

    print(f"  Droplet {droplet_id}: {len(image_paths)} frames, "
          f"reference={'yes' if ref_path else 'no'}")

    # Run pipeline
    pipeline = SingleDropletPipeline(config)
    frame_df = pipeline.process_timelapse(
        [str(p) for p in image_paths],
        timestamps=timestamps,
        reference_path=ref_path,
        verbose=False
    )

    # Export
    droplet_dir = results_dir / f"droplet_{droplet_id:03d}"
    pipeline.export_results(str(droplet_dir), verbose=False)

    return pipeline


def analyze_all_droplets(experiment_dir: Path, droplet_ids: list,
                         config: PipelineConfig, results_dir: Path):
    """Analyze multiple droplets and generate cross-droplet statistics."""

    results_dir.mkdir(parents=True, exist_ok=True)

    # Analyze each droplet
    pipelines = {}
    for did in droplet_ids:
        pipeline = analyze_single_droplet(experiment_dir, did, config, results_dir)
        if pipeline is not None:
            pipelines[did] = pipeline

    if not pipelines:
        print("No droplets had images to analyze.")
        return

    # Combine results into multi-droplet summary
    all_summaries = []
    for did, pipeline in pipelines.items():
        track_summary = pipeline.tracker.summary_dataframe()
        if not track_summary.empty:
            track_summary['droplet_id'] = did
            all_summaries.append(track_summary)

    if all_summaries:
        combined = pd.concat(all_summaries, ignore_index=True)
        combined.to_csv(results_dir / "all_droplets_tracks.csv", index=False)
        print(f"\n  Combined track data: {len(combined)} crystal tracks across "
              f"{len(pipelines)} droplets")

    # Per-droplet summary
    droplet_summary = []
    for did, pipeline in pipelines.items():
        n_frames = len(pipeline.frame_results)
        tracks = pipeline.tracker.get_all_tracks()
        if pipeline.frame_results:
            last = pipeline.frame_results[-1]
            droplet_summary.append({
                'droplet_id': did,
                'frames_analyzed': n_frames,
                'final_crystal_count': last.get('num_crystals', 0),
                'final_nucleation_count': last.get('num_nucleation', 0),
                'final_total_area': last.get('total_crystal_area_px', 0),
                'total_tracks': len(tracks),
                'nucleation_detected': any(t.nucleated for t in tracks.values()),
            })

    if droplet_summary:
        summary_df = pd.DataFrame(droplet_summary)
        summary_df.to_csv(results_dir / "droplet_summary.csv", index=False)
        print(f"  Droplet summary saved ({len(droplet_summary)} droplets)")

    # Generate heatmaps if enough droplets
    if len(pipelines) >= 4:
        print("  Generating heatmap visualizations...")
        dashboard = CrystalDashboard(config,
                                      output_dir=str(results_dir / "figures"))

        # Build a MultiDropletManager from the results
        config.multi_droplet.num_droplets = max(droplet_ids) + 1
        manager = MultiDropletManager(config)

        for did, pipeline in pipelines.items():
            if did < config.multi_droplet.num_droplets:
                state = manager.droplets.get(did)
                if state and pipeline.frame_results:
                    last = pipeline.frame_results[-1]
                    state.current_crystal_count = last.get('num_crystals', 0)
                    state.current_total_area_px = last.get('total_crystal_area_px', 0)
                    tracks = pipeline.tracker.get_all_tracks()
                    for t in tracks.values():
                        if t.nucleated:
                            state.nucleation_detected = True
                            state.nucleation_time_sec = t.timestamps[0] if t.timestamps else None
                            break

        dashboard.plot_multi_droplet_heatmaps(manager)
        dashboard.plot_nucleation_statistics(manager)

    print(f"\n  All results saved to: {results_dir}")


def main():
    parser = argparse.ArgumentParser(
        description='Analyze a captured crystallization experiment'
    )
    parser.add_argument('--experiment', required=True,
                        help='Path to experiment folder (created by autocapture.py)')
    parser.add_argument('--droplets', default=None,
                        help='Droplet IDs to analyze (e.g., "0-9" or "0,5,10")')
    parser.add_argument('--all', action='store_true',
                        help='Analyze all droplets found in the experiment')
    parser.add_argument('--output', default=None,
                        help='Output directory (default: experiment/results)')
    parser.add_argument('--config', default=None,
                        help='Path to pipeline config JSON')

    args = parser.parse_args()

    experiment_dir = Path(args.experiment)
    metadata = load_experiment_metadata(experiment_dir)

    num_droplets = metadata.get('num_droplets', 1)
    print(f"Experiment: {metadata.get('experiment_name', '?')}")
    print(f"  Droplets: {num_droplets}")
    print(f"  Interval: {metadata.get('interval_sec', '?')}s")

    # Config
    config_path = args.config or (experiment_dir / "config.json")
    if Path(config_path).exists():
        config = PipelineConfig.load(str(config_path))
    else:
        config = PipelineConfig()
        config.multi_droplet.snapshot_interval_sec = metadata.get('interval_sec', 60.0)

    # Determine which droplets to analyze
    if args.droplets:
        droplet_ids = parse_droplet_range(args.droplets, num_droplets)
    elif args.all:
        droplet_ids = list(range(num_droplets))
    else:
        # Default: analyze all droplets that have images
        droplet_ids = []
        for i in range(num_droplets):
            if get_droplet_images(experiment_dir, i):
                droplet_ids.append(i)

    if not droplet_ids:
        print("No droplets to analyze.")
        return

    # Output directory
    results_dir = Path(args.output) if args.output else experiment_dir / "results"

    print(f"  Analyzing {len(droplet_ids)} droplet(s)...\n")

    analyze_all_droplets(experiment_dir, droplet_ids, config, results_dir)

    print("\nDone!")


if __name__ == '__main__':
    main()
