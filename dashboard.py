"""
Dashboard Module
================
Real-time visualization dashboard for crystal monitoring.
Uses matplotlib for plotting (works without extra dependencies).
Can run in interactive mode or generate static report figures.
"""

import cv2
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for file output
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from typing import List, Dict, Tuple, Optional
import os

from config import PipelineConfig
from crystal_detector import CrystalDetection
from growth_tracker import GrowthTracker, CrystalTrack
from multi_droplet import MultiDropletManager


class CrystalDashboard:
    """
    Generates visualization figures for crystal monitoring data.

    Can produce:
    - Single-droplet analysis plots (growth curves, detection overlays)
    - Multi-droplet heatmaps (nucleation times, crystal counts, growth rates)
    - Statistical distribution plots
    - Time-lapse montages
    """

    def __init__(self, config: PipelineConfig, output_dir: str = "./results/figures"):
        self.config = config
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        # Style settings
        plt.rcParams.update({
            'figure.facecolor': '#1a1a2e',
            'axes.facecolor': '#16213e',
            'axes.edgecolor': '#e2e2e2',
            'axes.labelcolor': '#e2e2e2',
            'text.color': '#e2e2e2',
            'xtick.color': '#e2e2e2',
            'ytick.color': '#e2e2e2',
            'grid.color': '#2a2a4a',
            'font.size': 10,
        })

    # ---- Single Droplet Visualizations ----

    def plot_growth_curves(self, tracker: GrowthTracker,
                           droplet_id: int = 0,
                           save: bool = True) -> plt.Figure:
        """
        Plot crystal area and diameter vs time for all tracks in a droplet.
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f'Crystal Growth Analysis — Droplet {droplet_id}',
                     fontsize=14, fontweight='bold', color='#00d4ff')

        tracks = tracker.get_all_tracks()

        # 1) Area vs Time
        ax = axes[0, 0]
        for tid, track in tracks.items():
            t = np.array(track.timestamps) - (track.timestamps[0] if track.timestamps else 0)
            ax.plot(t, track.areas, 'o-', markersize=3, label=f'Crystal {tid}')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Area (px²)')
        ax.set_title('Crystal Area Growth')
        ax.grid(True, alpha=0.3)
        if len(tracks) <= 10:
            ax.legend(fontsize=7)

        # 2) Equivalent Diameter vs Time
        ax = axes[0, 1]
        for tid, track in tracks.items():
            t = np.array(track.timestamps) - (track.timestamps[0] if track.timestamps else 0)
            ax.plot(t, track.diameters, 'o-', markersize=3, label=f'Crystal {tid}')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Equiv. Diameter (px)')
        ax.set_title('Crystal Diameter Growth')
        ax.grid(True, alpha=0.3)

        # 3) Growth Rate Distribution
        ax = axes[1, 0]
        rates = [t.growth_rate_px2_per_sec for t in tracks.values() if t.growth_rate_px2_per_sec > 0]
        if rates:
            ax.hist(rates, bins=max(5, len(rates) // 3), color='#00d4ff',
                    alpha=0.7, edgecolor='white')
        ax.set_xlabel('Growth Rate (px²/s)')
        ax.set_ylabel('Count')
        ax.set_title('Growth Rate Distribution')
        ax.grid(True, alpha=0.3)

        # 4) Aspect Ratio vs Time
        ax = axes[1, 1]
        for tid, track in tracks.items():
            t = np.array(track.timestamps) - (track.timestamps[0] if track.timestamps else 0)
            ax.plot(t, track.aspect_ratios, 'o-', markersize=3, label=f'Crystal {tid}')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Aspect Ratio')
        ax.set_title('Crystal Morphology Evolution')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save:
            path = os.path.join(self.output_dir, f'growth_curves_droplet_{droplet_id}.png')
            fig.savefig(path, dpi=150, bbox_inches='tight')
            print(f"  Saved: {path}")

        return fig

    def plot_detection_overview(self, image: np.ndarray,
                                 detections: List[CrystalDetection],
                                 frame_info: str = "",
                                 save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot an annotated image with detection details sidebar.
        """
        fig = plt.figure(figsize=(16, 8))
        gs = GridSpec(1, 3, width_ratios=[2, 1, 1], figure=fig)

        # Main annotated image
        ax_img = fig.add_subplot(gs[0])
        from crystal_detector import CrystalDetector as CD
        temp_detector = CD(self.config)
        annotated = temp_detector.annotate_image(image, detections)
        if len(annotated.shape) == 3:
            annotated = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        ax_img.imshow(annotated)
        ax_img.set_title(f'Detections {frame_info}', color='#00d4ff')
        ax_img.axis('off')

        # Area bar chart
        ax_bar = fig.add_subplot(gs[1])
        if detections:
            ids = [d.detection_id for d in detections]
            areas = [d.area_px for d in detections]
            colors = ['#ffcc00' if d.is_nucleation else '#00d4ff' for d in detections]
            ax_bar.barh(range(len(ids)), areas, color=colors)
            ax_bar.set_yticks(range(len(ids)))
            ax_bar.set_yticklabels([f'#{i}' for i in ids], fontsize=8)
            ax_bar.set_xlabel('Area (px²)')
            ax_bar.set_title('Crystal Areas')
        ax_bar.grid(True, alpha=0.3)

        # Feature table
        ax_table = fig.add_subplot(gs[2])
        ax_table.axis('off')
        if detections:
            table_data = []
            for d in detections[:10]:  # Limit to first 10
                table_data.append([
                    f'#{d.detection_id}',
                    f'{d.area_px:.0f}',
                    f'{d.circularity:.2f}',
                    f'{d.aspect_ratio:.2f}',
                    'Nuc' if d.is_nucleation else 'Crys'
                ])
            table = ax_table.table(
                cellText=table_data,
                colLabels=['ID', 'Area', 'Circ.', 'AR', 'Type'],
                loc='center',
                cellLoc='center'
            )
            table.auto_set_font_size(False)
            table.set_fontsize(8)
            table.scale(1, 1.5)
            # Style the table
            for cell in table.get_celld().values():
                cell.set_facecolor('#16213e')
                cell.set_edgecolor('#2a2a4a')
                cell.set_text_props(color='#e2e2e2')

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
        elif self.output_dir:
            path = os.path.join(self.output_dir, f'detection_overview_{frame_info}.png')
            fig.savefig(path, dpi=150, bbox_inches='tight')

        return fig

    # ---- Multi-Droplet Visualizations ----

    def plot_multi_droplet_heatmaps(self, manager: MultiDropletManager,
                                      save: bool = True) -> plt.Figure:
        """
        Generate 2x2 heatmaps for the multi-droplet experiment:
        - Crystal count per droplet
        - Nucleation time per droplet
        - Total crystal area per droplet
        - Growth rate per droplet
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Multi-Droplet Crystallization Map',
                     fontsize=16, fontweight='bold', color='#00d4ff')

        datasets = [
            (manager.crystal_count_map(), 'Crystal Count', 'YlOrRd'),
            (manager.nucleation_time_map(), 'Nucleation Time (s)', 'viridis'),
            (manager.total_area_map(), 'Total Crystal Area (px²)', 'plasma'),
            (manager.growth_rate_map(), 'Mean Growth Rate (px²/s)', 'coolwarm'),
        ]

        for ax, (data, title, cmap) in zip(axes.flat, datasets):
            im = ax.imshow(data, cmap=cmap, aspect='auto', interpolation='nearest')
            ax.set_title(title, fontsize=11)
            ax.set_xlabel('Column')
            ax.set_ylabel('Row')
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        plt.tight_layout()

        if save:
            path = os.path.join(self.output_dir, 'multi_droplet_heatmaps.png')
            fig.savefig(path, dpi=150, bbox_inches='tight')
            print(f"  Saved: {path}")

        return fig

    def plot_nucleation_statistics(self, manager: MultiDropletManager,
                                    save: bool = True) -> plt.Figure:
        """
        Statistical plots for nucleation across all droplets:
        - Nucleation time distribution (histogram)
        - Nucleation probability over time (survival curve)
        - Crystal count distribution
        """
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle('Nucleation Statistics',
                     fontsize=14, fontweight='bold', color='#00d4ff')

        nuc_times = manager.nucleation_times()
        valid_times = [t for t in nuc_times.values() if t is not None]

        # 1) Nucleation time histogram
        ax = axes[0]
        if valid_times:
            ax.hist(valid_times, bins=max(5, len(valid_times) // 5),
                    color='#00d4ff', alpha=0.7, edgecolor='white')
        ax.set_xlabel('Nucleation Time (s)')
        ax.set_ylabel('Number of Droplets')
        ax.set_title('Nucleation Time Distribution')
        ax.grid(True, alpha=0.3)

        # 2) Survival curve (fraction not yet nucleated vs time)
        ax = axes[1]
        if valid_times:
            sorted_times = sorted(valid_times)
            n_total = len(nuc_times)
            survival = []
            time_points = [0] + sorted_times + [sorted_times[-1] * 1.1]
            for t in time_points:
                frac_not_nucleated = sum(1 for nt in sorted_times if nt > t) / n_total
                frac_not_nucleated += sum(1 for nt in nuc_times.values() if nt is None) / n_total
                survival.append(frac_not_nucleated)
            ax.step(time_points, survival, where='post', color='#ff6b6b', linewidth=2)
            ax.fill_between(time_points, survival, alpha=0.2, color='#ff6b6b', step='post')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Fraction Not Nucleated')
        ax.set_title('Nucleation Survival Curve')
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.3)

        # 3) Crystal count distribution
        ax = axes[2]
        summary = manager.summary_dataframe()
        if not summary.empty:
            counts = summary['current_crystal_count']
            ax.hist(counts, bins=max(5, int(counts.max()) + 1),
                    color='#4ecdc4', alpha=0.7, edgecolor='white')
        ax.set_xlabel('Number of Crystals')
        ax.set_ylabel('Number of Droplets')
        ax.set_title('Crystal Count Distribution')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save:
            path = os.path.join(self.output_dir, 'nucleation_statistics.png')
            fig.savefig(path, dpi=150, bbox_inches='tight')
            print(f"  Saved: {path}")

        return fig

    def create_timelapse_montage(self, images: List[np.ndarray],
                                  detections_list: List[List[CrystalDetection]],
                                  timestamps: List[float],
                                  cols: int = 4,
                                  save_path: Optional[str] = None) -> plt.Figure:
        """
        Create a montage showing the time-lapse of crystallization.
        """
        n = len(images)
        rows = (n + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
        fig.suptitle('Crystallization Time-Lapse', fontsize=14,
                     fontweight='bold', color='#00d4ff')

        if rows == 1:
            axes = [axes] if cols == 1 else axes
        axes_flat = np.array(axes).flatten()

        for i in range(len(axes_flat)):
            ax = axes_flat[i]
            if i < n:
                img = images[i].copy()
                if len(img.shape) == 3:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                # Draw contours on the image
                if i < len(detections_list):
                    for det in detections_list[i]:
                        color = (255, 255, 0) if det.is_nucleation else (0, 255, 0)
                        if len(img.shape) == 2:
                            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
                        cv2.drawContours(img, [det.contour], -1, color, 1)

                ax.imshow(img)
                t_str = f't = {timestamps[i]:.1f}s' if i < len(timestamps) else ''
                n_crys = len(detections_list[i]) if i < len(detections_list) else 0
                ax.set_title(f'{t_str}\n{n_crys} crystals', fontsize=9)
            ax.axis('off')

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
        elif self.output_dir:
            path = os.path.join(self.output_dir, 'timelapse_montage.png')
            fig.savefig(path, dpi=150, bbox_inches='tight')

        return fig

    def plot_single_frame_summary(self, image: np.ndarray,
                                    detections: List[CrystalDetection],
                                    frame_idx: int,
                                    timestamp: float,
                                    save: bool = True) -> plt.Figure:
        """Quick single-frame summary for real-time monitoring."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # Annotated image
        from crystal_detector import CrystalDetector as CD
        temp_det = CD(self.config)
        annotated = temp_det.annotate_image(image, detections)
        if len(annotated.shape) == 3:
            annotated = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        ax1.imshow(annotated)
        ax1.set_title(f'Frame {frame_idx} | t={timestamp:.1f}s', color='#00d4ff')
        ax1.axis('off')

        # Summary stats
        ax2.axis('off')
        n_nuc = sum(1 for d in detections if d.is_nucleation)
        n_crys = sum(1 for d in detections if not d.is_nucleation)
        total_area = sum(d.area_px for d in detections)
        mean_ar = np.mean([d.aspect_ratio for d in detections]) if detections else 0

        stats_text = (
            f"Frame: {frame_idx}\n"
            f"Time: {timestamp:.1f} s\n"
            f"──────────────────\n"
            f"Nucleation events: {n_nuc}\n"
            f"Grown crystals: {n_crys}\n"
            f"Total area: {total_area:.0f} px²\n"
            f"Mean aspect ratio: {mean_ar:.2f}\n"
        )
        ax2.text(0.1, 0.5, stats_text, transform=ax2.transAxes,
                fontsize=13, verticalalignment='center',
                fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='#16213e',
                         edgecolor='#00d4ff', alpha=0.9))

        plt.tight_layout()

        if save:
            path = os.path.join(self.output_dir, f'frame_{frame_idx:05d}_summary.png')
            fig.savefig(path, dpi=120, bbox_inches='tight')

        return fig
