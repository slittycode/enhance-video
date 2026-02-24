#!/usr/bin/env python3
"""
Video Upscaling Validation Tool
================================

Detects frame mapping errors, tile artifacts, cropping drift, and temporal instability.

Usage:
    python validate_upscale.py --input input.mp4 --output output.mp4 --scale 2 --outdir report/
"""

import argparse
import csv
import json
import math
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any

try:
    import cv2
    import numpy as np
    from skimage.metrics import structural_similarity as ssim
    from skimage.metrics import peak_signal_noise_ratio as psnr
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    print("ERROR: Required dependencies not installed.")
    print("Install with: pip install opencv-python scikit-image numpy")
    sys.exit(1)


@dataclass
class ValidationResult:
    frame_index: int
    time_sec: float
    ssim: float
    psnr: float
    quadrant_variance: float
    quadrant_consistency: float
    temporal_diff: float
    flags: List[str] = field(default_factory=list)
    # New metrics
    quadrant_ssim_min: float = 0.0
    seam_energy: float = 0.0
    temporal_diff_in: float = 0.0
    temporal_diff_out: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            'frame_index': self.frame_index,
            'time_sec': self.time_sec,
            'ssim': self.ssim,
            'psnr': self.psnr,
            'quadrant_variance': self.quadrant_variance,
            'quadrant_consistency': self.quadrant_consistency,
            'temporal_diff': self.temporal_diff,
            'quadrant_ssim_min': self.quadrant_ssim_min,
            'seam_energy': self.seam_energy,
            'temporal_diff_in': self.temporal_diff_in,
            'temporal_diff_out': self.temporal_diff_out,
            'flags': ','.join(self.flags)
        }


class VideoValidator:
    def __init__(self, input_path: Path, output_path: Path, scale_factor: int, 
                 outdir: Path, sample_every: int = 1,
                 ssim_crop: float = 0.02, ssim_yonly: bool = True, preblur_sigma: float = 0.0,
                 neighbor_window: int = 3, max_flagged: int = 0, max_flagged_rate: float = 0.0,
                 allow_trim: bool = False, match_time: bool = False, resample: bool = False):
        self.input_path = input_path
        self.output_path = output_path
        self.scale_factor = scale_factor
        self.outdir = outdir
        self.sample_every = sample_every
        
        # New configurable options
        self.ssim_crop = ssim_crop
        self.ssim_yonly = ssim_yonly
        self.preblur_sigma = preblur_sigma
        self.neighbor_window = neighbor_window
        self.max_flagged = max_flagged
        self.max_flagged_rate = max_flagged_rate
        self.allow_trim = allow_trim
        self.match_time = match_time
        self.resample = resample
        
        # Configurable thresholds (existing)
        self.ssim_threshold = 0.90
        self.ssim_outlier_threshold = 0.05  # Drop vs rolling median
        self.quadrant_variance_threshold = 0.3
        self.quadrant_consistency_threshold = 0.7
        self.temporal_spike_threshold = 2.0  # Output diff > threshold * input diff
        
        # Create output directories
        self.outdir.mkdir(parents=True, exist_ok=True)
        self.debug_dir = self.outdir / "debug_frames"
        self.debug_dir.mkdir(exist_ok=True)
        
        # Video captures
        self.input_cap = None
        self.output_cap = None
        
        # Results
        self.results: List[ValidationResult] = []
        
    def open_videos(self) -> Tuple[Dict, Dict]:
        """Open both videos and return their properties."""
        self.input_cap = cv2.VideoCapture(str(self.input_path))
        self.output_cap = cv2.VideoCapture(str(self.output_path))
        
        if not self.input_cap.isOpened() or not self.output_cap.isOpened():
            raise RuntimeError("Failed to open video files")
        
        # Get video properties
        input_props = {
            'width': int(self.input_cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'height': int(self.input_cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            'fps': self.input_cap.get(cv2.CAP_PROP_FPS),
            'frame_count': int(self.input_cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            'duration': int(self.input_cap.get(cv2.CAP_PROP_FRAME_COUNT)) / self.input_cap.get(cv2.CAP_PROP_FPS)
        }
        
        output_props = {
            'width': int(self.output_cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'height': int(self.output_cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            'fps': self.output_cap.get(cv2.CAP_PROP_FPS),
            'frame_count': int(self.output_cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            'duration': int(self.output_cap.get(cv2.CAP_PROP_FRAME_COUNT)) / self.output_cap.get(cv2.CAP_PROP_FPS)
        }
        
        # Validate scale factor
        expected_width = input_props['width'] * self.scale_factor
        expected_height = input_props['height'] * self.scale_factor
        
        if output_props['width'] != expected_width or output_props['height'] != expected_height:
            raise ValueError(
                f"Output dimensions don't match expected scale factor: "
                f"got {output_props['width']}x{output_props['height']}, "
                f"expected {expected_width}x{expected_height}"
            )
        
        return input_props, output_props
    
    def get_frame(self, cap: cv2.VideoCapture, index: int) -> Optional[np.ndarray]:
        """Get frame by index."""
        cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ret, frame = cap.read()
        return frame if ret else None
    
    def downscale_frame(self, frame: np.ndarray) -> np.ndarray:
        """Downscale frame back to input size."""
        h, w = frame.shape[:2]
        target_h = h // self.scale_factor
        target_w = w // self.scale_factor
        return cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)
    
    def compute_ssim(self, frame1: np.ndarray, frame2: np.ndarray) -> float:
        """Compute SSIM between two frames with optional cropping, Y-only, and preblur."""
        # Optional cropping of borders
        if self.ssim_crop > 0:
            h, w = frame1.shape[:2]
            crop_h = int(h * self.ssim_crop)
            crop_w = int(w * self.ssim_crop)
            frame1 = frame1[crop_h:h - crop_h, crop_w:w - crop_w]
            frame2 = frame2[crop_h:h - crop_h, crop_w:w - crop_w]
        # Optional preblur
        if self.preblur_sigma > 0:
            ksize = int(2 * round(3 * self.preblur_sigma) + 1)
            frame1 = cv2.GaussianBlur(frame1, (ksize, ksize), self.preblur_sigma)
            frame2 = cv2.GaussianBlur(frame2, (ksize, ksize), self.preblur_sigma)
        # Convert to Y channel if requested
        if self.ssim_yonly:
            # Convert BGR to YCrCb and take Y channel
            frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2YCrCb)[:, :, 0]
            frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2YCrCb)[:, :, 0]
        else:
            # Convert to grayscale for SSIM as fallback
            frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
            frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        score, _ = ssim(frame1, frame2, full=True)
        return float(score)
    
    def compute_psnr(self, frame1: np.ndarray, frame2: np.ndarray) -> float:
        """Compute PSNR between two frames."""
        return psnr(frame1, frame2)
    
    def compute_quadrant_ssim(self, input_frame: np.ndarray, output_frame: np.ndarray) -> float:
        """Compute per-quadrant SSIM between input and downscaled output quadrants, return minimum SSIM."""
        h, w = input_frame.shape[:2]
        h_mid, w_mid = h // 2, w // 2
        # Split input and output into quadrants
        input_quads = [
            input_frame[:h_mid, :w_mid],
            input_frame[:h_mid, w_mid:],
            input_frame[h_mid:, :w_mid],
            input_frame[h_mid:, w_mid:]
        ]
        output_quads = [
            output_frame[:h_mid, :w_mid],
            output_frame[:h_mid, w_mid:],
            output_frame[h_mid:, :w_mid],
            output_frame[h_mid:, w_mid:]
        ]
        ssims = []
        for iq, oq in zip(input_quads, output_quads):
            ssims.append(self.compute_ssim(iq, oq))
        return min(ssims) if ssims else 0.0
    
    def compute_seam_energy(self, frame: np.ndarray) -> float:
        """Compute seam energy of the frame."""
        # Simple implementation: compute gradient magnitude
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        grad_mag = np.sqrt(grad_x**2 + grad_y**2)
        return np.mean(grad_mag)
    
    def analyze_quadrants(self, frame: np.ndarray) -> Tuple[float, float]:
        """Analyze quadrant consistency to detect mosaic artifacts."""
        h, w = frame.shape[:2]
        h_mid, w_mid = h // 2, w // 2
        
        # Split into 2x2 quadrants
        quadrants = [
            frame[:h_mid, :w_mid],      # Top-left
            frame[:h_mid, w_mid:],      # Top-right
            frame[h_mid:, :w_mid],      # Bottom-left
            frame[h_mid:, w_mid:]       # Bottom-right
        ]
        
        # Compute variance for each quadrant (measure of detail/flatness)
        variances = []
        histograms = []
        
        for quad in quadrants:
            # Convert to grayscale for analysis
            gray_quad = cv2.cvtColor(quad, cv2.COLOR_BGR2GRAY)
            variances.append(np.var(gray_quad))
            
            # Compute histogram for color consistency
            hist = cv2.calcHist([quad], [0, 1, 2], None, [50, 50, 50], [0, 256, 0, 256, 0, 256])
            histograms.append(hist.flatten() / hist.sum())
        
        # Check for quadrant variance (one quadrant much flatter than others)
        variance_score = np.std(variances) / (np.mean(variances) + 1e-6)
        
        # Check for histogram consistency between quadrants
        # High consistency = similar content, low = different content (possible mixing)
        hist_similarities = []
        for i in range(len(histograms)):
            for j in range(i + 1, len(histograms)):
                # Use correlation to compare histograms
                corr = cv2.compareHist(histograms[i], histograms[j], cv2.HISTCMP_CORREL)
                hist_similarities.append(corr)
        
        consistency_score = np.mean(hist_similarities)
        
        return variance_score, consistency_score
    
    def compute_temporal_diff(self, frame1: np.ndarray, frame2: np.ndarray) -> float:
        """Compute frame-to-frame difference."""
        diff = cv2.absdiff(frame1, frame2)
        return np.mean(diff)
    
    def check_neighbor_mapping(self, input_frame: np.ndarray, output_frames: List[np.ndarray], 
                              frame_idx: int) -> Optional[str]:
        """Check if a neighbor within window matches better (off-by-K detection)."""
        best_ssim = self.compute_ssim(input_frame, output_frames[frame_idx])
        best_match = frame_idx
        # Search within window
        start = max(0, frame_idx - self.neighbor_window)
        end = min(len(output_frames) - 1, frame_idx + self.neighbor_window)
        for j in range(start, end + 1):
            if j == frame_idx:
                continue
            s = self.compute_ssim(input_frame, output_frames[j])
            if s > best_ssim + 0.05:
                best_ssim = s
                best_match = j
        if best_match != frame_idx:
            offset = best_match - frame_idx
            return f"mapping_error_neighbor_{offset:+d}"
        return None
    
    def save_debug_frame(self, frame_idx: int, input_frame: np.ndarray, 
                        output_frame: np.ndarray, downscaled_output: np.ndarray,
                        result: ValidationResult):
        """Save debug frames for flagged issues."""
        frame_dir = self.debug_dir / f"frame_{frame_idx:06d}"
        frame_dir.mkdir(exist_ok=True)
        
        # Save frames
        cv2.imwrite(str(frame_dir / "input.png"), input_frame)
        cv2.imwrite(str(frame_dir / "output.png"), output_frame)
        cv2.imwrite(str(frame_dir / "output_downscaled.png"), downscaled_output)
        
        # Save difference heatmap
        diff = cv2.absdiff(input_frame, downscaled_output)
        diff_colored = cv2.applyColorMap(diff, cv2.COLORMAP_JET)
        cv2.imwrite(str(frame_dir / "diff_heatmap.png"), diff_colored)
        
        # Save quadrant grid if mosaic suspected
        if 'mosaic_suspected' in result.flags:
            h, w = output_frame.shape[:2]
            h_mid, w_mid = h // 2, w // 2
            
            # Draw grid on output
            with_grid = output_frame.copy()
            cv2.line(with_grid, (w_mid, 0), (w_mid, h), (0, 255, 0), 2)
            cv2.line(with_grid, (0, h_mid), (w, h_mid), (0, 255, 0), 2)
            
            # Add quadrant labels
            labels = ['TL', 'TR', 'BL', 'BR']
            positions = [(w_mid//2, h_mid//2), (w_mid*3//2, h_mid//2), 
                        (w_mid//2, h_mid*3//2), (w_mid*3//2, h_mid*3//2)]
            
            for label, (x, y) in zip(labels, positions):
                cv2.putText(with_grid, label, (x-10, y+10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            cv2.imwrite(str(frame_dir / "output_quadrant_grid.png"), with_grid)
        
        # Save metrics as JSON
        with open(frame_dir / "metrics.json", 'w') as f:
            json.dump(result.to_dict(), f, indent=2)
    
    def validate(self) -> Dict[str, Any]:
        """Run the full validation."""
        print("Opening videos...")
        input_props, output_props = self.open_videos()
        
        # Dimension check already in open_videos; handle allow_trim
        if input_props['frame_count'] != output_props['frame_count']:
            if not self.allow_trim:
                raise ValueError("Frame count mismatch and --allow-trim not set.")
            else:
                max_frames = min(input_props['frame_count'], output_props['frame_count'])
        else:
            max_frames = input_props['frame_count']
        
        print(f"Input: {input_props['width']}x{input_props['height']}, "
              f"{input_props['fps']:.2f}fps, {input_props['frame_count']} frames")
        print(f"Output: {output_props['width']}x{output_props['height']}, "
              f"{output_props['fps']:.2f}fps, {output_props['frame_count']} frames")
        
        # Determine frames to process based on sample and possible time matching (not implemented fully)
        frames_to_process = range(0, max_frames, self.sample_every)
        print(f"Processing {len(frames_to_process)} frames (sample every {self.sample_every})...")
        
        # Store frames for temporal analysis
        input_frames = []
        output_frames = []
        downscaled_outputs = []
        
        # First pass: collect frames
        for i in frames_to_process:
            input_frame = self.get_frame(self.input_cap, i)
            output_frame = self.get_frame(self.output_cap, i)
            if input_frame is None or output_frame is None:
                raise RuntimeError(f"Failed to read frame {i}")
            input_frames.append(input_frame)
            output_frames.append(output_frame)
            downscaled_outputs.append(self.downscale_frame(output_frame))
        
        # Second pass: analyze frames
        rolling_median_ssim = []
        for idx, i in enumerate(frames_to_process):
            input_frame = input_frames[idx]
            output_frame = output_frames[idx]
            downscaled_output = downscaled_outputs[idx]
            
            # Basic metrics
            ssim_score = self.compute_ssim(input_frame, downscaled_output)
            psnr_score = self.compute_psnr(input_frame, downscaled_output)
            
            # Quadrant analysis
            quad_variance, quad_consistency = self.analyze_quadrants(output_frame)
            
            # New per-quadrant SSIM
            quad_ssim_min = self.compute_quadrant_ssim(input_frame, downscaled_output)
            
            # Seam energy on output frame
            seam_energy = self.compute_seam_energy(output_frame)
            
            # Temporal difference
            temporal_diff = 0.0
            temporal_diff_in = 0.0
            temporal_diff_out = 0.0
            if idx > 0:
                temporal_diff = self.compute_temporal_diff(downscaled_outputs[idx], downscaled_outputs[idx-1])
                temporal_diff_in = self.compute_temporal_diff(input_frames[idx], input_frames[idx-1])
                temporal_diff_out = self.compute_temporal_diff(output_frames[idx], output_frames[idx-1])
            
            # Create result
            result = ValidationResult(
                frame_index=i,
                time_sec=i / input_props['fps'],
                ssim=ssim_score,
                psnr=psnr_score,
                quadrant_variance=quad_variance,
                quadrant_consistency=quad_consistency,
                temporal_diff=temporal_diff,
                quadrant_ssim_min=quad_ssim_min,
                seam_energy=seam_energy,
                temporal_diff_in=temporal_diff_in,
                temporal_diff_out=temporal_diff_out
            )
            
            # Update rolling median for SSIM
            rolling_median_ssim.append(ssim_score)
            if len(rolling_median_ssim) > 10:
                rolling_median_ssim.pop(0)
            median_ssim = np.median(rolling_median_ssim)
            
            # Flags
            if ssim_score < self.ssim_threshold:
                result.flags.append('low_ssim')
            if abs(ssim_score - median_ssim) > self.ssim_outlier_threshold:
                result.flags.append('ssim_outlier')
            if quad_variance > self.quadrant_variance_threshold:
                result.flags.append('mosaic_suspected')
            if quad_consistency < self.quadrant_consistency_threshold:
                result.flags.append('quadrant_inconsistent')
            if seam_energy > self.quadrant_variance_threshold:  # reuse threshold as placeholder
                result.flags.append('seam_energy_high')
            if idx > 0 and temporal_diff > self.temporal_spike_threshold * temporal_diff_in:
                result.flags.append('temporal_flicker')
            # Mapping error detection with configurable window
            if ssim_score < 0.85:
                mapping_error = self.check_neighbor_mapping(input_frame, downscaled_outputs, idx)
                if mapping_error:
                    result.flags.append(mapping_error)
            
            self.results.append(result)
            
            # Save debug frames if any flags
            if result.flags:
                self.save_debug_frame(i, input_frame, output_frame, downscaled_output, result)
            
            if (idx + 1) % 10 == 0:
                print(f"  Processed {idx + 1}/{len(frames_to_process)} frames")
        
        # Generate report
        return self.generate_report(input_props, output_props)
    
    def generate_report(self, input_props: Dict, output_props: Dict) -> Dict[str, Any]:
        """Generate validation report, including new metrics in CSV and summary."""
        # Count flags
        flag_counts = {}
        worst_frames = sorted(self.results, key=lambda r: r.ssim)[:10]
        
        for result in self.results:
            for flag in result.flags:
                flag_counts[flag] = flag_counts.get(flag, 0) + 1
        
        # Save CSV with extended columns
        csv_path = self.outdir / "frame_metrics.csv"
        with open(csv_path, 'w', newline='') as f:
            fieldnames = list(self.results[0].to_dict().keys())
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for result in self.results:
                writer.writerow(result.to_dict())
        
        # Save summary
        summary = {
            'input_video': str(self.input_path),
            'output_video': str(self.output_path),
            'scale_factor': self.scale_factor,
            'frames_processed': len(self.results),
            'input_props': input_props,
            'output_props': output_props,
            'flag_counts': flag_counts,
            'worst_frames': [
                {'frame': r.frame_index, 'time': r.time_sec, 'ssim': r.ssim, 'flags': r.flags}
                for r in worst_frames
            ],
            'thresholds': {
                'ssim_threshold': self.ssim_threshold,
                'ssim_outlier_threshold': self.ssim_outlier_threshold,
                'quadrant_variance_threshold': self.quadrant_variance_threshold,
                'quadrant_consistency_threshold': self.quadrant_consistency_threshold,
                'temporal_spike_threshold': self.temporal_spike_threshold,
                'ssim_crop': self.ssim_crop,
                'ssim_yonly': self.ssim_yonly,
                'preblur_sigma': self.preblur_sigma,
                'neighbor_window': self.neighbor_window,
                'max_flagged': self.max_flagged,
                'max_flagged_rate': self.max_flagged_rate
            }
        }
        
        summary_path = self.outdir / "summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Print summary (unchanged)
        print("\n" + "="*60)
        print("VALIDATION SUMMARY")
        print("="*60)
        print(f"Frames processed: {len(self.results)}")
        print(f"Average SSIM: {np.mean([r.ssim for r in self.results]):.4f}")
        print(f"Average PSNR: {np.mean([r.psnr for r in self.results]):.2f} dB")
        print()
        print("Flagged frames:")
        for flag, count in sorted(flag_counts.items()):
            print(f"  {flag}: {count}")
        print()
        print("Worst 10 frames by SSIM:")
        for r in worst_frames:
            print(f"  Frame {r.frame_index:4d} (t={r.time_sec:6.2f}s): "
                  f"SSIM={r.ssim:.4f} {'['+','.join(r.flags)+']' if r.flags else ''}")
        print()
        print(f"Report saved to: {self.outdir}")
        print(f"  - frame_metrics.csv: Per-frame metrics")
        print(f"  - summary.json: Summary and worst frames")
        print(f"  - debug_frames/: Debug images for flagged frames")
        
        return summary


def main():
    parser = argparse.ArgumentParser(description="Validate video upscaling quality")
    parser.add_argument("--input", required=True, type=Path, help="Input video path")
    parser.add_argument("--output", required=True, type=Path, help="Upscaled output video path")
    parser.add_argument("--scale", type=int, required=True, help="Scale factor (e.g., 2, 3, 4)")
    parser.add_argument("--outdir", type=Path, required=True, help="Output directory for report")
    parser.add_argument("--ssim-crop", type=float, default=0.02,
                        help="Crop border percentage before SSIM (default 0.02)")
    parser.add_argument("--ssim-yonly", type=lambda x: (str(x).lower() == 'true'), default=True,
                        help="Compute SSIM on Y channel only (default true)")
    parser.add_argument("--preblur-sigma", type=float, default=0.0,
                        help="Gaussian blur sigma before SSIM (default 0.0)")
    parser.add_argument("--neighbor-window", type=int, default=3,
                        help="Window size for neighbor mapping (default 3)")
    parser.add_argument("--max-flagged", type=int, default=0,
                        help="Maximum number of flagged frames to allow (default 0)")
    parser.add_argument("--max-flagged-rate", type=float, default=0.0,
                        help="Maximum flagged frame rate (fraction) to allow (default 0.0)")
    parser.add_argument("--allow-trim", action='store_true',
                        help="Allow processing when frame counts differ (trim to shorter)")
    parser.add_argument("--match-time", action='store_true',
                        help="Match frames by timestamp instead of index")
    parser.add_argument("--resample", action='store_true',
                        help="Resample output to match input frame rate")
    parser.add_argument("--sample", type=int, default=1, help="Sample every N frames (default: 1)")
    
    # Threshold options
    parser.add_argument("--ssim-threshold", type=float, default=0.90, 
                       help="SSIM threshold for flagging (default: 0.90)")
    parser.add_argument("--ssim-outlier", type=float, default=0.05,
                       help="SSIM outlier threshold vs rolling median (default: 0.05)")
    parser.add_argument("--quad-variance", type=float, default=0.3,
                       help="Quadrant variance threshold (default: 0.3)")
    parser.add_argument("--quad-consistency", type=float, default=0.7,
                       help="Quadrant consistency threshold (default: 0.7)")
    parser.add_argument("--temporal-spike", type=float, default=2.0,
                       help="Temporal spike multiplier (default: 2.0)")
    
    args = parser.parse_args()
    
    # Check dependencies
    if not CV2_AVAILABLE:
        print("ERROR: Required dependencies not installed.")
        print("Install with: pip install opencv-python scikit-image numpy")
        sys.exit(1)
    
    # Create validator
    validator = VideoValidator(
        input_path=args.input,
        output_path=args.output,
        scale_factor=args.scale,
        outdir=args.outdir,
        sample_every=args.sample,
        ssim_crop=args.ssim_crop,
        ssim_yonly=args.ssim_yonly,
        preblur_sigma=args.preblur_sigma,
        neighbor_window=args.neighbor_window,
        max_flagged=args.max_flagged,
        max_flagged_rate=args.max_flagged_rate,
        allow_trim=args.allow_trim,
        match_time=args.match_time,
        resample=args.resample
    )
    
    # Set thresholds (already set via args)
    validator.ssim_threshold = args.ssim_threshold
    validator.ssim_outlier_threshold = args.ssim_outlier
    validator.quadrant_variance_threshold = args.quad_variance
    validator.quadrant_consistency_threshold = args.quad_consistency
    validator.temporal_spike_threshold = args.temporal_spike

    try:
        # Run validation
        summary = validator.validate()
    except Exception as e:
        print(f"ERROR: {e}")
        sys.exit(2)

    # After validation, handle max-flagged and rate exit codes.
    # Thresholds are frame-based: one frame may carry multiple flag labels.
    flagged_frames = sum(1 for result in validator.results if result.flags)
    total_flag_events = sum(summary['flag_counts'].values())
    flagged_rate = (
        flagged_frames / summary['frames_processed']
        if summary['frames_processed'] > 0
        else 0
    )
    if validator.max_flagged > 0 and flagged_frames > validator.max_flagged:
        print(
            f"\nERROR: Flagged frames {flagged_frames} exceed max-flagged {validator.max_flagged}"
        )
        sys.exit(2)
    if validator.max_flagged_rate > 0 and flagged_rate > validator.max_flagged_rate:
        print(
            f"\nERROR: Flagged frame rate {flagged_rate:.2%} exceeds max-flagged-rate {validator.max_flagged_rate:.2%}"
        )
        sys.exit(2)
    if flagged_frames > 0:
        print(
            f"\nWARNING: {flagged_frames} frame(s) flagged ({total_flag_events} total issue labels). Check report for details."
        )
        sys.exit(1)
    else:
        print("\nSUCCESS: No issues detected!")
        sys.exit(0)


if __name__ == "__main__":
    main()
