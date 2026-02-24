# Video Upscaling Validation Tool

## Overview

The `validate_upscale.py` script performs comprehensive validation of upscaled videos to detect:
- Frame mapping errors (off‑by‑one issues)
- Tile mosaic/quadrant artifacts
- Cropping/offset drift
- Temporal instability (flicker)
- Cross‑frame contamination

## Installation

```bash
pip install -r requirements_validation.txt
```

## Usage

### Basic Usage

```bash
python validate_upscale.py \
    --input input.mp4 \
    --output output.mp4 \
    --scale 2 \
    --outdir report/ \
    [options]
```

### Options

- `--sample N` (default: 1) – Process every Nth frame.
- `--ssim-crop P` (default: 0.02) – Crop `P` fraction of border before SSIM/PSNR.
- `--ssim-yonly true|false` (default: true) – Compute SSIM on the Y (luma) channel only.
- `--preblur-sigma S` (default: 0.0) – Apply Gaussian blur with sigma `S` before SSIM.
- `--neighbor-window K` (default: 3) – Window size for off‑by‑K mapping detection.
- `--max-flagged N` (default: 0) – Fail with exit code 2 if more than `N` frames are flagged.
- `--max-flagged-rate R` (default: 0.0) – Fail if flagged frame rate exceeds `R` (0‑1).
- `--allow-trim` – Allow processing when input and output frame counts differ (trims to shorter).
- `--match-time` – Match frames by timestamp instead of index (not yet implemented).
- `--resample` – Resample output to match input frame rate (not yet implemented).
- Threshold options (see below) can also be overridden.

### Advanced Usage with Custom Thresholds

```bash
python validate_upscale.py \
    --input input.mp4 \
    --output output.mp4 \
    --scale 2 \
    --outdir report/ \
    --sample 2 \
    --ssim-threshold 0.92 \
    --ssim-outlier 0.03 \
    --quad-variance 0.25 \
    --quad-consistency 0.75 \
    --temporal-spike 1.5
```

### Quick Test

```bash
./test_validation.sh
```

## Thresholds Explained

### SSIM Threshold (`--ssim-threshold`, default: 0.90)

- **What it measures**: Structural similarity between input and downscaled output
- **Why 0.90**: Good quality upscaling should maintain >90% structural similarity
- **Lower values**: More lenient, may miss subtle artifacts
- **Higher values**: More strict, may flag acceptable quality loss

### SSIM Outlier Threshold (`--ssim-outlier`, default: 0.05)

- **What it measures**: Sudden SSIM drops vs rolling median (last 10 frames)
- **Why 0.05**: Detects sudden quality drops that indicate tile errors
- **Lower values**: Only detect severe drops
- **Higher values**: More sensitive to temporal inconsistencies

### Quadrant Variance Threshold (`--quad-variance`, default: 0.3)

- **What it measures**: Standard deviation of variance across 2x2 quadrants
- **Why 0.3**: Detects when one quadrant is unusually flat (possible placeholder)
- **Lower values**: More sensitive to flat regions
- **Higher values**: Only detect severe mosaic artifacts

### Quadrant Consistency Threshold (`--quad-consistency`, default: 0.7)

- **What it measures**: Histogram correlation between quadrants
- **Why 0.7**: Detects when quadrants have mismatched content (tile mixing)
- **Lower values**: More sensitive to content mismatches
- **Higher values**: Only detect severe inconsistencies

### Temporal Spike Threshold (`--temporal-spike`, default: 2.0)

- **What it measures**: Multiplier for frame‑to‑frame difference spikes
- **Why 2.0**: Output shouldn't change more than 2× the input change
- **Lower values**: More sensitive to flicker
- **Higher values**: Only detect severe temporal artifacts

## Output Files

### `frame_metrics.csv`

Per‑frame metrics with columns:
- `frame_index`
- `time_sec`
- `ssim`
- `psnr`
- `quadrant_variance`
- `quadrant_consistency`
- `quadrant_ssim_min`
- `seam_energy`
- `temporal_diff`
- `temporal_diff_in`
- `temporal_diff_out`
- `flags`

### `summary.json`

Overall validation summary including:
- Video properties
- Flag counts by category
- Worst 10 frames by SSIM
- `Thresholds used`

### `debug_frames/`

For each flagged frame, saves:
- `input.png`
- `output.png`
- `output_downscaled.png`
- `diff_heatmap.png`
- `output_quadrant_grid.png`
- `metrics.json`

## Flag Types

- `low_ssim`
- `ssim_outlier`
- `mosaic_suspected`
- `quadrant_inconsistent`
- `seam_energy_high`
- `temporal_flicker`
- `mapping_error_neighbor_+K` / `mapping_error_neighbor_-K` – Off‑by‑K mapping error where a neighbor within the configured window matches better

## Exit Codes

- `0`: Success, no issues detected
- `1`: Warning, issues detected (check report)
- `2`: Error, validation failed

## Performance Tips

- Use `--sample N` to process every Nth frame for faster validation
- For 30‑second videos, `--sample 3` gives good coverage in ~10 seconds
- Full validation (`--sample 1`) takes ~30 seconds for a 30‑second video

## Example Report Interpretation

```text
VALIDATION SUMMARY
============================================================
Frames processed: 300
Average SSIM: 0.9456
Average PSNR: 28.43 dB
Flagged frames:
  low_ssim: 2
  mosaic_suspected: 1
  mapping_error_neighbor_-1: 1
Worst 10 frames by SSIM:
  Frame   851 (t=28.34s): SSIM=0.8234 [low_ssim,mapping_error_neighbor_-1]
  Frame   423 (t=14.08s): SSIM=0.8567 [low_ssim]
  Frame   212 (t= 7.06s): SSIM=0.8912 [mosaic_suspected]
```

This indicates:
- Frame 851 has wrong content (matches previous frame better)
- Frame 423 has quality issues
- Frame 212 shows possible tile artifacts
Check `debug_frames/frame_000851/` to see the visual evidence.
