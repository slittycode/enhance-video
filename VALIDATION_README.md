# Video Upscaling Validation Tool

## Overview

The `validate_upscale.py` script performs comprehensive validation of upscaled videos to detect:
- Frame mapping errors (off-by-one issues)
- Tile mosaic/quadrant artifacts
- Cropping/offset drift
- Temporal instability (flicker)
- Cross-frame contamination

## Installation

```bash
pip install -r requirements_validation.txt
```

## Usage

### Basic Usage
```bash
python validate_upscale.py --input input.mp4 --output output.mp4 --scale 2 --outdir report/
```

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
- **What it measures**: Multiplier for frame-to-frame difference spikes
- **Why 2.0**: Output shouldn't change more than 2x the input change
- **Lower values**: More sensitive to flicker
- **Higher values**: Only detect severe temporal artifacts

## Output Files

### `frame_metrics.csv`
Per-frame metrics with columns:
- `frame_index`: Frame number
- `time_sec`: Timestamp in seconds
- `ssim`: Structural similarity score
- `psnr`: Peak signal-to-noise ratio (dB)
- `quadrant_variance`: Quadrant variance score
- `quadrant_consistency`: Quadrant histogram consistency
- `temporal_diff`: Frame-to-frame difference
- `flags`: Comma-separated list of detected issues

### `summary.json`
Overall validation summary including:
- Video properties
- Flag counts by category
- Worst 10 frames by SSIM
- Thresholds used

### `debug_frames/`
For each flagged frame, saves:
- `input.png`: Original input frame
- `output.png`: Upscaled output frame
- `output_downscaled.png`: Output downscaled to input size
- `diff_heatmap.png`: Visual difference heatmap
- `output_quadrant_grid.png`: Output with 2x2 grid (if mosaic suspected)
- `metrics.json`: Frame-specific metrics

## Flag Types

### `low_ssim`
SSIM below threshold indicates poor quality or wrong content

### `ssim_outlier`
Sudden SSIM drop vs rolling median suggests tile errors

### `mosaic_suspected`
High quadrant variance indicates possible placeholder tiles

### `quadrant_inconsistent`
Low quadrant consistency suggests mixed content between tiles

### `temporal_flicker`
Frame-to-frame change exceeds input change by threshold multiplier

### `mapping_error_neighbor_+1` / `mapping_error_neighbor_-1`
Neighbor frame matches better than current frame (off-by-one error)

## Exit Codes

- `0`: Success, no issues detected
- `1`: Warning, issues detected (check report)
- `2`: Error, validation failed

## Performance Tips

- Use `--sample N` to process every Nth frame for faster validation
- For 30-second videos, `--sample 3` gives good coverage in ~10 seconds
- Full validation (`--sample 1`) takes ~30 seconds for 30-second video

## Example Report Interpretation

```
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
