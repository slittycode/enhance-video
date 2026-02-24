# Refactor Model Selection for Scale and Heuristics

## Summary
Implements explicit `--type` logic handling, scale-aware model selection with fallback support, and a computer vision frame-analysis heuristic to automatically detect "animated" content vs "real-life" content for optimal upscaling results.

## Key Changes
- **Explicit Override Tracking**: Added mappings for `--model` in `parse_cli_overrides` so that variables provided manually don't get silently cleared by `max_quality` profile defaults.
- **Scale-Aware Resolution**: `resolve_model` checks for missing `2x` models first, warns in console if they are absent, and dynamically defaults back to calculating `4x` with outscale.
- **Auto-Feature Hook (`--type auto`)**: Created `analyze_video_type` via `cv2` and `ffmpeg` to sample frames across the video and categorize the input via Laplacian variance / Canny line algorithms. Outputs the reasoning back to the CLI in text form.
- **Documentation**: Standardized `README.md` to directly feature the "best-looks" commands for `--type animation`, and the new `--type auto` functionality.

## Testing
- Extended the testing suite via `test_model_selection.py` handling explicit configuration overrides, model existence fallback logic, and mock CV2 image generations to confirm heuristic triggers work correctly.
- Maintained a 100% pass-rate on the entire pre-existing E2E smoke tests and integration layer test suite.
