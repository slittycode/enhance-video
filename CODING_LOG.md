# Coding Log

## 2026-02-04
- Added unit tests covering dry-run behavior, output path safety, and dependency/path handling.
- Adjusted audio extraction to normalize output extension based on codec and return the output path.
- Added a guard against overwriting the input file and skipped reassembly in dry-run mode.
- Tests: `python -m unittest discover -s tests`

## 2026-02-20
- Refactored `upscale_video.py` into structured, testable stages with dataclasses and `main(argv)` return codes.
- Added bundled Real-ESRGAN auto-detection, explicit toolchain resolution, and validated runtime args.
- Added `--upscale-mode` (`auto`/`batch`/`frame`) and `--jobs`; `auto` now prefers batch for speed.
- Added frame-failure fallback resizing to preserve output dimensions when per-frame AI upscale fails.
- Switched frame extraction to `-fps_mode passthrough` to avoid deprecated `-vsync` usage.
- Expanded tests to cover binary resolution, mode selection, and `main()` return behavior.
- Verification: `./.venv/bin/python -m unittest discover -s tests -v`, plus end-to-end smoke runs (video-only and video+audio).
- Added reusable workspace flow with `--work-dir` and `--cleanup-work-dir`, including cached frame reuse for fast retries.
- Added optional temporal anti-flicker post-process (`--temporal-filter` with `light`/`medium`/`strong`) and audio-safe remux behavior.
- Expanded tests to cover workspace preparation, frame extraction reuse, and temporal filter expression mapping.
- Verification: repeated run using shared `--work-dir` switched to frame-mode resume and skipped already-upscaled frames.
- Added quality profile support with `--profile {custom,max_quality}` (default `max_quality`) and profile-driven defaults for model/TTA/temporal/preset/CRF.
- Added runtime guardrail estimation with sampling (`--runtime-guardrail-hours`, `--runtime-sample-frames`) and quality fallback ladder.
- Added workspace manifest fingerprinting + stale cache invalidation for safe resume/retry behavior.
- Added project `README.md` with max-quality and guardrailed workflows.
- Expanded tests to cover profile defaults, guardrail candidate ladder, and workspace fingerprint validation.
- Added machine-specific runtime calibration persistence (`--calibration-file`, `--reset-calibration`) with EMA FPS updates.
- Guardrail estimator now uses learned calibration entries when available (`source=calibration`) and falls back to sampling when needed.
- Expanded tests to cover calibration load/update and calibration-derived runtime estimates.
- Added scene-adaptive mode (`--scene-adaptive`) with scene boundary detection, scene range merging, and per-scene guardrail candidate selection.
- Added scene controls (`--scene-threshold`, `--scene-min-frames`, `--scene-sample-frames`, `--scene-budget-slack`).
- Added scene helper tests for range building/merging and per-scene candidate projection selection.
- Added texture-aware scene planning with normalized scene texture scoring and weighted per-scene budget allocation (`--texture-priority`).
- Upgraded scene-adaptive execution to true scene chunk pipeline: per-scene upscale -> scene chunk render -> concat -> final audio mux.
- Scene chunk artifacts are auto-cleaned by default; opt-out via `--keep-scene-chunks`.
- Added `--plan-only` JSON preflight mode for strategy inspection without running upscale/reassembly.
- Expanded tests to cover texture-biased candidate selection, scene texture scoring, chunk concat/mux helpers, and plan-only JSON output path.
- Verification: `./.venv/bin/python -m py_compile upscale_video.py tests/test_upscale_video.py` and `./.venv/bin/python -m unittest discover -s tests -v`.
