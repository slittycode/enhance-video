# Enhance AI Video Upscaler

Local Real-ESRGAN video upscaler tuned for Apple machines with a quality-first default profile.

## Requirements

- Python 3.11+
- `ffmpeg` and `ffprobe` in `PATH`
- `tqdm` in your Python environment
- `realesrgan-ncnn-vulkan` binary (auto-detected from bundled vendor directory — e.g. `enhance-ai/realesrgan-ncnn-v0.2.0-macos` in this repo)

## Quick Start

Use the project venv:

```bash
./.venv/bin/python upscale_video.py --help
```

### Tracing

The tool includes optional OpenTelemetry tracing so you can profile runtime behavior.
Install tracing dependencies from `requirements.txt` and then start the AI Toolkit trace
collector (`ai-mlstudio.tracing.open`). Spans will be exported to
`http://localhost:4318`.

```bash
# install dependencies (venv active)
pip install -r requirements.txt
# open trace viewer from VS Code command palette: "AI Toolkit: Open tracing"
# then run the script as usual; spans appear in the trace viewer
./.venv/bin/python upscale_video.py input.mp4 --output output.mp4
```

## Flow 1: Max Quality

Use this when image quality is the absolute priority.

```bash
./.venv/bin/python upscale_video.py \
  input.mp4 \
  --output input_upscaled_maxq.mp4 \
  --scale 2
```

Defaults under `max_quality` profile:

- model: `realesrgan-x4plus`
- TTA: enabled
- temporal filter: `strong`
- x264 preset: `veryslow`
- CRF: `14`

## Flow 2: Max Quality with Runtime Guardrail

Use this when you still want max quality, but want automatic fallback if runtime is projected to exceed a limit.

```bash
./.venv/bin/python upscale_video.py \
  input.mp4 \
  --output input_upscaled_guarded.mp4 \
  --scale 2 \
  --runtime-guardrail-hours 72 \
  --runtime-sample-frames 12
```

How it works:

- The tool upscales a sample of frames and projects full runtime.
- If projected runtime exceeds guardrail, it walks a quality ladder (for example: disable TTA, lower temporal filter strength, faster encode preset) until target is met or minimum rung is reached.
- Runtime estimates are machine-calibrated over time using EMA FPS history.

## Machine Calibration (Apple-Tuned Learning)

The runtime guardrail now learns your machine performance automatically.

- Calibration file default: `~/.cache/enhance-ai/runtime_calibration.json`
- First run samples runtime; later runs can use `source=calibration` estimates.
- Actual measured FPS updates calibration after each completed upscale stage.

Optional controls:

```bash
--calibration-file /path/to/runtime_calibration.json
--reset-calibration
```

## Resume / Retry Workflow

Use a persistent workspace to reuse extracted and upscaled frames:

```bash
./.venv/bin/python upscale_video.py \
  input.mp4 \
  --output input_upscaled_resume.mp4 \
  --work-dir /tmp/enhance_ai_workspace
```

Workspace safety:

- A manifest fingerprint is stored in workspace.
- If input/settings mismatch is detected, stale cached artifacts are invalidated automatically.

## Scene-Adaptive Ladder (Hard Mode)

Use this when you want per-scene quality adaptation under a runtime cap.

```bash
./.venv/bin/python upscale_video.py \
  input.mp4 \
  --output input_upscaled_scene_adaptive.mp4 \
  --scene-adaptive \
  --scene-threshold 0.35 \
  --scene-min-frames 24 \
  --scene-sample-frames 4 \
  --runtime-guardrail-hours 72
```

Behavior:

- Detects scene boundaries and builds merged scene ranges.
- Uses a quality-first ladder per scene (same ladder as global guardrail under `max_quality`).
- Computes texture score per scene and biases runtime budget toward texture-heavy scenes.
- Uses calibration-backed runtime estimates when available.
- Upscales scene-by-scene, renders scene chunks, concatenates, then remuxes audio.
- Auto-cleans scene chunk artifacts by default (`--keep-scene-chunks` disables cleanup).

### Plan-Only JSON Preflight

Use this to inspect strategy before running expensive upscales.

```bash
./.venv/bin/python upscale_video.py \
  input.mp4 \
  --scene-adaptive \
  --plan-only > plan.json
```

`--plan-only` emits structured JSON with scene ranges, texture scores, selected candidates, and projected runtimes.

## Common Flags

- `--profile custom` to opt out of quality profile defaults.
- `--disable-runtime-guardrail` to skip estimation.
- `--force` to re-extract/re-upscale everything.
- `--temporal-filter {none,light,medium,strong}` to control anti-flicker pass.
- `--cleanup-work-dir` to remove workdir artifacts after run.
- `--runtime-sample-frames N` to control estimator sample depth.
- `--scene-adaptive` to enable per-scene guardrail decisions.
- `--texture-priority` to bias per-scene budget toward detail-heavy scenes.
- `--keep-scene-chunks` to keep intermediate scene chunk files.
- `--plan-only` to print preflight JSON and skip upscale/reassembly.
