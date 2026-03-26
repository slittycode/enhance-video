# Enhance Video CLI (v1)

A production-grade, local Real-ESRGAN video upscaler tuned for Apple machines. This CLI provides a robust pipeline with a quality-first default profile, scene-adaptive processing, runtime guardrails, resume-safe workspaces, and a multi-stage quality fallback ladder.

## Prerequisites

- **Python 3.11+**
- **ffmpeg & ffprobe**: Must be installed and available in your system `PATH`.
  ```bash
  brew install ffmpeg
  ```
- **Real-ESRGAN (vulkan)**: Provide it via one of the supported resolution paths:
  - `--realesrgan-path`
  - `ENHANCE_AI_REALESRGAN_PATH`
  - `ENHANCE_AI_VENDOR_ROOT`
  - `~/.cache/enhance-ai/vendor`
  - `PATH`

## Installation

Install the package in editable mode from the project directory. This exposes `enhance-video` and `enhance-validate`.

```bash
cd path/to/project/enhance-ai
python -m pip install -e ".[dev,validation,telemetry]"
```

The default install provides the CLI and `rich`. The `validation` extra provides OpenCV, NumPy, scikit-image, and Pillow for offline validation.

## Quick Start

Upscale a real-life (or photographic) video to 4x resolution, prioritizing maximum quality (this is the default behavior):

```bash
enhance-video input.mp4 --output output.mp4
```

To upscale 2D animation or cartoons, which require a specialized AI model to keep edges crisp:

```bash
enhance-video input.mp4 --type animation
```

For the absolute best reproducible results on stylized/anime content (disabling the strong temporal filter that can sometimes smudge fast-moving animation):
```bash
enhance-video input.mp4 -o out_anime.mp4 --type animation --profile custom --temporal-filter none
```

Alternatively, let the CLI auto-detect whether the content is real-life or animation by analyzing the frames for edge density and texture variance:
```bash
enhance-video input.mp4 --type auto
```

## Core Workflows & Examples

### 1. Max Quality (Default)

Use this when image quality is the absolute priority. The default profile (`max_quality`) applies strict, high-fidelity settings.

```bash
enhance-video input.mp4 --scale 2 --type real-life
```

**Under the hood (defaults for `max_quality` profile):**
- **Model Type**: `realesrgan-x4plus` (real-life) or `realesrgan-x4plus-anime` (animation)
- **TTA (Test-Time Augmentation)**: Enabled
- **Temporal Filter**: `strong`
- **x264 Preset**: `veryslow`
- **CRF**: `14`

### 2. Guardrailed Upscaling (Auto-Fallback)

If you want maximum quality but need the upscaling to finish within a specific timeframe (e.g., overnight), use the runtime guardrail.

```bash
enhance-video input.mp4 \
  --runtime-guardrail-hours 12 \
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
enhance-video input.mp4 \
  --scene-adaptive \
  --scene-threshold 0.35 \
  --runtime-guardrail-hours 24
```

**Behavior:**
- Cuts the video into logical scenes.
- Computes a "texture score" for each scene. High-detail/texture scenes preserve more budget for maximum quality, while smooth/static scenes fallback faster to save time.
- Upscales scene chunks independently, concatenates them, and muxes the audio back in seamlessly.

### 4. Resume & Retry Workflows

If an upscale halts mid-way or crashes (e.g., power loss), you can resume exactly where it left off by persisting the workspace.

```bash
enhance-video input.mp4 \
  --work-dir /path/to/my_workspace
```

A fingerprint is maintained in the workspace. If you resume with identical settings, it skips already-extracted and already-upscaled frames. If settings change, the cache safely invalidates.

### 5. Preflight (Plan-Only) & Dry-Run

Inspect what the tool *will* do without executing the heavy lifting:

**Dry Run:** Prints the exact underlying `ffmpeg` and `realesrgan` commands that would be executed.
```bash
enhance-video input.mp4 --dry-run
```

**Plan-Only (JSON):** Ideal for scripting. Emits a structured JSON plan detailing scene boundaries, candidate selections, and runtime projections.
```bash
enhance-video input.mp4 --scene-adaptive --plan-only > plan.json
```

## Useful Flags

- `--profile custom` - Opt out of the `max_quality` defaults and supply your own parameter combinations.
- `--type {real-life,animation}` - Specify the video content type to use the correct AI model. Real-life uses the standard model, while animation uses a model tailored for flat artwork and sharp vector edges (default: `real-life`).
- `--scale {2,3,4}` - Upscale factor (default: `4`).
- `--disable-runtime-guardrail` - Skip estimation entirely and force the defined settings regardless of how long it takes.
- `--force` - Ignore cached frames and re-upscale everything.
- `--jobs` - Real-ESRGAN thread tuple (e.g., `2:2:2` for load:proc:save).
- `--cleanup-work-dir` - Automatically delete intermediate artifacts when the run completes.

## Environment Variables

- `ENHANCE_AI_REALESRGAN_PATH` - Explicit path to the Real-ESRGAN binary.
- `ENHANCE_AI_MODEL_PATH` - Explicit path to the model directory.
- `ENHANCE_AI_VENDOR_ROOT` - Root folder containing cached/extracted Real-ESRGAN assets.
- `ENHANCE_AI_CACHE_ROOT` - Override the default `~/.cache/enhance-ai` root.

## Tracing & Profiling (Optional)

The tool includes instrumentation via OpenTelemetry to trace runtime stage durations.

1. Install optional telemetry dependencies:
   ```bash
   pip install -e ".[telemetry]"
   ```
2. Open a local trace collector/viewer (e.g., AI Toolkit's tracing pane or an OTLP endpoint running on `localhost:4318`).
3. Run `enhance-video` normally. Spans like `extract_frames`, `run_upscale_batch`, and `reassemble_video` will automatically export to your viewer.
