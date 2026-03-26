# Architecture

## Runtime Surface
- `enhance_video.pipeline` remains the compatibility hub for the existing CLI behavior.
- `enhance_video.cli` owns argument parsing and quality-profile defaults.
- `enhance_video.toolchain` owns binary/model discovery, cache roots, and subprocess helpers.
- `enhance_video.validation` owns offline validation/report generation.
- `enhance_video.ui`, `enhance_video.telemetry`, `enhance_video.media`, and `enhance_video.scene_adaptive` provide package-level module boundaries for future extraction work.

## Resolution Order
1. Explicit CLI paths
2. `ENHANCE_AI_REALESRGAN_PATH` / `ENHANCE_AI_MODEL_PATH`
3. `ENHANCE_AI_VENDOR_ROOT`
4. `~/.cache/enhance-ai/vendor`
5. Repo-relative assets when running from a checkout
6. `PATH`

## Compatibility Strategy
- Existing imports from `upscale_video`, `cli`, `toolchain`, and `validate_upscale` continue to work via shims.
- New code should import from `enhance_video.*`.
