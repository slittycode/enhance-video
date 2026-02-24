# Review Pack Summary: Model Selection & Constraints

## What Changed
- **CLI Precedence Guarantees**: Refactored `apply_quality_profile` so that explicitly passed flags like `--model` or `--temporal-filter` natively overwrite any and all defaults set by `--profile` (even via `-m` aliases).
- **Scale-Aware Resolution Fallbacks**: Created a `resolve_model` pass that correctly handles upscaling dimensions. When requesting `--scale 2`, the engine explicitly attempts to use specialized `x2plus` models (real-life and anime).
- **Missing Model Safety**: Added detection inside `resolve_model` so that if `x2plus` versions aren't available, the CLI prints a transparent warning and relies cleanly on the existing `x4plus` versions alongside output outscaling.
- **Auto-Feature Hook (`--type auto`)**: Created `analyze_video_type` using image analysis frameworks (CV2, NumPy). Samples frames and outputs a transparent logic text describing *why* a video was classed as anime or real-life (by gauging CV2 Laplacian texture variance against Canny edge density).
- **Extended Test Suite Coverage**: Re-configured `Toolchain` instantiation mocks inside `tests/test_model_selection.py`, creating 4 new passing test blocks that test precedence protection, model existence degradation, and synthetic CV2 array analysis triggering!
- **Documentation Standardization**: In `README.md` we showcased the specific CLI parameter set combining `--profile custom` and `--temporal-filter none` needed to reproduce the highest clarity result for anime clips, and cataloged the usage of `--type auto`.

## Key Invariants Guaranteed
1. **Explicit Flags Always Win**: Profile-injected variables can never overwrite explicitly passed tokens, guaranteeing consistent CLI operation matching user expectations.
2. **Deterministic Auto Detection**: Sampling is limited to short deterministic bounds (defaulting to 5 frames) avoiding unbounded runtime execution during heuristic detection hooks.
3. **No-Crash Fallback Dependencies**: If CV2/Numpy are missing from the pip library for any reason, the `analyze` hook catches the `ImportError` gracefully, logs a gentle warning, and guarantees execution continues using a safe "real-life" default fallback. Handled identical fail-safes around `ffmpeg` errors when extracting thumbnails!

## Known Limitations
- If a user doesn't have the explicit `{machine}x2plus` model `.param`/`.bin` files within their local models directory, `scale 2` commands will incur a minor computational penalty via the higher resolution `x4plus` model stepping downwards (although properly mitigated visually by the outscale routines).
- `--type auto` explicitly requires the `opencv-python` wrapper and `numpy` inside the pip matrix. We've written it currently as an optional import block to protect core application dependencies if they choose to skip it. 

## How to Reproduce "Best Looking Anime" Reference Behavior
Running the underlying model against stylized frames works optimally when temporal decimation/blur is bypassed entirely on fast-moving vector art.

```bash
python upscale_video.py dummy.mp4 -o out_anime.mp4 --type animation --profile custom --temporal-filter none
```
