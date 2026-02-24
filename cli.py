"""CLI: argument parsing, quality profiles, and runtime validation."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Sequence

from toolchain import get_default_calibration_path

# ── Constants ──────────────────────────────────────────────────────────────────

SUPPORTED_SCALES = (2, 3, 4)
TEMPORAL_FILTER_LEVELS = ("none", "light", "medium", "strong")
QUALITY_PROFILES = ("custom", "max_quality")
SUPPORTED_PRESETS = (
    "ultrafast",
    "superfast",
    "veryfast",
    "faster",
    "fast",
    "medium",
    "slow",
    "slower",
    "veryslow",
)


# ── Functions ──────────────────────────────────────────────────────────────────


def parse_cli_overrides(argv: Sequence[str]) -> set[str]:
    """Return canonical option names explicitly provided by the caller."""
    option_to_key = {
        "--profile": "profile",
        "--type": "type_alias",
        "-m": "model",
        "--tta": "tta",
        "--temporal-filter": "temporal_filter",
        "--preset": "preset",
        "--crf": "crf",
        "--upscale-mode": "upscale_mode",
        "--audio-bitrate": "audio_bitrate",
        "--runtime-guardrail-hours": "runtime_guardrail_hours",
        "--runtime-sample-frames": "runtime_sample_frames",
        "--disable-runtime-guardrail": "disable_runtime_guardrail",
    }

    overrides: set[str] = set()
    for token in argv:
        if not token.startswith("-"):
            continue
        option = token.split("=", maxsplit=1)[0]
        key = option_to_key.get(option)
        if key:
            overrides.add(key)
    return overrides


def apply_quality_profile(args: argparse.Namespace, cli_overrides: set[str]) -> None:
    """Apply quality profile defaults unless overridden by explicit flags."""
    if args.profile != "max_quality":
        return

    profile_defaults: dict[str, object] = {
        "tta": True,
        "temporal_filter": "strong",
        "preset": "veryslow",
        "crf": 14,
        "upscale_mode": "auto",
        "audio_bitrate": "256k",
    }

    if "model" not in cli_overrides and "type_alias" not in cli_overrides:
        # Default model logic when no explicit profile is requested
        profile_defaults["model"] = "realesrgan-x4plus"

    for key, value in profile_defaults.items():
        if key not in cli_overrides:
            setattr(args, key, value)

    if "runtime_guardrail_hours" not in cli_overrides and args.runtime_guardrail_hours <= 0:
        args.runtime_guardrail_hours = 72.0


def resolve_output_path(input_video: Path, output_arg: Optional[str], scale: int) -> Path:
    if output_arg:
        return Path(output_arg).expanduser().resolve()
    return (input_video.parent / f"{input_video.stem}_upscaled_{scale}x.mp4").resolve()


def validate_runtime_args(args: argparse.Namespace) -> None:
    if args.crf < 0 or args.crf > 51:
        raise ValueError("CRF must be between 0 and 51.")
    if args.tile_size < 0:
        raise ValueError("Tile size must be >= 0.")
    if args.runtime_guardrail_hours <= 0:
        raise ValueError("Runtime guardrail hours must be > 0.")
    if args.runtime_sample_frames <= 0:
        raise ValueError("Runtime sample frames must be > 0.")
    if not (0.0 < args.scene_threshold < 1.0):
        raise ValueError("Scene threshold must be between 0 and 1.")
    if args.scene_min_frames <= 0:
        raise ValueError("Scene min frames must be > 0.")
    if args.scene_sample_frames <= 0:
        raise ValueError("Scene sample frames must be > 0.")
    if args.scene_budget_slack <= 0:
        raise ValueError("Scene budget slack must be > 0.")
    if args.texture_priority < 0:
        raise ValueError("Texture priority must be >= 0.")
    calibration_target = Path(args.calibration_file).expanduser()
    if calibration_target.exists() and calibration_target.is_dir():
        raise ValueError("Calibration file path must be a file, not a directory.")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Upscale video using Real-ESRGAN (vendored in this repository)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("input_video", type=str, help="Path to input video")
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help="Output video path (default: <input>_upscaled_<scale>x.mp4)",
    )
    parser.add_argument(
        "-s",
        "--scale",
        type=int,
        default=4,
        choices=SUPPORTED_SCALES,
        help="Upscaling factor",
    )
    parser.add_argument(
        "--type",
        dest="type_alias",
        type=str,
        default="real-life",
        choices=("real-life", "animation"),
        help="Video content type (determines underlying AI model)",
    )
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default=None,
        help=argparse.SUPPRESS,  # Hidden advanced override
    )
    parser.add_argument(
        "--profile",
        type=str,
        choices=QUALITY_PROFILES,
        default="max_quality",
        help="Quality profile defaults",
    )
    parser.add_argument("-g", "--gpu", type=int, default=0, help="GPU device ID")
    parser.add_argument(
        "--realesrgan-path",
        type=str,
        default=None,
        help="Custom path to realesrgan-ncnn-vulkan binary",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Custom model directory path",
    )
    parser.add_argument(
        "-t",
        "--tile-size",
        type=int,
        default=0,
        help="Tile size (0 = auto)",
    )
    parser.add_argument("--tta", action="store_true", help="Enable test-time augmentation")
    parser.add_argument("--force", action="store_true", help="Re-upscale all frames")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing")
    parser.add_argument(
        "--plan-only",
        action="store_true",
        help="Analyze pipeline strategy and print clean JSON plan only",
    )
    parser.add_argument(
        "--work-dir",
        type=str,
        default=None,
        help="Reusable workspace directory for resume/retry workflows",
    )
    parser.add_argument(
        "--cleanup-work-dir",
        action="store_true",
        help="Remove --work-dir contents after run completes",
    )
    parser.add_argument(
        "--scene-adaptive",
        action="store_true",
        help="Enable scene-aware per-scene candidate selection",
    )
    parser.add_argument(
        "--scene-threshold",
        type=float,
        default=0.35,
        help="Scene detection threshold (0-1)",
    )
    parser.add_argument(
        "--scene-min-frames",
        type=int,
        default=24,
        help="Minimum frames per scene after merge",
    )
    parser.add_argument(
        "--scene-sample-frames",
        type=int,
        default=4,
        help="Per-scene sample frames for adaptive candidate estimation",
    )
    parser.add_argument(
        "--scene-budget-slack",
        type=float,
        default=1.10,
        help="Scene budget multiplier (higher means less aggressive fallback)",
    )
    parser.add_argument(
        "--texture-priority",
        type=float,
        default=0.85,
        help="Texture weighting strength for scene budget allocation (0 disables)",
    )
    parser.add_argument(
        "--keep-scene-chunks",
        action="store_true",
        help="Keep intermediate scene chunk artifacts after concatenation",
    )
    parser.add_argument(
        "--upscale-mode",
        type=str,
        choices=("auto", "batch", "frame"),
        default="auto",
        help="Upscale mode: batch is fastest, frame supports resume",
    )
    parser.add_argument(
        "--jobs",
        type=str,
        default=None,
        help="Real-ESRGAN thread tuple (load:proc:save), for example 2:2:2",
    )
    parser.add_argument("--crf", type=int, default=18, help="x264 CRF (0-51)")
    parser.add_argument(
        "--preset",
        type=str,
        default="slow",
        choices=SUPPORTED_PRESETS,
        help="x264 preset",
    )
    parser.add_argument(
        "--audio-bitrate",
        type=str,
        default="192k",
        help="Audio bitrate for final AAC encode",
    )
    parser.add_argument(
        "--runtime-guardrail-hours",
        type=float,
        default=72.0,
        help="Maximum projected runtime before quality ladder fallback",
    )
    parser.add_argument(
        "--runtime-sample-frames",
        type=int,
        default=12,
        help="Number of frames to sample for runtime estimation",
    )
    parser.add_argument(
        "--disable-runtime-guardrail",
        action="store_true",
        help="Disable runtime estimation and fallback ladder",
    )
    parser.add_argument(
        "--calibration-file",
        type=str,
        default=str(get_default_calibration_path()),
        help="Path to machine runtime calibration JSON",
    )
    parser.add_argument(
        "--reset-calibration",
        action="store_true",
        help="Ignore existing calibration data for this run",
    )
    parser.add_argument(
        "--temporal-filter",
        type=str,
        choices=TEMPORAL_FILTER_LEVELS,
        default="none",
        help="Optional temporal anti-flicker post-process pass",
    )
    parser.add_argument("--keep-temp", action="store_true", help="Keep temporary workspace")
    parser.add_argument(
        "--codec",
        type=str,
        choices=("h264", "h265", "h265-hw"),
        default="h264",
        help="Output video codec. h265-hw uses Apple VideoToolbox hardware encoder.",
    )

    return parser.parse_args(argv)
