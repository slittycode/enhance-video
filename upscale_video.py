#!/usr/bin/env python3
"""
Video upscaler pipeline (vendored Real-ESRGAN — repo renamed to `enhance-ai`).

This script extracts frames, upscales them, and rebuilds a final video.
"""

from __future__ import annotations

import argparse
import contextlib
import functools
import json
import hashlib
import io
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

# ── Extracted modules (backward-compatible re-exports) ────────────────────────
from toolchain import (  # noqa: F401
    Toolchain,
    get_default_calibration_path,
    get_machine_id,
    get_realesrgan_binary_name,
    find_bundled_realesrgan_binary,
    progress_write,
    resolve_model_path,
    resolve_realesrgan_binary,
    resolve_toolchain,
    run_subprocess,
)
from cli import (  # noqa: F401
    QUALITY_PROFILES,
    SUPPORTED_PRESETS,
    SUPPORTED_SCALES,
    TEMPORAL_FILTER_LEVELS,
    apply_quality_profile,
    parse_args,
    parse_cli_overrides,
    resolve_output_path,
    validate_runtime_args,
)

# tracing dependencies added for lightweight performance profiling
try:
    from opentelemetry import trace
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
except ImportError:  # silence if not installed; tracing is optional
    trace = None

tracer = None

def init_tracing() -> None:
    """Configure OpenTelemetry tracer to export spans to localhost OTLP endpoint."""
    global tracer
    if trace is None:
        return
    if tracer is not None:
        return

    resource = Resource.create({"service.name": "enhance-ai-video-upscaler"})
    provider = TracerProvider(resource=resource)
    exporter = OTLPSpanExporter(endpoint="http://localhost:4318/v1/traces")
    provider.add_span_processor(BatchSpanProcessor(exporter))
    trace.set_tracer_provider(provider)
    tracer = trace.get_tracer(__name__)


def _traced(func):
    """Decorator that wraps a function call in a tracing span if tracing is enabled."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if tracer is None:
            init_tracing()
        if tracer is not None:
            with tracer.start_as_current_span(func.__name__):
                return func(*args, **kwargs)
        return func(*args, **kwargs)

    return wrapper


try:
    from tqdm import tqdm as _tqdm
except ImportError:
    def _tqdm(iterable=None, **_kwargs):
        if iterable is None:
            return []
        return iterable


DEFAULT_FPS = 30.0
DEFAULT_WIDTH = 1920
DEFAULT_HEIGHT = 1080

INPUT_FRAME_PATTERN = "frame_%08d.jpg"
INPUT_FRAME_GLOB = "frame_*.jpg"
OUTPUT_FRAME_PATTERN = "frame_%08d.png"
OUTPUT_FRAME_GLOB = "frame_*.png"






@dataclass(frozen=True)
class VideoInfo:
    framerate: float
    width: int
    height: int
    audio_codec: Optional[str]
    has_audio: bool
    duration_seconds: float


@dataclass(frozen=True)
class GuardrailCandidate:
    name: str
    tta: bool
    temporal_filter: str
    preset: str
    crf: int


@dataclass(frozen=True)
class RuntimeEstimate:
    sample_frames: int
    sample_fps: float
    projected_seconds: float
    candidate_name: str
    source: str = "sample"


@dataclass(frozen=True)
class ScenePlanEntry:
    scene_number: int
    start_frame: int
    end_frame: int
    frame_count: int
    texture_score: float
    budget_seconds: Optional[float]
    selected_candidate: GuardrailCandidate
    projected_seconds_by_name: dict[str, float]
    source_by_name: dict[str, str]


@dataclass(frozen=True)
class SceneAdaptiveResult:
    worst_candidate_index: int
    concatenated_video: Optional[Path]
    chunk_count: int





def parse_framerate(value: str) -> float:
    """Parse ffprobe framerate strings like 30000/1001 safely."""
    if not value:
        return DEFAULT_FPS

    try:
        if "/" in value:
            num, den = value.split("/", maxsplit=1)
            denominator = float(den)
            if denominator == 0:
                return DEFAULT_FPS
            framerate = float(num) / denominator
        else:
            framerate = float(value)
    except (TypeError, ValueError):
        return DEFAULT_FPS

    if framerate <= 0 or framerate > 240:
        return DEFAULT_FPS
    return framerate






def get_video_info(ffprobe_bin: str, input_video: Path) -> VideoInfo:
    """Read metadata with ffprobe and return parsed info."""
    cmd = [
        ffprobe_bin,
        "-v",
        "error",
        "-print_format",
        "json",
        "-show_streams",
        "-show_format",
        str(input_video),
    ]
    result = run_subprocess(cmd, capture_output=True)

    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Failed to parse ffprobe output: {exc}") from exc

    video_stream = None
    audio_stream = None
    for stream in payload.get("streams", []):
        stream_type = stream.get("codec_type")
        if stream_type == "video" and video_stream is None:
            video_stream = stream
        elif stream_type == "audio" and audio_stream is None:
            audio_stream = stream

    if video_stream is None:
        raise RuntimeError("No video stream found in input file.")

    framerate = parse_framerate(
        video_stream.get("avg_frame_rate") or video_stream.get("r_frame_rate", "")
    )

    width = int(video_stream.get("width", DEFAULT_WIDTH))
    height = int(video_stream.get("height", DEFAULT_HEIGHT))
    audio_codec = audio_stream.get("codec_name") if audio_stream else None
    duration_raw = (
        video_stream.get("duration")
        or payload.get("format", {}).get("duration")
        or "0"
    )
    try:
        duration_seconds = max(float(duration_raw), 0.0)
    except (TypeError, ValueError):
        duration_seconds = 0.0

    return VideoInfo(
        framerate=framerate,
        width=width,
        height=height,
        audio_codec=audio_codec,
        has_audio=audio_stream is not None,
        duration_seconds=duration_seconds,
    )


def extract_frames(ffmpeg_bin: str, input_video: Path, frames_dir: Path) -> int:
    """Extract input video frames into PNG files."""
    output_pattern = frames_dir / INPUT_FRAME_PATTERN
    cmd = [
        ffmpeg_bin,
        "-i",
        str(input_video),
        "-fps_mode",
        "passthrough",
        "-start_number",
        "1",
        "-q:v",
        "2",
        str(output_pattern),
        "-y",
        "-hide_banner",
        "-loglevel",
        "warning",
    ]
    run_subprocess(cmd)

    frame_count = len(list(frames_dir.glob(INPUT_FRAME_GLOB)))
    if frame_count == 0:
        raise RuntimeError("Frame extraction produced zero output frames.")
    return frame_count


def extract_audio(
    ffmpeg_bin: str,
    input_video: Path,
    temp_dir: Path,
    *,
    has_audio: bool,
) -> Optional[Path]:
    """Extract first audio stream to a temporary file."""
    if not has_audio:
        return None

    copy_target = temp_dir / "audio_track.mka"
    copy_cmd = [
        ffmpeg_bin,
        "-i",
        str(input_video),
        "-vn",
        "-map",
        "0:a:0",
        "-c:a",
        "copy",
        str(copy_target),
        "-y",
        "-hide_banner",
        "-loglevel",
        "warning",
    ]

    copy_result = run_subprocess(copy_cmd, check=False, capture_output=True)
    if copy_result.returncode == 0 and copy_target.exists():
        return copy_target

    # If stream-copy fails, transcode to AAC for predictable mux support.
    transcode_target = temp_dir / "audio_track.m4a"
    transcode_cmd = [
        ffmpeg_bin,
        "-i",
        str(input_video),
        "-vn",
        "-map",
        "0:a:0",
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        str(transcode_target),
        "-y",
        "-hide_banner",
        "-loglevel",
        "warning",
    ]
    transcode_result = run_subprocess(transcode_cmd, check=False, capture_output=True)
    if transcode_result.returncode == 0 and transcode_target.exists():
        return transcode_target

    progress_write("Warning: Audio extraction failed. Continuing without audio.")
    return None


def choose_upscale_mode(requested_mode: str, output_dir: Path, force: bool) -> str:
    """Choose batch or frame mode for current execution."""
    if requested_mode != "auto":
        return requested_mode
    if force:
        return "batch"
    has_existing = any(output_dir.glob(OUTPUT_FRAME_GLOB))
    return "frame" if has_existing else "batch"


def prepare_workspace(
    work_dir_arg: Optional[str],
    keep_temp: bool,
    cleanup_work_dir: bool,
) -> tuple[Path, bool]:
    """
    Prepare workspace root and determine whether cleanup should run at exit.

    If `work_dir_arg` is provided, workspace persists by default unless
    `cleanup_work_dir` is explicitly enabled.
    """
    if work_dir_arg:
        workspace_root = Path(work_dir_arg).expanduser().resolve()
        workspace_root.mkdir(parents=True, exist_ok=True)
        return workspace_root, cleanup_work_dir

    workspace_root = Path(tempfile.mkdtemp(prefix="video_upscale_"))
    return workspace_root, not keep_temp


def ensure_input_frames(
    ffmpeg_bin: str,
    input_video: Path,
    input_frames_dir: Path,
    force: bool,
) -> int:
    """Reuse cached frames when possible, otherwise extract frames."""
    existing_frames = sorted(input_frames_dir.glob(INPUT_FRAME_GLOB))
    if existing_frames and not force:
        print(f"Reusing {len(existing_frames)} existing extracted frame(s).")
        return len(existing_frames)

    if existing_frames and force:
        for frame in existing_frames:
            frame.unlink(missing_ok=True)

    return extract_frames(ffmpeg_bin, input_video, input_frames_dir)


def build_input_identity(input_video: Path) -> dict[str, object]:
    """Build stable identity data for input cache reuse checks."""
    stat_info = input_video.stat()
    return {
        "path": str(input_video),
        "size": stat_info.st_size,
        "mtime_ns": stat_info.st_mtime_ns,
    }


def build_workspace_fingerprint(
    args: argparse.Namespace,
    input_identity: dict[str, object],
) -> dict[str, object]:
    """Build a fingerprint for cached artifacts validity."""
    return {
        "input": input_identity,
        "profile": args.profile,
        "scale": args.scale,
        "type": args.type_alias,
        "model": args.model,
        "gpu": args.gpu,
        "tile_size": args.tile_size,
        "tta": bool(args.tta),
        "jobs": args.jobs,
        "scene_adaptive": bool(args.scene_adaptive),
        "scene_threshold": args.scene_threshold,
        "scene_min_frames": args.scene_min_frames,
        "scene_sample_frames": args.scene_sample_frames,
        "scene_budget_slack": args.scene_budget_slack,
        "texture_priority": args.texture_priority,
        "keep_scene_chunks": bool(args.keep_scene_chunks),
        "temporal_filter": args.temporal_filter,
        "preset": args.preset,
        "crf": args.crf,
        "audio_bitrate": args.audio_bitrate,
    }


def validate_workspace_cache(
    manifest_path: Path,
    current_fingerprint: dict[str, object],
) -> tuple[bool, str]:
    """Validate workspace cache manifest against current fingerprint."""
    if not manifest_path.exists():
        return False, "manifest missing"

    try:
        payload = json.loads(manifest_path.read_text())
    except json.JSONDecodeError:
        return False, "manifest unreadable"

    cached = payload.get("fingerprint")
    if cached != current_fingerprint:
        return False, "fingerprint mismatch"
    return True, "cache valid"


def read_workspace_manifest(manifest_path: Path) -> dict[str, object]:
    if not manifest_path.exists():
        return {}
    try:
        payload = json.loads(manifest_path.read_text())
    except json.JSONDecodeError:
        return {}
    if not isinstance(payload, dict):
        return {}
    return payload


def write_workspace_manifest(
    manifest_path: Path,
    current_fingerprint: dict[str, object],
) -> None:
    payload = {
        "fingerprint": current_fingerprint,
        "written_at": int(time.time()),
    }
    manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True))


def clear_output_cache(output_frames_dir: Path, workspace_root: Path) -> None:
    if output_frames_dir.exists():
        shutil.rmtree(output_frames_dir, ignore_errors=True)
    output_frames_dir.mkdir(parents=True, exist_ok=True)

    for stale_file in (
        workspace_root / "reassembled_raw.mp4",
        workspace_root / "runtime_estimate.mp4",
    ):
        stale_file.unlink(missing_ok=True)


def clear_full_workspace_cache(
    input_frames_dir: Path,
    output_frames_dir: Path,
    workspace_root: Path,
) -> None:
    for target in (input_frames_dir, output_frames_dir):
        if target.exists():
            shutil.rmtree(target, ignore_errors=True)
        target.mkdir(parents=True, exist_ok=True)

    for stale in workspace_root.glob("audio_track.*"):
        stale.unlink(missing_ok=True)

    clear_output_cache(output_frames_dir, workspace_root)


def load_calibration(calibration_path: Path) -> dict[str, dict]:
    if not calibration_path.exists():
        return {"entries": {}}

    try:
        payload = json.loads(calibration_path.read_text())
    except json.JSONDecodeError:
        return {"entries": {}}

    if not isinstance(payload, dict):
        return {"entries": {}}
    entries = payload.get("entries")
    if not isinstance(entries, dict):
        return {"entries": {}}
    return {"entries": entries}


def save_calibration(calibration_path: Path, calibration: dict[str, dict]) -> None:
    calibration_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"entries": calibration.get("entries", {})}
    calibration_path.write_text(json.dumps(payload, indent=2, sort_keys=True))


def update_calibration_entry(
    calibration: dict[str, dict],
    *,
    key: str,
    fps: float,
    source: str,
    alpha: float = 0.4,
) -> None:
    if fps <= 0:
        return

    entries = calibration.setdefault("entries", {})
    current = entries.get(key, {})
    prev_fps = float(current.get("ema_fps", fps))
    prev_samples = int(current.get("samples", 0))
    ema_fps = fps if prev_samples == 0 else (alpha * fps + (1.0 - alpha) * prev_fps)

    entries[key] = {
        "ema_fps": ema_fps,
        "samples": prev_samples + 1,
        "updated_at": int(time.time()),
        "last_source": source,
    }


def estimate_runtime_from_calibration(
    calibration: dict[str, dict],
    *,
    key: str,
    total_frames: int,
    candidate_name: str,
) -> Optional[RuntimeEstimate]:
    entry = calibration.get("entries", {}).get(key)
    if not entry:
        return None

    fps = float(entry.get("ema_fps", 0.0))
    if fps <= 0:
        return None

    projected_seconds = total_frames / fps
    return RuntimeEstimate(
        sample_frames=int(entry.get("samples", 0)),
        sample_fps=fps,
        projected_seconds=projected_seconds,
        candidate_name=candidate_name,
        source="calibration",
    )


def get_resolution_bucket(width: int, height: int) -> str:
    longest = max(width, height)
    if longest <= 720:
        return "sd"
    if longest <= 1280:
        return "hd"
    if longest <= 1920:
        return "fhd"
    if longest <= 2560:
        return "qhd"
    return "uhd"


def build_calibration_key(
    *,
    machine_id: str,
    args: argparse.Namespace,
    info: VideoInfo,
    candidate: GuardrailCandidate,
) -> str:
    parts = [
        machine_id,
        args.model,
        f"scale{args.scale}",
        get_resolution_bucket(info.width, info.height),
        f"tta{int(candidate.tta)}",
        f"tf_{candidate.temporal_filter}",
        f"preset_{candidate.preset}",
        f"crf_{candidate.crf}",
        f"tile_{args.tile_size}",
        f"jobs_{args.jobs or 'none'}",
    ]
    return "|".join(parts)


def build_scene_ranges(
    total_frames: int,
    boundaries: list[int],
    min_scene_frames: int,
) -> list[tuple[int, int]]:
    """Build scene ranges from boundary end-frame indices."""
    if total_frames <= 0:
        return []

    valid_boundaries = sorted(
        {
            boundary
            for boundary in boundaries
            if 1 <= boundary < total_frames
        }
    )
    ranges: list[tuple[int, int]] = []
    start = 1
    for boundary in valid_boundaries:
        if boundary >= start:
            ranges.append((start, boundary))
            start = boundary + 1
    ranges.append((start, total_frames))

    if min_scene_frames <= 1:
        return ranges

    merged: list[tuple[int, int]] = []
    for current in ranges:
        current_len = current[1] - current[0] + 1
        if not merged:
            merged.append(current)
            continue

        if current_len < min_scene_frames:
            previous = merged[-1]
            merged[-1] = (previous[0], current[1])
        else:
            merged.append(current)

    return merged


def compute_scene_texture_raw(scene_frames: list[Path], sample_size: int) -> float:
    """Approximate scene texture using sampled PNG frame sizes."""
    sampled_frames = select_sample_frames(scene_frames, sample_size)
    if not sampled_frames:
        return 0.0
    sizes = [
        float(frame.stat().st_size)
        for frame in sampled_frames
        if frame.exists() and frame.is_file()
    ]
    if not sizes:
        return 0.0
    return sum(sizes) / len(sizes)


def normalize_scores(values: list[float]) -> list[float]:
    """Normalize values into [0, 1], using 0.5 when all values are equal."""
    if not values:
        return []
    minimum = min(values)
    maximum = max(values)
    if maximum - minimum <= 1e-9:
        return [0.5 for _ in values]
    return [(value - minimum) / (maximum - minimum) for value in values]


def build_scene_texture_scores(
    frames: list[Path],
    scene_ranges: list[tuple[int, int]],
    sample_size: int,
) -> list[float]:
    raw_scores: list[float] = []
    for start, end in scene_ranges:
        scene_frames = frames[max(start - 1, 0):min(end, len(frames))]
        raw_scores.append(compute_scene_texture_raw(scene_frames, sample_size))
    return normalize_scores(raw_scores)


def get_texture_weight(texture_score: float, texture_priority: float) -> float:
    """Return texture weight multiplier used for budget sharing."""
    clamped_score = max(0.0, min(1.0, texture_score))
    priority = max(0.0, texture_priority)
    weight = 1.0 + (clamped_score - 0.5) * priority
    return max(0.2, weight)


def select_scene_candidate_by_projection(
    candidates: list[GuardrailCandidate],
    *,
    projected_seconds_by_name: dict[str, float],
    budget_seconds: float,
    texture_score: float = 0.5,
    texture_priority: float = 0.0,
) -> GuardrailCandidate:
    """Select first candidate whose projected runtime fits scene budget."""
    if not candidates:
        raise ValueError("No candidates provided for scene selection.")

    effective_budget = budget_seconds * get_texture_weight(texture_score, texture_priority)
    for candidate in candidates:
        projected = projected_seconds_by_name.get(candidate.name)
        if projected is None:
            continue
        if projected <= effective_budget:
            return candidate
    return candidates[-1]


def detect_scene_boundaries(
    ffmpeg_bin: str,
    input_video: Path,
    *,
    scene_threshold: float,
    framerate: float,
    total_frames: int,
) -> list[int]:
    """Detect scene boundary end-frames using ffmpeg scene score metadata."""
    cmd = [
        ffmpeg_bin,
        "-i",
        str(input_video),
        "-vf",
        f"select='gt(scene,{scene_threshold})',showinfo",
        "-f",
        "null",
        "-",
    ]
    result = run_subprocess(cmd, check=False, capture_output=True)
    if result.returncode not in (0, 255):  # ffmpeg sometimes emits 255 on benign exit paths
        return []

    pts_pattern = re.compile(r"pts_time:\s*([0-9.]+)")
    boundary_ends: set[int] = set()

    for line in (result.stderr or "").splitlines():
        if "showinfo" not in line or "pts_time:" not in line:
            continue
        match = pts_pattern.search(line)
        if not match:
            continue
        try:
            pts_time = float(match.group(1))
        except ValueError:
            continue
        start_frame = max(1, min(total_frames, int(round(pts_time * framerate)) + 1))
        boundary = start_frame - 1
        if 1 <= boundary < total_frames:
            boundary_ends.add(boundary)

    return sorted(boundary_ends)


def build_guardrail_candidates(args: argparse.Namespace) -> list[GuardrailCandidate]:
    """Build ordered fallback ladder for runtime guardrail adjustments."""
    candidates = [
        GuardrailCandidate(
            name="full_quality",
            tta=bool(args.tta),
            temporal_filter=args.temporal_filter,
            preset=args.preset,
            crf=args.crf,
        ),
        GuardrailCandidate(
            name="disable_tta",
            tta=False,
            temporal_filter="strong",
            preset="veryslow",
            crf=14,
        ),
        GuardrailCandidate(
            name="temporal_medium",
            tta=False,
            temporal_filter="medium",
            preset="veryslow",
            crf=14,
        ),
        GuardrailCandidate(
            name="slower_encode",
            tta=False,
            temporal_filter="medium",
            preset="slower",
            crf=14,
        ),
        GuardrailCandidate(
            name="slow_encode",
            tta=False,
            temporal_filter="medium",
            preset="slow",
            crf=15,
        ),
        GuardrailCandidate(
            name="light_temporal",
            tta=False,
            temporal_filter="light",
            preset="slow",
            crf=16,
        ),
        GuardrailCandidate(
            name="last_resort",
            tta=False,
            temporal_filter="none",
            preset="medium",
            crf=17,
        ),
    ]

    deduped: list[GuardrailCandidate] = []
    seen = set()
    for candidate in candidates:
        key = (
            candidate.tta,
            candidate.temporal_filter,
            candidate.preset,
            candidate.crf,
        )
        if key in seen:
            continue
        deduped.append(candidate)
        seen.add(key)
    return deduped


def build_scene_adaptive_candidates(args: argparse.Namespace) -> list[GuardrailCandidate]:
    """Scene-adaptive ladder with quality-first fallbacks."""
    if args.profile == "max_quality":
        return build_guardrail_candidates(args)

    candidates = [
        GuardrailCandidate(
            name="current_settings",
            tta=bool(args.tta),
            temporal_filter=args.temporal_filter,
            preset=args.preset,
            crf=args.crf,
        )
    ]
    if args.tta:
        candidates.append(
            GuardrailCandidate(
                name="disable_tta",
                tta=False,
                temporal_filter=args.temporal_filter,
                preset=args.preset,
                crf=args.crf,
            )
        )
    return candidates


def select_sample_frames(frames: list[Path], sample_size: int) -> list[Path]:
    if sample_size <= 0 or len(frames) <= sample_size:
        return frames

    if sample_size == 1:
        return [frames[len(frames) // 2]]

    step = (len(frames) - 1) / float(sample_size - 1)
    indices = sorted({round(index * step) for index in range(sample_size)})
    return [frames[index] for index in indices]


def estimate_candidate_runtime(
    toolchain: Toolchain,
    input_frames_dir: Path,
    *,
    total_frames: int,
    scale_factor: int,
    model_name: str,
    gpu_id: int,
    tile_size: int,
    jobs: Optional[str],
    model_path: Optional[Path],
    sample_size: int,
    candidate: GuardrailCandidate,
    workspace_root: Path,
) -> RuntimeEstimate:
    frames = sorted(input_frames_dir.glob(INPUT_FRAME_GLOB))
    return estimate_candidate_runtime_for_frames(
        toolchain,
        frames=frames,
        total_frames=total_frames,
        scale_factor=scale_factor,
        model_name=model_name,
        gpu_id=gpu_id,
        tile_size=tile_size,
        jobs=jobs,
        model_path=model_path,
        sample_size=sample_size,
        candidate=candidate,
        workspace_root=workspace_root,
    )


def estimate_candidate_runtime_for_frames(
    toolchain: Toolchain,
    *,
    frames: list[Path],
    total_frames: int,
    scale_factor: int,
    model_name: str,
    gpu_id: int,
    tile_size: int,
    jobs: Optional[str],
    model_path: Optional[Path],
    sample_size: int,
    candidate: GuardrailCandidate,
    workspace_root: Path,
) -> RuntimeEstimate:
    """Estimate runtime by upscaling sampled frames from the given frame list."""
    selected = select_sample_frames(frames, sample_size)
    if not selected:
        raise RuntimeError("Unable to estimate runtime; no frames available.")

    estimate_dir = workspace_root / "_runtime_estimate_frames"
    if estimate_dir.exists():
        shutil.rmtree(estimate_dir, ignore_errors=True)
    estimate_dir.mkdir(parents=True, exist_ok=True)

    start = time.time()
    completed = 0
    try:
        for frame in selected:
            output_frame = estimate_dir / frame.name
            cmd = build_realesrgan_command(
                toolchain.realesrgan_binary,
                frame,
                output_frame,
                scale_factor=scale_factor,
                model_name=model_name,
                gpu_id=gpu_id,
                tile_size=tile_size,
                tta=candidate.tta,
                model_path=model_path,
                jobs=jobs,
            )
            result = run_subprocess(cmd, check=False, capture_output=True)
            if result.returncode != 0:
                stderr = result.stderr.strip() if result.stderr else f"exit={result.returncode}"
                raise RuntimeError(f"Runtime estimation failed: {stderr}")
            completed += 1
    finally:
        shutil.rmtree(estimate_dir, ignore_errors=True)

    elapsed = max(time.time() - start, 1e-6)
    sample_fps = completed / elapsed
    projected_seconds = total_frames / sample_fps
    return RuntimeEstimate(
        sample_frames=completed,
        sample_fps=sample_fps,
        projected_seconds=projected_seconds,
        candidate_name=candidate.name,
        source="sample",
    )


def apply_guardrail_candidate(args: argparse.Namespace, candidate: GuardrailCandidate) -> None:
    args.tta = candidate.tta
    args.temporal_filter = candidate.temporal_filter
    args.preset = candidate.preset
    args.crf = candidate.crf

def build_realesrgan_command(
    realesrgan_binary: Path,
    input_path: Path,
    output_path: Path,
    *,
    scale_factor: int,
    model_name: str,
    gpu_id: int,
    tile_size: int,
    tta: bool,
    model_path: Optional[Path],
    jobs: Optional[str],
) -> list[str]:
    cmd = [
        str(realesrgan_binary),
        "-i",
        str(input_path),
        "-o",
        str(output_path),
        "-n",
        model_name,
        "-s",
        str(scale_factor),
        "-f",
        "png",
        "-g",
        str(gpu_id),
        "-t",
        str(tile_size),
    ]

    if model_path is not None:
        cmd.extend(["-m", str(model_path)])

    if jobs:
        cmd.extend(["-j", jobs])

    if tta:
        cmd.append("-x")

    return cmd


def fallback_resize_frame(
    ffmpeg_bin: str,
    input_frame: Path,
    output_frame: Path,
    target_width: int,
    target_height: int,
) -> None:
    """Produce dimension-safe fallback frame when AI upscale fails."""
    cmd = [
        ffmpeg_bin,
        "-i",
        str(input_frame),
        "-vf",
        f"scale={target_width}:{target_height}:flags=lanczos",
        "-frames:v",
        "1",
        str(output_frame),
        "-y",
        "-hide_banner",
        "-loglevel",
        "warning",
    ]
    run_subprocess(cmd)


def run_upscale_batch(
    toolchain: Toolchain,
    input_frames_dir: Path,
    output_frames_dir: Path,
    *,
    scale_factor: int,
    model_name: str,
    gpu_id: int,
    tile_size: int,
    tta: bool,
    jobs: Optional[str],
    dry_run: bool,
) -> None:
    # Build tile-size fallback ladder: requested → 512 → 256 → 128
    tile_attempts = [tile_size]
    if tile_size == 0:
        tile_attempts.extend([512, 256, 128])
    elif tile_size > 128:
        for fallback in (tile_size // 2, 128):
            if fallback not in tile_attempts and fallback >= 128:
                tile_attempts.append(fallback)

    for attempt_idx, attempt_tile in enumerate(tile_attempts):
        cmd = build_realesrgan_command(
            toolchain.realesrgan_binary,
            input_frames_dir,
            output_frames_dir,
            scale_factor=scale_factor,
            model_name=model_name,
            gpu_id=gpu_id,
            tile_size=attempt_tile,
            tta=tta,
            model_path=toolchain.model_path,
            jobs=jobs,
        )

        if dry_run:
            print(f"[DRY RUN] {' '.join(cmd)}")
            return

        result = run_subprocess(cmd, check=False, capture_output=True)
        if result.returncode == 0:
            if attempt_idx > 0:
                print(f"  Batch upscale succeeded with tile_size={attempt_tile}")
            return

        stderr = result.stderr.strip() if result.stderr else "no stderr"
        is_last = attempt_idx == len(tile_attempts) - 1
        if is_last:
            raise RuntimeError(f"Batch upscale failed: {stderr}")

        next_tile = tile_attempts[attempt_idx + 1]
        progress_write(
            f"Warning: Batch upscale failed (tile={attempt_tile}), "
            f"retrying with tile_size={next_tile}..."
        )
        time.sleep(2)


def run_upscale_frame_mode(
    toolchain: Toolchain,
    input_frames_dir: Path,
    output_frames_dir: Path,
    *,
    scale_factor: int,
    model_name: str,
    gpu_id: int,
    tile_size: int,
    tta: bool,
    jobs: Optional[str],
    force: bool,
    dry_run: bool,
    target_width: int,
    target_height: int,
) -> None:
    frames = sorted(input_frames_dir.glob(INPUT_FRAME_GLOB))
    if not frames:
        raise RuntimeError("No input frames found for upscaling.")

    failed_frames = 0
    skipped_frames = 0

    for frame in _tqdm(frames, desc="Upscaling", unit="frame"):
        output_frame = output_frames_dir / frame.name
        if not force and output_frame.exists() and output_frame.stat().st_size > 0:
            skipped_frames += 1
            continue

        cmd = build_realesrgan_command(
            toolchain.realesrgan_binary,
            frame,
            output_frame,
            scale_factor=scale_factor,
            model_name=model_name,
            gpu_id=gpu_id,
            tile_size=tile_size,
            tta=tta,
            model_path=toolchain.model_path,
            jobs=jobs,
        )

        if dry_run:
            progress_write(f"[DRY RUN] {' '.join(cmd)}")
            continue

        succeeded = False
        result = None
        for attempt in range(2):
            try:
                result = run_subprocess(cmd, check=False, capture_output=True, timeout=300)
            except subprocess.TimeoutExpired:
                progress_write(f"Warning: Upscale timed out for {frame.name} (attempt {attempt + 1})")
                if attempt == 0:
                    time.sleep(2)
                continue
            if result.returncode == 0 and output_frame.exists() and output_frame.stat().st_size > 0:
                succeeded = True
                break
            if attempt == 0:
                time.sleep(2)
        if succeeded:
            continue

        failed_frames += 1
        stderr = result.stderr.strip() if hasattr(result, "stderr") and result.stderr else "timed out or unknown error"
        progress_write(f"Warning: AI upscale failed for {frame.name}: {stderr}")
        fallback_resize_frame(
            toolchain.ffmpeg,
            frame,
            output_frame,
            target_width=target_width,
            target_height=target_height,
        )

    if skipped_frames:
        print(f"Skipped {skipped_frames} already-upscaled frame(s).")
    if failed_frames:
        print(f"Recovered {failed_frames} failed frame(s) with Lanczos fallback.")


def plan_scene_adaptive_strategy(
    toolchain: Toolchain,
    frames: list[Path],
    *,
    scene_ranges: list[tuple[int, int]],
    candidates: list[GuardrailCandidate],
    calibration: dict[str, dict],
    machine_id: str,
    args: argparse.Namespace,
    info: VideoInfo,
    workspace_root: Path,
    scene_sample_frames: int,
    guardrail_seconds: Optional[float],
    scene_budget_slack: float,
    texture_priority: float,
    allow_sampling: bool = True,
    update_calibration: bool = True,
) -> list[ScenePlanEntry]:
    if not frames:
        raise RuntimeError("No input frames found for scene planning.")
    if not candidates:
        raise ValueError("Scene-adaptive mode requires at least one candidate.")

    texture_scores = build_scene_texture_scores(
        frames=frames,
        scene_ranges=scene_ranges,
        sample_size=scene_sample_frames,
    )
    scene_frame_counts = [(end - start + 1) for start, end in scene_ranges]
    remaining_guardrail = guardrail_seconds
    estimated_fps_cache: dict[str, float] = {}
    planned: list[ScenePlanEntry] = []

    for scene_index, (start, end) in enumerate(scene_ranges):
        scene_number = scene_index + 1
        scene_frames = frames[max(start - 1, 0):min(end, len(frames))]
        if not scene_frames:
            continue
        scene_frame_count = len(scene_frames)
        texture_score = texture_scores[scene_index] if scene_index < len(texture_scores) else 0.5

        scene_budget: Optional[float] = None
        if remaining_guardrail is not None:
            remaining_weight = 0.0
            for remaining_index in range(scene_index, len(scene_ranges)):
                remaining_count = scene_frame_counts[remaining_index]
                remaining_texture = (
                    texture_scores[remaining_index]
                    if remaining_index < len(texture_scores)
                    else 0.5
                )
                remaining_weight += remaining_count * get_texture_weight(
                    remaining_texture,
                    texture_priority,
                )
            if remaining_weight > 0:
                scene_weight = scene_frame_count * get_texture_weight(
                    texture_score,
                    texture_priority,
                )
                proportional_budget = remaining_guardrail * (scene_weight / remaining_weight)
                scene_budget = max(0.0, proportional_budget * scene_budget_slack)
            else:
                scene_budget = max(0.0, remaining_guardrail)

        projected_by_candidate: dict[str, float] = {}
        source_by_candidate: dict[str, str] = {}
        if scene_budget is not None:
            for candidate in candidates:
                calibration_key = build_calibration_key(
                    machine_id=machine_id,
                    args=args,
                    info=info,
                    candidate=candidate,
                )
                estimate = estimate_runtime_from_calibration(
                    calibration,
                    key=calibration_key,
                    total_frames=scene_frame_count,
                    candidate_name=candidate.name,
                )
                if estimate is None:
                    cached_fps = estimated_fps_cache.get(candidate.name, 0.0)
                    if cached_fps > 0:
                        estimate = RuntimeEstimate(
                            sample_frames=0,
                            sample_fps=cached_fps,
                            projected_seconds=scene_frame_count / cached_fps,
                            candidate_name=candidate.name,
                            source="scene_cache",
                        )
                    elif allow_sampling:
                        estimate = estimate_candidate_runtime_for_frames(
                            toolchain,
                            frames=scene_frames,
                            total_frames=scene_frame_count,
                            scale_factor=args.scale,
                            model_name=args.model,
                            gpu_id=args.gpu,
                            tile_size=args.tile_size,
                            jobs=args.jobs,
                            model_path=toolchain.model_path,
                            sample_size=scene_sample_frames,
                            candidate=candidate,
                            workspace_root=workspace_root,
                        )
                        estimated_fps_cache[candidate.name] = estimate.sample_fps
                        if update_calibration:
                            update_calibration_entry(
                                calibration,
                                key=calibration_key,
                                fps=estimate.sample_fps,
                                source="sample",
                            )
                if estimate is not None:
                    projected_by_candidate[candidate.name] = estimate.projected_seconds
                    source_by_candidate[candidate.name] = estimate.source

        if scene_budget is None:
            selected_candidate = candidates[0]
        else:
            selected_candidate = select_scene_candidate_by_projection(
                candidates,
                projected_seconds_by_name=projected_by_candidate,
                budget_seconds=scene_budget,
                texture_score=texture_score,
                texture_priority=texture_priority,
            )

        selected_projection = projected_by_candidate.get(selected_candidate.name)
        if remaining_guardrail is not None and selected_projection is not None:
            remaining_guardrail = max(0.0, remaining_guardrail - selected_projection)

        planned.append(
            ScenePlanEntry(
                scene_number=scene_number,
                start_frame=start,
                end_frame=end,
                frame_count=scene_frame_count,
                texture_score=texture_score,
                budget_seconds=scene_budget,
                selected_candidate=selected_candidate,
                projected_seconds_by_name=projected_by_candidate,
                source_by_name=source_by_candidate,
            )
        )

    return planned


def render_scene_chunk(
    ffmpeg_bin: str,
    output_frames_dir: Path,
    chunk_path: Path,
    *,
    start_frame: int,
    frame_count: int,
    framerate: float,
    preset: str,
    crf: int,
    codec: str = "h264",
) -> None:
    """Render a scene chunk from upscaled frames."""
    if frame_count <= 0:
        raise ValueError("Scene chunk frame count must be > 0.")

    input_pattern = output_frames_dir / OUTPUT_FRAME_PATTERN
    cmd = [
        ffmpeg_bin,
        "-framerate",
        str(framerate),
        "-start_number",
        str(start_frame),
        "-i",
        str(input_pattern),
        "-frames:v",
        str(frame_count),
    ]
    cmd.extend(get_codec_flags(codec, preset, crf))
    cmd.extend([
        "-pix_fmt",
        "yuv420p",
        str(chunk_path),
        "-y",
        "-hide_banner",
        "-loglevel",
        "warning",
    ])
    run_subprocess(cmd)


def concat_scene_chunks(
    ffmpeg_bin: str,
    chunk_paths: list[Path],
    output_video: Path,
    *,
    preset: str,
    crf: int,
    codec: str = "h264",
) -> Path:
    """Concatenate scene chunk videos into a single video-only file."""
    if not chunk_paths:
        raise ValueError("No scene chunks provided for concat.")

    concat_list = output_video.parent / "scene_chunks_concat.txt"
    lines: list[str] = []
    for chunk in chunk_paths:
        escaped = str(chunk).replace("'", r"'\''")
        lines.append(f"file '{escaped}'")
    concat_list.write_text("\n".join(lines) + "\n")

    copy_cmd = [
        ffmpeg_bin,
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(concat_list),
        "-c",
        "copy",
        str(output_video),
        "-y",
        "-hide_banner",
        "-loglevel",
        "warning",
    ]
    copy_result = run_subprocess(copy_cmd, check=False, capture_output=True)
    if copy_result.returncode == 0 and output_video.exists():
        return concat_list

    # Some ffmpeg builds require re-encode when chunk stream params differ.
    reencode_cmd = [
        ffmpeg_bin,
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(concat_list),
    ]
    reencode_cmd.extend(get_codec_flags(codec, preset, crf))
    reencode_cmd.extend([
        "-pix_fmt",
        "yuv420p",
        str(output_video),
        "-y",
        "-hide_banner",
        "-loglevel",
        "warning",
    ])
    run_subprocess(reencode_cmd)
    return concat_list


def mux_audio_to_video(
    ffmpeg_bin: str,
    video_path: Path,
    output_video: Path,
    *,
    audio_path: Optional[Path],
    audio_bitrate: str,
) -> None:
    """Mux optional audio track into a pre-rendered video stream."""
    if audio_path and audio_path.exists():
        # Try stream-copy first (lossless, fast).
        copy_cmd = [
            ffmpeg_bin,
            "-i", str(video_path),
            "-i", str(audio_path),
            "-map", "0:v:0",
            "-map", "1:a:0",
            "-c:v", "copy",
            "-c:a", "copy",
            "-shortest",
            str(output_video),
            "-y", "-hide_banner", "-loglevel", "warning",
        ]
        copy_result = run_subprocess(copy_cmd, check=False, capture_output=True)
        if copy_result.returncode == 0 and output_video.exists():
            return

        # Fallback: transcode audio to AAC for container compatibility.
        transcode_cmd = [
            ffmpeg_bin,
            "-i", str(video_path),
            "-i", str(audio_path),
            "-map", "0:v:0",
            "-map", "1:a:0",
            "-c:v", "copy",
            "-c:a", "aac",
            "-b:a", audio_bitrate,
            "-shortest",
            str(output_video),
            "-y", "-hide_banner", "-loglevel", "warning",
        ]
        run_subprocess(transcode_cmd)
        return

    if video_path != output_video:
        shutil.move(str(video_path), str(output_video))


def run_upscale_scene_adaptive(
    toolchain: Toolchain,
    input_frames_dir: Path,
    output_frames_dir: Path,
    *,
    scale_factor: int,
    model_name: str,
    gpu_id: int,
    tile_size: int,
    jobs: Optional[str],
    force: bool,
    dry_run: bool,
    target_width: int,
    target_height: int,
    scene_plan: list[ScenePlanEntry],
    candidates: list[GuardrailCandidate],
    calibration: dict[str, dict],
    machine_id: str,
    args: argparse.Namespace,
    info: VideoInfo,
    workspace_root: Path,
    framerate: float,
    preset: str,
    crf: int,
    keep_scene_chunks: bool,
    codec: str = "h264",
) -> SceneAdaptiveResult:
    """Upscale frames scene-by-scene, render chunk videos, and concatenate."""
    frames = sorted(input_frames_dir.glob(INPUT_FRAME_GLOB))
    if not frames:
        raise RuntimeError("No input frames found for scene-adaptive upscaling.")
    if not scene_plan:
        raise ValueError("Scene-adaptive execution requires a non-empty scene plan.")

    total_skipped = 0
    total_failed = 0
    worst_candidate_index = 0
    candidate_index_by_name = {
        candidate.name: index for index, candidate in enumerate(candidates)
    }
    scene_chunks_dir = workspace_root / "scene_chunks"
    scene_chunks_dir.mkdir(parents=True, exist_ok=True)
    chunk_paths: list[Path] = []
    concat_manifest: Optional[Path] = None

    for plan_entry in scene_plan:
        scene_number = plan_entry.scene_number
        start = plan_entry.start_frame
        end = plan_entry.end_frame
        scene_frames = frames[max(start - 1, 0):min(end, len(frames))]
        if not scene_frames:
            continue
        scene_frame_count = len(scene_frames)
        selected_candidate = plan_entry.selected_candidate
        selected_index = candidate_index_by_name.get(selected_candidate.name, 0)
        worst_candidate_index = max(worst_candidate_index, selected_index)

        selected_projection = plan_entry.projected_seconds_by_name.get(selected_candidate.name)
        selected_source = plan_entry.source_by_name.get(selected_candidate.name, "default")
        if plan_entry.budget_seconds is None:
            budget_text = "n/a"
        else:
            budget_text = format_time(plan_entry.budget_seconds)

        if selected_projection is not None:
            projection_text = format_time(selected_projection)
            source_text = selected_source
        else:
            projection_text = "n/a"
            source_text = "default"

        print(
            f"  Scene {scene_number}/{len(scene_plan)} [{start}-{end}] "
            f"candidate={selected_candidate.name} "
            f"texture={plan_entry.texture_score:.2f} "
            f"budget={budget_text} projected={projection_text} source={source_text}"
        )

        scene_start = time.time()
        processed_scene = 0

        for frame in _tqdm(scene_frames, desc=f"Scene {scene_number}", unit="frame"):
            output_frame = output_frames_dir / frame.name
            if not force and output_frame.exists() and output_frame.stat().st_size > 0:
                total_skipped += 1
                continue

            cmd = build_realesrgan_command(
                toolchain.realesrgan_binary,
                frame,
                output_frame,
                scale_factor=scale_factor,
                model_name=model_name,
                gpu_id=gpu_id,
                tile_size=tile_size,
                tta=selected_candidate.tta,
                model_path=toolchain.model_path,
                jobs=jobs,
            )

            if dry_run:
                progress_write(f"[DRY RUN] {' '.join(cmd)}")
                continue

            result = run_subprocess(cmd, check=False, capture_output=True)
            if result.returncode == 0 and output_frame.exists() and output_frame.stat().st_size > 0:
                processed_scene += 1
                continue

            total_failed += 1
            stderr = result.stderr.strip() if result.stderr else f"exit={result.returncode}"
            progress_write(f"Warning: Scene upscale failed for {frame.name}: {stderr}")
            fallback_resize_frame(
                toolchain.ffmpeg,
                frame,
                output_frame,
                target_width=target_width,
                target_height=target_height,
            )

        scene_elapsed = max(time.time() - scene_start, 1e-6)
        if processed_scene > 0 and not dry_run:
            calibration_key = build_calibration_key(
                machine_id=machine_id,
                args=args,
                info=info,
                candidate=selected_candidate,
            )
            update_calibration_entry(
                calibration,
                key=calibration_key,
                fps=processed_scene / scene_elapsed,
                source="actual",
            )
        if not dry_run:
            chunk_path = scene_chunks_dir / f"scene_{scene_number:04d}.mp4"
            render_scene_chunk(
                toolchain.ffmpeg,
                output_frames_dir,
                chunk_path,
                start_frame=start,
                frame_count=scene_frame_count,
                framerate=framerate,
                preset=preset,
                crf=crf,
                codec=codec,
            )
            chunk_paths.append(chunk_path)

    if total_skipped:
        print(f"Skipped {total_skipped} already-upscaled frame(s).")
    if total_failed:
        print(f"Recovered {total_failed} failed frame(s) with Lanczos fallback.")

    if dry_run:
        return SceneAdaptiveResult(
            worst_candidate_index=worst_candidate_index,
            concatenated_video=None,
            chunk_count=0,
        )

    if not chunk_paths:
        raise RuntimeError("Scene-adaptive chunk rendering produced no output chunks.")

    concatenated_video = workspace_root / "scene_concat_raw.mp4"
    concat_manifest = concat_scene_chunks(
        toolchain.ffmpeg,
        chunk_paths,
        concatenated_video,
        preset=preset,
        crf=crf,
        codec=codec,
    )

    if not keep_scene_chunks:
        for chunk in chunk_paths:
            chunk.unlink(missing_ok=True)
        if concat_manifest:
            concat_manifest.unlink(missing_ok=True)
        try:
            scene_chunks_dir.rmdir()
        except OSError:
            pass

    return SceneAdaptiveResult(
        worst_candidate_index=worst_candidate_index,
        concatenated_video=concatenated_video,
        chunk_count=len(chunk_paths),
    )


def reassemble_video(
    ffmpeg_bin: str,
    frames_dir: Path,
    output_video: Path,
    *,
    framerate: float,
    audio_path: Optional[Path],
    crf: int,
    preset: str,
    audio_bitrate: str,
    codec: str = "h264",
) -> None:
    input_pattern = frames_dir / OUTPUT_FRAME_PATTERN
    cmd = [
        ffmpeg_bin,
        "-framerate",
        str(framerate),
        "-i",
        str(input_pattern),
    ]

    if audio_path and audio_path.exists():
        cmd.extend(["-i", str(audio_path)])

    cmd.extend(["-map", "0:v:0"])
    if audio_path and audio_path.exists():
        cmd.extend(["-map", "1:a:0"])

    cmd.extend(get_codec_flags(codec, preset, crf))
    cmd.extend(["-pix_fmt", "yuv420p"])

    if audio_path and audio_path.exists():
        cmd.extend(["-c:a", "copy", "-shortest"])

    cmd.extend(
        [
            str(output_video),
            "-y",
            "-hide_banner",
            "-loglevel",
            "warning",
        ]
    )

    run_subprocess(cmd)


def get_temporal_filter_expression(level: str) -> Optional[str]:
    """Return ffmpeg filter expression for temporal anti-flicker postprocessing."""
    if level == "none":
        return None
    if level == "light":
        return "atadenoise=0a=0.01:0b=0.02:1a=0.01:1b=0.02,unsharp=5:5:0.20:3:3:0.00"
    if level == "medium":
        return "atadenoise=0a=0.03:0b=0.06:1a=0.03:1b=0.06,unsharp=5:5:0.25:3:3:0.00"
    if level == "strong":
        return "atadenoise=0a=0.06:0b=0.12:1a=0.06:1b=0.12,unsharp=5:5:0.30:3:3:0.00"
    raise ValueError(f"Unsupported temporal filter level: {level}")


def apply_temporal_filter(
    ffmpeg_bin: str,
    input_video: Path,
    output_video: Path,
    *,
    level: str,
    preset: str,
    crf: int,
    codec: str = "h264",
) -> None:
    """Apply optional temporal denoise/sharpen pass to reduce flicker."""
    filter_expression = get_temporal_filter_expression(level)
    if filter_expression is None:
        if input_video != output_video:
            shutil.move(str(input_video), str(output_video))
        return

    cmd = [
        ffmpeg_bin,
        "-i",
        str(input_video),
        "-vf",
        filter_expression,
        "-map",
        "0:v:0",
        "-map",
        "0:a?",
    ]
    cmd.extend(get_codec_flags(codec, preset, crf))
    cmd.extend([
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "copy",
        str(output_video),
        "-y",
        "-hide_banner",
        "-loglevel",
        "warning",
    ])
    run_subprocess(cmd)


def format_time(seconds: float) -> str:
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    if hours:
        return f"{hours}h {minutes}m {secs:.1f}s"
    if minutes:
        return f"{minutes}m {secs:.1f}s"
    return f"{secs:.1f}s"






def check_disk_space(
    workspace_root: Path,
    info: VideoInfo,
    scale: int,
    frame_count: int,
) -> None:
    """Warn or fail if projected disk usage exceeds available space."""
    # JPEG input: ~0.15 bytes per pixel; PNG output: ~3 bytes per pixel
    input_bytes_per_frame = info.width * info.height * 0.15
    output_bytes_per_frame = (info.width * scale) * (info.height * scale) * 3.0
    projected_bytes = (input_bytes_per_frame + output_bytes_per_frame) * frame_count

    available = shutil.disk_usage(workspace_root).free
    if projected_bytes > available * 0.9:
        projected_gb = projected_bytes / (1024**3)
        available_gb = available / (1024**3)
        raise RuntimeError(
            f"Projected disk usage ({projected_gb:.1f} GB) exceeds 90% of "
            f"available space ({available_gb:.1f} GB). Use --work-dir to "
            f"point to a larger volume, or reduce --scale."
        )
    elif projected_bytes > available * 0.5:
        projected_gb = projected_bytes / (1024**3)
        available_gb = available / (1024**3)
        progress_write(
            f"Warning: Projected disk usage ({projected_gb:.1f} GB) is over "
            f"50% of available space ({available_gb:.1f} GB)."
        )


def get_codec_flags(codec: str, preset: str, crf: int) -> list[str]:
    """Return ffmpeg codec flags for the requested encoder."""
    if codec == "h264":
        return ["-c:v", "libx264", "-preset", preset, "-crf", str(crf)]
    if codec == "h265":
        return ["-c:v", "libx265", "-preset", preset, "-crf", str(crf)]
    if codec == "h265-hw":
        # Apple VideoToolbox hardware encoder; -q:v maps roughly to CRF
        return ["-c:v", "hevc_videotoolbox", "-q:v", str(max(1, crf))]
    raise ValueError(f"Unsupported codec: {codec}")


def serialize_scene_plan(scene_plan: list[ScenePlanEntry]) -> list[dict[str, object]]:
    payload: list[dict[str, object]] = []
    for entry in scene_plan:
        selected_name = entry.selected_candidate.name
        payload.append(
            {
                "scene_number": entry.scene_number,
                "start_frame": entry.start_frame,
                "end_frame": entry.end_frame,
                "frame_count": entry.frame_count,
                "texture_score": round(entry.texture_score, 6),
                "budget_seconds": entry.budget_seconds,
                "selected_candidate": selected_name,
                "selected_projected_seconds": entry.projected_seconds_by_name.get(selected_name),
                "selected_projection_source": entry.source_by_name.get(selected_name, "default"),
                "candidate_projected_seconds": entry.projected_seconds_by_name,
                "candidate_projection_sources": entry.source_by_name,
            }
        )
    return payload


def run_plan_only(args: argparse.Namespace) -> int:
    """Analyze strategy and print a clean JSON plan without running upscale."""
    validate_runtime_args(args)

    input_video = Path(args.input_video).expanduser().resolve()
    if not input_video.is_file():
        raise FileNotFoundError(f"Input video not found: {input_video}")

    output_video = resolve_output_path(input_video, args.output, args.scale)
    if output_video == input_video:
        raise ValueError("Output video path must be different from input video path.")

    toolchain = resolve_toolchain(args)
    machine_id = get_machine_id()
    calibration_path = Path(args.calibration_file).expanduser().resolve()
    calibration = {"entries": {}} if args.reset_calibration else load_calibration(calibration_path)

    workspace_root, should_cleanup_workspace = prepare_workspace(
        args.work_dir,
        args.keep_temp,
        args.cleanup_work_dir,
    )
    input_frames_dir = workspace_root / "input_frames"
    output_frames_dir = workspace_root / "upscaled_frames"
    input_frames_dir.mkdir(parents=True, exist_ok=True)
    output_frames_dir.mkdir(parents=True, exist_ok=True)

    try:
        info = get_video_info(toolchain.ffprobe, input_video)
        with contextlib.redirect_stdout(io.StringIO()):
            frame_count = ensure_input_frames(
                toolchain.ffmpeg,
                input_video,
                input_frames_dir,
                args.force,
            )

        guardrail_seconds: Optional[float] = None
        if not args.disable_runtime_guardrail:
            guardrail_seconds = args.runtime_guardrail_hours * 3600.0

        plan_payload: dict[str, object] = {
            "mode": "scene" if args.scene_adaptive else args.upscale_mode,
            "input_video": str(input_video),
            "output_video": str(output_video),
            "workspace": str(workspace_root),
            "machine_id": machine_id,
            "video": {
                "width": info.width,
                "height": info.height,
                "framerate": info.framerate,
                "duration_seconds": info.duration_seconds,
                "frame_count": frame_count,
                "has_audio": info.has_audio,
                "audio_codec": info.audio_codec,
            },
            "settings": {
                "profile": args.profile,
                "scale": args.scale,
                "model": args.model,
                "tta": bool(args.tta),
                "temporal_filter": args.temporal_filter,
                "preset": args.preset,
                "crf": args.crf,
                "scene_adaptive": bool(args.scene_adaptive),
                "runtime_guardrail_hours": (
                    None if args.disable_runtime_guardrail else args.runtime_guardrail_hours
                ),
                "scene_threshold": args.scene_threshold,
                "scene_min_frames": args.scene_min_frames,
                "scene_sample_frames": args.scene_sample_frames,
                "scene_budget_slack": args.scene_budget_slack,
                "texture_priority": args.texture_priority,
                "auto_clean_scene_chunks": not args.keep_scene_chunks,
            },
        }

        if args.scene_adaptive:
            scene_boundaries = detect_scene_boundaries(
                toolchain.ffmpeg,
                input_video,
                scene_threshold=args.scene_threshold,
                framerate=info.framerate,
                total_frames=frame_count,
            )
            scene_ranges = build_scene_ranges(
                total_frames=frame_count,
                boundaries=scene_boundaries,
                min_scene_frames=args.scene_min_frames,
            )
            if not scene_ranges:
                scene_ranges = [(1, frame_count)]

            frames = sorted(input_frames_dir.glob(INPUT_FRAME_GLOB))
            candidates = build_scene_adaptive_candidates(args)
            scene_plan = plan_scene_adaptive_strategy(
                toolchain,
                frames,
                scene_ranges=scene_ranges,
                candidates=candidates,
                calibration=calibration,
                machine_id=machine_id,
                args=args,
                info=info,
                workspace_root=workspace_root,
                scene_sample_frames=args.scene_sample_frames,
                guardrail_seconds=guardrail_seconds,
                scene_budget_slack=args.scene_budget_slack,
                texture_priority=args.texture_priority,
                allow_sampling=guardrail_seconds is not None,
                update_calibration=guardrail_seconds is not None,
            )
            scene_rows = serialize_scene_plan(scene_plan)
            selected_projections = [
                float(row["selected_projected_seconds"])
                for row in scene_rows
                if row.get("selected_projected_seconds") is not None
            ]
            total_projected_seconds = (
                sum(selected_projections) if selected_projections else None
            )
            plan_payload["scene"] = {
                "boundary_points": scene_boundaries,
                "scene_ranges": scene_ranges,
                "candidate_ladder": [candidate.name for candidate in candidates],
                "projected_total_seconds": total_projected_seconds,
                "projected_total_30m_seconds": (
                    (info.framerate * 60.0 * 30.0 / max(1e-6, frame_count))
                    * total_projected_seconds
                    if total_projected_seconds is not None and frame_count > 0
                    else None
                ),
                "entries": scene_rows,
            }
        else:
            if args.profile == "max_quality":
                candidates = build_guardrail_candidates(args)
            else:
                candidates = [
                    GuardrailCandidate(
                        name="current_settings",
                        tta=bool(args.tta),
                        temporal_filter=args.temporal_filter,
                        preset=args.preset,
                        crf=args.crf,
                    )
                ]
            candidate_rows: list[dict[str, object]] = []
            selected_name = candidates[-1].name
            for candidate in candidates:
                estimate: Optional[RuntimeEstimate] = None
                if guardrail_seconds is not None:
                    calibration_key = build_calibration_key(
                        machine_id=machine_id,
                        args=args,
                        info=info,
                        candidate=candidate,
                    )
                    estimate = estimate_runtime_from_calibration(
                        calibration,
                        key=calibration_key,
                        total_frames=frame_count,
                        candidate_name=candidate.name,
                    )
                    if estimate is None:
                        estimate = estimate_candidate_runtime(
                            toolchain,
                            input_frames_dir,
                            total_frames=frame_count,
                            scale_factor=args.scale,
                            model_name=args.model,
                            gpu_id=args.gpu,
                            tile_size=args.tile_size,
                            jobs=args.jobs,
                            model_path=toolchain.model_path,
                            sample_size=args.runtime_sample_frames,
                            candidate=candidate,
                            workspace_root=workspace_root,
                        )
                        update_calibration_entry(
                            calibration,
                            key=calibration_key,
                            fps=estimate.sample_fps,
                            source="sample",
                        )
                candidate_rows.append(
                    {
                        "name": candidate.name,
                        "tta": candidate.tta,
                        "temporal_filter": candidate.temporal_filter,
                        "preset": candidate.preset,
                        "crf": candidate.crf,
                        "projected_seconds": estimate.projected_seconds if estimate else None,
                        "source": estimate.source if estimate else "disabled",
                    }
                )
                if (
                    guardrail_seconds is not None
                    and estimate is not None
                    and estimate.projected_seconds <= guardrail_seconds
                ):
                    selected_name = candidate.name
                    break
            plan_payload["guardrail"] = {
                "enabled": guardrail_seconds is not None,
                "budget_seconds": guardrail_seconds,
                "selected_candidate": selected_name,
                "candidates": candidate_rows,
            }

        if guardrail_seconds is not None:
            save_calibration(calibration_path, calibration)

        print(json.dumps(plan_payload, indent=2, sort_keys=True))
        return 0
    finally:
        if should_cleanup_workspace:
            shutil.rmtree(workspace_root, ignore_errors=True)


@_traced
def run_pipeline(args: argparse.Namespace) -> int:
    validate_runtime_args(args)

    input_video = Path(args.input_video).expanduser().resolve()
    if not input_video.is_file():
        raise FileNotFoundError(f"Input video not found: {input_video}")

    output_video = resolve_output_path(input_video, args.output, args.scale)
    if output_video == input_video:
        raise ValueError("Output video path must be different from input video path.")

    output_video.parent.mkdir(parents=True, exist_ok=True)
    toolchain = resolve_toolchain(args)
    machine_id = get_machine_id()
    calibration_path = Path(args.calibration_file).expanduser().resolve()
    calibration = {"entries": {}} if args.reset_calibration else load_calibration(calibration_path)

    workspace_root, should_cleanup_workspace = prepare_workspace(
        args.work_dir,
        args.keep_temp,
        args.cleanup_work_dir,
    )
    input_frames_dir = workspace_root / "input_frames"
    output_frames_dir = workspace_root / "upscaled_frames"
    manifest_path = workspace_root / "workspace_manifest.json"
    input_frames_dir.mkdir(parents=True, exist_ok=True)
    output_frames_dir.mkdir(parents=True, exist_ok=True)

    input_identity = build_input_identity(input_video)
    manifest_payload = read_workspace_manifest(manifest_path)
    cached_fingerprint = manifest_payload.get("fingerprint", {})
    cached_input_identity = cached_fingerprint.get("input")
    if cached_input_identity and cached_input_identity != input_identity:
        print("Workspace input changed. Clearing stale cached artifacts.")
        clear_full_workspace_cache(input_frames_dir, output_frames_dir, workspace_root)
        manifest_path.unlink(missing_ok=True)

    print("\n" + "=" * 60)
    print("Video Upscaler - Real-ESRGAN")
    if args.dry_run:
        print("*** DRY RUN MODE ***")
    print("=" * 60)
    print(f"Input:  {input_video}")
    print(f"Output: {output_video}")
    print(f"Scale:  {args.scale}x")
    print(f"Model:  {args.model}")
    print(f"Profile: {args.profile}")
    print(f"GPU:    {args.gpu}")
    print(f"Mode:   {args.upscale_mode}")
    print(f"Scene adaptive: {'enabled' if args.scene_adaptive else 'disabled'}")
    if args.scene_adaptive:
        print(
            f"Scene params: threshold={args.scene_threshold:.2f}, "
            f"min_frames={args.scene_min_frames}, "
            f"sample_frames={args.scene_sample_frames}, "
            f"budget_slack={args.scene_budget_slack:.2f}, "
            f"texture_priority={args.texture_priority:.2f}, "
            f"auto_clean_chunks={'yes' if not args.keep_scene_chunks else 'no'}"
        )
    print(f"Temporal filter: {args.temporal_filter}")
    print(f"Guardrail: {'disabled' if args.disable_runtime_guardrail else f'{args.runtime_guardrail_hours:.1f}h'}")
    print(f"Machine ID: {machine_id}")
    print(f"Calibration: {calibration_path}")
    print(f"Workspace: {workspace_root}")
    print("=" * 60 + "\n")

    total_start = time.time()
    runtime_estimate: Optional[RuntimeEstimate] = None
    selected_candidate: Optional[GuardrailCandidate] = None
    selected_candidate_key: Optional[str] = None
    try:
        print("Analyzing video...")
        step_start = time.time()
        info = get_video_info(toolchain.ffprobe, input_video)
        print(f"  Resolution: {info.width}x{info.height}")
        print(f"  Framerate:  {info.framerate:.3f} fps")
        print(f"  Audio:      {info.audio_codec if info.audio_codec else 'None'}")
        if info.duration_seconds > 0:
            print(f"  Duration:   {info.duration_seconds:.1f}s")
        print(f"  Time: {format_time(time.time() - step_start)}\n")

        print("Preparing frames...")
        step_start = time.time()
        frame_count = ensure_input_frames(
            toolchain.ffmpeg,
            input_video,
            input_frames_dir,
            args.force,
        )
        print(f"  Frames: {frame_count}")
        print(f"  Time: {format_time(time.time() - step_start)}\n")

        existing_upscaled_count = len(list(output_frames_dir.glob(OUTPUT_FRAME_GLOB)))
        needs_upscale_work = args.force or existing_upscaled_count < frame_count
        if needs_upscale_work:
            check_disk_space(workspace_root, info, args.scale, frame_count)
        if args.scene_adaptive:
            print("Detecting scene boundaries...")
            scene_boundaries = detect_scene_boundaries(
                toolchain.ffmpeg,
                input_video,
                scene_threshold=args.scene_threshold,
                framerate=info.framerate,
                total_frames=frame_count,
            )
            scene_ranges = build_scene_ranges(
                total_frames=frame_count,
                boundaries=scene_boundaries,
                min_scene_frames=args.scene_min_frames,
            )
            if not scene_ranges:
                scene_ranges = [(1, frame_count)]
            print(
                f"  Detected {len(scene_ranges)} scene range(s) "
                f"from {len(scene_boundaries)} boundary point(s).\n"
            )
        else:
            scene_ranges = [(1, frame_count)]

        scene_guardrail_seconds: Optional[float] = None
        if args.scene_adaptive:
            if args.disable_runtime_guardrail:
                print("Runtime guardrail disabled by flag.\n")
            elif args.dry_run or not needs_upscale_work:
                print("Runtime guardrail skipped (no upscale work required or dry run).\n")
            else:
                scene_guardrail_seconds = args.runtime_guardrail_hours * 3600.0
                print(
                    "Runtime guardrail will be evaluated per scene during "
                    "scene-adaptive upscaling.\n"
                )
        else:
            if not args.disable_runtime_guardrail and not args.dry_run and needs_upscale_work:
                guardrail_seconds = args.runtime_guardrail_hours * 3600.0
                if args.profile == "max_quality":
                    candidates = build_guardrail_candidates(args)
                else:
                    candidates = [
                        GuardrailCandidate(
                            name="current_settings",
                            tta=bool(args.tta),
                            temporal_filter=args.temporal_filter,
                            preset=args.preset,
                            crf=args.crf,
                        )
                    ]

                print("Estimating runtime...")
                for index, candidate in enumerate(candidates):
                    calibration_key = build_calibration_key(
                        machine_id=machine_id,
                        args=args,
                        info=info,
                        candidate=candidate,
                    )
                    estimate = estimate_runtime_from_calibration(
                        calibration,
                        key=calibration_key,
                        total_frames=frame_count,
                        candidate_name=candidate.name,
                    )
                    if estimate is None:
                        estimate = estimate_candidate_runtime(
                            toolchain,
                            input_frames_dir,
                            total_frames=frame_count,
                            scale_factor=args.scale,
                            model_name=args.model,
                            gpu_id=args.gpu,
                            tile_size=args.tile_size,
                            jobs=args.jobs,
                            model_path=toolchain.model_path,
                            sample_size=args.runtime_sample_frames,
                            candidate=candidate,
                            workspace_root=workspace_root,
                        )
                        update_calibration_entry(
                            calibration,
                            key=calibration_key,
                            fps=estimate.sample_fps,
                            source="sample",
                        )
                    runtime_estimate = estimate
                    projected_30m_seconds = (info.framerate * 60.0 * 30.0) / estimate.sample_fps
                    print(
                        f"  Candidate {candidate.name}: "
                        f"{estimate.sample_fps:.2f} fps sampled, "
                        f"projected {format_time(estimate.projected_seconds)} "
                        f"(source={estimate.source}, 30m clip ~{format_time(projected_30m_seconds)})"
                    )

                    if estimate.projected_seconds <= guardrail_seconds:
                        apply_guardrail_candidate(args, candidate)
                        selected_candidate = candidate
                        selected_candidate_key = calibration_key
                        break

                    is_last = index == len(candidates) - 1
                    if is_last:
                        apply_guardrail_candidate(args, candidate)
                        selected_candidate = candidate
                        selected_candidate_key = calibration_key
                        print(
                            "  Warning: projected runtime still exceeds guardrail at "
                            "lowest quality-ladder setting."
                        )
                    elif args.profile != "max_quality":
                        apply_guardrail_candidate(args, candidate)
                        selected_candidate = candidate
                        selected_candidate_key = calibration_key
                        break

                if runtime_estimate and runtime_estimate.projected_seconds > guardrail_seconds:
                    print(
                        f"  Guardrail exceeded ({format_time(runtime_estimate.projected_seconds)} "
                        f"> {format_time(guardrail_seconds)})."
                    )
                elif runtime_estimate:
                    print("  Guardrail target satisfied.\n")
            elif args.disable_runtime_guardrail:
                print("Runtime guardrail disabled by flag.\n")
                selected_candidate = GuardrailCandidate(
                    name="guardrail_disabled",
                    tta=bool(args.tta),
                    temporal_filter=args.temporal_filter,
                    preset=args.preset,
                    crf=args.crf,
                )
                selected_candidate_key = build_calibration_key(
                    machine_id=machine_id,
                    args=args,
                    info=info,
                    candidate=selected_candidate,
                )
            else:
                print("Runtime guardrail skipped (no upscale work required or dry run).\n")
                selected_candidate = GuardrailCandidate(
                    name="guardrail_skipped",
                    tta=bool(args.tta),
                    temporal_filter=args.temporal_filter,
                    preset=args.preset,
                    crf=args.crf,
                )
                selected_candidate_key = build_calibration_key(
                    machine_id=machine_id,
                    args=args,
                    info=info,
                    candidate=selected_candidate,
                )

        scene_candidates: list[GuardrailCandidate] = []
        scene_plan: list[ScenePlanEntry] = []
        if args.scene_adaptive:
            scene_candidates = build_scene_adaptive_candidates(args)
            if len(scene_candidates) > 1:
                print(
                    f"  Scene-adaptive ladder: {', '.join(candidate.name for candidate in scene_candidates)}"
                )
            planning_frames = sorted(input_frames_dir.glob(INPUT_FRAME_GLOB))
            allow_scene_sampling = (
                scene_guardrail_seconds is not None and not args.dry_run and needs_upscale_work
            )
            scene_plan = plan_scene_adaptive_strategy(
                toolchain,
                planning_frames,
                scene_ranges=scene_ranges,
                candidates=scene_candidates,
                calibration=calibration,
                machine_id=machine_id,
                args=args,
                info=info,
                workspace_root=workspace_root,
                scene_sample_frames=args.scene_sample_frames,
                guardrail_seconds=scene_guardrail_seconds,
                scene_budget_slack=args.scene_budget_slack,
                texture_priority=args.texture_priority,
                allow_sampling=allow_scene_sampling,
                update_calibration=allow_scene_sampling,
            )

        current_fingerprint = build_workspace_fingerprint(args, input_identity)
        cache_valid, cache_reason = validate_workspace_cache(manifest_path, current_fingerprint)
        if not cache_valid:
            if cache_reason == "fingerprint mismatch":
                print("Workspace settings changed. Clearing stale upscaled frame cache.")
            clear_output_cache(output_frames_dir, workspace_root)
        else:
            print("Workspace cache fingerprint validated.\n")

        print("Extracting audio...")
        step_start = time.time()
        cached_audio = sorted(workspace_root.glob("audio_track.*"))
        if info.has_audio and cached_audio and not args.force:
            audio_path = cached_audio[0]
            print(f"  Reusing cached audio track: {audio_path.name}")
        else:
            for stale_audio in cached_audio:
                stale_audio.unlink(missing_ok=True)
            audio_path = extract_audio(
                toolchain.ffmpeg,
                input_video,
                workspace_root,
                has_audio=info.has_audio,
            )
        print(f"  Time: {format_time(time.time() - step_start)}\n")

        mode = choose_upscale_mode(args.upscale_mode, output_frames_dir, args.force)
        if args.scene_adaptive:
            mode = "scene"
        target_width = info.width * args.scale
        target_height = info.height * args.scale

        print(f"Upscaling frames ({mode} mode)...")
        step_start = time.time()
        scene_result: Optional[SceneAdaptiveResult] = None
        if mode == "scene":
            scene_result = run_upscale_scene_adaptive(
                toolchain,
                input_frames_dir,
                output_frames_dir,
                scale_factor=args.scale,
                model_name=args.model,
                gpu_id=args.gpu,
                tile_size=args.tile_size,
                jobs=args.jobs,
                force=args.force,
                dry_run=args.dry_run,
                target_width=target_width,
                target_height=target_height,
                scene_plan=scene_plan,
                candidates=scene_candidates,
                calibration=calibration,
                machine_id=machine_id,
                args=args,
                info=info,
                workspace_root=workspace_root,
                framerate=info.framerate,
                preset=args.preset,
                crf=args.crf,
                keep_scene_chunks=args.keep_scene_chunks,
                codec=args.codec,
            )
            if scene_result.worst_candidate_index > 0:
                print(
                    f"  Scene-adaptive selected lower rung "
                    f"'{scene_candidates[scene_result.worst_candidate_index].name}' "
                    "for at least one scene."
                )
            if not args.keep_scene_chunks and scene_result.chunk_count > 0:
                print("  Auto-cleaned intermediate scene chunk artifacts.")
        else:
            if mode == "batch":
                try:
                    run_upscale_batch(
                        toolchain,
                        input_frames_dir,
                        output_frames_dir,
                        scale_factor=args.scale,
                        model_name=args.model,
                        gpu_id=args.gpu,
                        tile_size=args.tile_size,
                        tta=args.tta,
                        jobs=args.jobs,
                        dry_run=args.dry_run,
                    )
                except RuntimeError:
                    if args.upscale_mode == "batch":
                        raise
                    print("Batch mode failed; falling back to frame mode.")
                    mode = "frame"
                    run_upscale_frame_mode(
                        toolchain,
                        input_frames_dir,
                        output_frames_dir,
                        scale_factor=args.scale,
                        model_name=args.model,
                        gpu_id=args.gpu,
                        tile_size=args.tile_size,
                        tta=args.tta,
                        jobs=args.jobs,
                        force=args.force,
                        dry_run=args.dry_run,
                        target_width=target_width,
                        target_height=target_height,
                    )
            if mode == "frame":
                run_upscale_frame_mode(
                    toolchain,
                    input_frames_dir,
                    output_frames_dir,
                    scale_factor=args.scale,
                    model_name=args.model,
                    gpu_id=args.gpu,
                    tile_size=args.tile_size,
                    tta=args.tta,
                    jobs=args.jobs,
                    force=args.force,
                    dry_run=args.dry_run,
                    target_width=target_width,
                    target_height=target_height,
                )
        upscale_elapsed = time.time() - step_start
        fps_processed = frame_count / upscale_elapsed if upscale_elapsed > 0 else 0.0
        print(f"  Time: {format_time(upscale_elapsed)} ({fps_processed:.2f} frames/sec)\n")
        if selected_candidate_key and fps_processed > 0:
            update_calibration_entry(
                calibration,
                key=selected_candidate_key,
                fps=fps_processed,
                source="actual",
            )

        if args.dry_run:
            print("Dry run complete. Skipping video reassembly.")
            return 0

        print("Reassembling output video...")
        step_start = time.time()
        if mode == "scene":
            if scene_result is None or scene_result.concatenated_video is None:
                raise RuntimeError("Scene-adaptive output video was not produced.")
            video_for_mux = scene_result.concatenated_video
            if args.temporal_filter != "none":
                print("Applying temporal filter...")
                filtered_video = workspace_root / "scene_filtered_raw.mp4"
                apply_temporal_filter(
                    toolchain.ffmpeg,
                    video_for_mux,
                    filtered_video,
                    level=args.temporal_filter,
                    preset=args.preset,
                    crf=args.crf,
                    codec=args.codec,
                )
                video_for_mux = filtered_video
            mux_audio_to_video(
                toolchain.ffmpeg,
                video_for_mux,
                output_video,
                audio_path=audio_path,
                audio_bitrate=args.audio_bitrate,
            )
        else:
            raw_output_video = output_video
            if args.temporal_filter != "none":
                raw_output_video = workspace_root / "reassembled_raw.mp4"
            reassemble_video(
                toolchain.ffmpeg,
                output_frames_dir,
                raw_output_video,
                framerate=info.framerate,
                audio_path=audio_path,
                crf=args.crf,
                preset=args.preset,
                audio_bitrate=args.audio_bitrate,
                codec=args.codec,
            )
            if args.temporal_filter != "none":
                print("Applying temporal filter...")
                apply_temporal_filter(
                    toolchain.ffmpeg,
                    raw_output_video,
                    output_video,
                    level=args.temporal_filter,
                    preset=args.preset,
                    crf=args.crf,
                    codec=args.codec,
                )
        print(f"  Time: {format_time(time.time() - step_start)}\n")

        total_elapsed = time.time() - total_start
        print("=" * 60)
        print("Complete!")
        print(f"Total time: {format_time(total_elapsed)}")
        print(f"Output: {output_video}")
        if output_video.exists():
            output_size_mb = output_video.stat().st_size / (1024 * 1024)
            print(f"Output size: {output_size_mb:.1f} MB")
        if runtime_estimate:
            print(
                f"Projected runtime ({runtime_estimate.candidate_name}): "
                f"{format_time(runtime_estimate.projected_seconds)}"
            )
        print("=" * 60 + "\n")
        write_workspace_manifest(manifest_path, current_fingerprint)
        save_calibration(calibration_path, calibration)
        return 0
    finally:
        if should_cleanup_workspace:
            shutil.rmtree(workspace_root, ignore_errors=True)
        else:
            print(f"Workspace kept at: {workspace_root}")


@_traced
def main(argv: Optional[Sequence[str]] = None) -> int:
    raw_argv = list(argv) if argv is not None else sys.argv[1:]
    args = parse_args(raw_argv)

    # Map content type to underlying model unless explicitly overridden by hidden flag
    if not args.model:
        if args.type_alias == "animation":
            args.model = "realesrgan-x4plus-anime"
        else:
            args.model = "realesrgan-x4plus"

    cli_overrides = parse_cli_overrides(raw_argv)
    apply_quality_profile(args, cli_overrides)
    try:
        if args.plan_only:
            return run_plan_only(args)
        return run_pipeline(args)
    except KeyboardInterrupt:
        print("Interrupted by user.", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    # initialize tracing if OpenTelemetry is available
    init_tracing()
    raise SystemExit(main())
