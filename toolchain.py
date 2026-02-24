"""Toolchain: binary resolution, subprocess wrapper, and machine identification."""

from __future__ import annotations

import argparse
import hashlib
import os
import platform
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence


def get_default_calibration_path() -> Path:
    base = Path.home() / ".cache" / "enhance-ai"
    base.mkdir(parents=True, exist_ok=True)
    return base / "runtime_calibration.json"


@dataclass(frozen=True)
class Toolchain:
    ffmpeg: str
    ffprobe: str
    realesrgan_binary: Path
    model_path: Optional[Path]


def progress_write(message: str) -> None:
    """Write a progress message using tqdm if available, otherwise print."""
    try:
        from tqdm import tqdm

        tqdm.write(message)
    except ImportError:
        print(message)


def run_subprocess(
    cmd: Sequence[str],
    *,
    check: bool = True,
    capture_output: bool = False,
    timeout: Optional[float] = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(part) for part in cmd],
        check=check,
        capture_output=capture_output,
        text=True,
        timeout=timeout,
    )


def get_machine_id() -> str:
    raw = "|".join(
        [
            platform.system(),
            platform.machine(),
            platform.processor() or "unknown",
            platform.version(),
        ]
    )
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]
    return f"{platform.system().lower()}-{platform.machine().lower()}-{digest}"


def get_realesrgan_binary_name() -> str:
    """Return the expected Real-ESRGAN binary name for the current OS."""
    if platform.system().lower() == "windows":
        return "realesrgan-ncnn-vulkan.exe"
    return "realesrgan-ncnn-vulkan"


def find_bundled_realesrgan_binary(search_root: Path, binary_name: str) -> Optional[Path]:
    """Search the repository tree for a vendored Real-ESRGAN binary.

    Backward-compatible: after repository rename the vendored binary may live
    under `enhance-ai/` instead of `Real-ESRGAN-ncnn-vulkan/` — check both.
    """
    vendor_candidates = [
        search_root / "Real-ESRGAN-ncnn-vulkan",
        search_root / "enhance-ai",
    ]

    for vendor_root in vendor_candidates:
        if not vendor_root.exists():
            continue

        candidates = sorted(vendor_root.rglob(binary_name))
        for candidate in candidates:
            if not candidate.is_file():
                continue
            if platform.system().lower() == "windows":
                return candidate
            if os.access(candidate, os.X_OK):
                return candidate

    return None


def resolve_realesrgan_binary(
    custom_path: Optional[str],
    search_root: Optional[Path] = None,
) -> Path:
    """Resolve Real-ESRGAN binary from custom path, PATH, or vendored location."""
    if search_root is None:
        search_root = Path.cwd()

    if custom_path:
        candidate = Path(custom_path).expanduser().resolve()
        if not candidate.is_file():
            raise FileNotFoundError(f"Real-ESRGAN binary not found at: {candidate}")
        return candidate

    binary_name = get_realesrgan_binary_name()

    system_binary = shutil.which(binary_name)
    if system_binary:
        return Path(system_binary).resolve()

    bundled_binary = find_bundled_realesrgan_binary(search_root, binary_name)
    if bundled_binary:
        return bundled_binary.resolve()

    raise FileNotFoundError(
        "Unable to locate Real-ESRGAN binary. Install it in PATH or pass "
        "--realesrgan-path explicitly."
    )


def resolve_model_path(
    custom_model_path: Optional[str],
    realesrgan_binary: Path,
) -> Optional[Path]:
    """Resolve model directory from explicit value or binary-adjacent models folder."""
    if custom_model_path:
        model_dir = Path(custom_model_path).expanduser().resolve()
        if not model_dir.is_dir():
            raise FileNotFoundError(f"Model directory not found: {model_dir}")
        return model_dir

    sibling_models = realesrgan_binary.parent / "models"
    if sibling_models.is_dir():
        return sibling_models.resolve()
    return None


def resolve_toolchain(args: argparse.Namespace) -> Toolchain:
    """Resolve runtime binaries and raise clear dependency errors."""
    ffmpeg_bin = shutil.which("ffmpeg")
    ffprobe_bin = shutil.which("ffprobe")
    if not ffmpeg_bin or not ffprobe_bin:
        missing = []
        if not ffmpeg_bin:
            missing.append("ffmpeg")
        if not ffprobe_bin:
            missing.append("ffprobe")
        raise FileNotFoundError(
            f"Missing required dependency: {', '.join(missing)}. "
            "Install with Homebrew (macOS) or your system package manager."
        )

    realesrgan_binary = resolve_realesrgan_binary(args.realesrgan_path, Path.cwd())
    model_path = resolve_model_path(args.model_path, realesrgan_binary)

    return Toolchain(
        ffmpeg=ffmpeg_bin,
        ffprobe=ffprobe_bin,
        realesrgan_binary=realesrgan_binary,
        model_path=model_path,
    )
