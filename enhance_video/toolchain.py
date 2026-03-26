"""Toolchain: binary resolution, subprocess wrapper, and machine identification."""

from __future__ import annotations

import argparse
import hashlib
import os
import platform
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

REPO_VENDOR_ROOT_ENV = "ENHANCE_AI_VENDOR_ROOT"
REPO_BINARY_PATH_ENV = "ENHANCE_AI_REALESRGAN_PATH"
REPO_MODEL_PATH_ENV = "ENHANCE_AI_MODEL_PATH"
REPO_CACHE_ROOT_ENV = "ENHANCE_AI_CACHE_ROOT"


def get_default_cache_root() -> Path:
    cache_root = os.environ.get(REPO_CACHE_ROOT_ENV)
    if cache_root:
        return Path(cache_root).expanduser().resolve()
    return Path.home() / ".cache" / "enhance-ai"


def get_default_calibration_path() -> Path:
    return get_default_cache_root() / "runtime_calibration.json"


def get_default_vendor_root() -> Path:
    return get_default_cache_root() / "vendor"


@dataclass(frozen=True)
class Toolchain:
    ffmpeg: str
    ffprobe: str
    realesrgan_binary: Path
    model_path: Optional[Path]


def progress_write(message: str) -> None:
    """Write a progress message using rich if available, otherwise print."""
    try:
        from rich import print as rprint

        rprint(message)
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


def _iter_vendor_candidates(search_root: Path) -> list[Path]:
    return [
        search_root,
        search_root / "Real-ESRGAN-ncnn-vulkan",
        search_root / "enhance-ai",
    ]


def find_bundled_realesrgan_binary(search_root: Path, binary_name: str) -> Optional[Path]:
    """Search the repository tree for a vendored Real-ESRGAN binary.

    Backward-compatible: after repository rename the vendored binary may live
    under `enhance-ai/` instead of `Real-ESRGAN-ncnn-vulkan/` — check both.
    """
    for vendor_root in _iter_vendor_candidates(search_root):
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


def _get_repo_search_roots(search_root: Optional[Path]) -> list[Path]:
    roots: list[Path] = []

    if search_root is not None:
        roots.append(search_root.expanduser().resolve())

    roots.extend(
        [
            Path.cwd().resolve(),
            Path(__file__).resolve().parents[1],
        ]
    )

    unique_roots: list[Path] = []
    for root in roots:
        if root not in unique_roots:
            unique_roots.append(root)
    return unique_roots


def _resolve_binary_from_vendor_root(vendor_root: Path, binary_name: str) -> Optional[Path]:
    return find_bundled_realesrgan_binary(vendor_root.expanduser().resolve(), binary_name)


def resolve_realesrgan_binary(
    custom_path: Optional[str],
    search_root: Optional[Path] = None,
) -> Path:
    """Resolve Real-ESRGAN binary from custom path, PATH, or vendored location."""
    explicit_binary = custom_path or os.environ.get(REPO_BINARY_PATH_ENV)
    if explicit_binary:
        candidate = Path(explicit_binary).expanduser().resolve()
        if not candidate.is_file():
            raise FileNotFoundError(f"Real-ESRGAN binary not found at: {candidate}")
        return candidate

    binary_name = get_realesrgan_binary_name()

    vendor_root = os.environ.get(REPO_VENDOR_ROOT_ENV)
    if vendor_root:
        resolved = _resolve_binary_from_vendor_root(Path(vendor_root), binary_name)
        if resolved:
            return resolved.resolve()

    cached_binary = _resolve_binary_from_vendor_root(get_default_vendor_root(), binary_name)
    if cached_binary:
        return cached_binary.resolve()

    for repo_root in _get_repo_search_roots(search_root):
        bundled_binary = find_bundled_realesrgan_binary(repo_root, binary_name)
        if bundled_binary:
            return bundled_binary.resolve()

    system_binary = shutil.which(binary_name)
    if system_binary:
        return Path(system_binary).resolve()

    raise FileNotFoundError(
        "Unable to locate Real-ESRGAN binary. Install it in PATH or pass "
        "--realesrgan-path explicitly. You can also set "
        f"{REPO_BINARY_PATH_ENV} or {REPO_VENDOR_ROOT_ENV}."
    )


def resolve_model_path(
    custom_model_path: Optional[str],
    realesrgan_binary: Path,
) -> Optional[Path]:
    """Resolve model directory from explicit value or binary-adjacent models folder."""
    explicit_model_path = custom_model_path or os.environ.get(REPO_MODEL_PATH_ENV)
    if explicit_model_path:
        model_dir = Path(explicit_model_path).expanduser().resolve()
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
