import argparse
import tempfile
import cv2
import numpy as np
import shutil
import pathlib
import subprocess
from unittest import mock

import pytest

from cli import parse_args, parse_cli_overrides, apply_quality_profile
from upscale_video import resolve_model, analyze_video_type
from toolchain import Toolchain

def test_explicit_override_precedence():
    # User applies max_quality but overrides temporal-filter and model explicitly
    argv = [
        "dummy.mp4",
        "--profile", "max_quality",
        "--temporal-filter", "none",
        "-m", "my-custom-model",
        "--type", "real-life"
    ]
    args = parse_args(argv)
    overrides = parse_cli_overrides(argv)
    apply_quality_profile(args, overrides)
    
    assert args.temporal_filter == "none", "Explicit --temporal-filter should win over profile"
    assert args.model == "my-custom-model", "Explicit -m should win over profile defaults"

@mock.patch("upscale_video.resolve_toolchain")
@mock.patch("pathlib.Path.exists")
def test_scale_aware_selection_anime_2x(mock_exists, mock_resolve_toolchain):
    # Setup mock paths 
    mock_resolve_toolchain.return_value = Toolchain(
        ffmpeg="/bin/ffmpeg",
        ffprobe="/bin/ffprobe",
        realesrgan_binary=pathlib.Path("/bin/realesrgan"),
        model_path=pathlib.Path("/tmp/models")
    )
    # Mock that the 2x model EXISTS
    mock_exists.return_value = True

    args = argparse.Namespace(
        input_video="dummy.mp4",
        model=None,
        type_alias="animation",
        scale=2,
        model_path=None
    )
    
    resolve_model(args)
    assert args.model == "realesrgan-x2plus-anime"

@mock.patch("upscale_video.resolve_toolchain")
@mock.patch("pathlib.Path.exists")
def test_scale_aware_selection_anime_2x_fallback(mock_exists, mock_resolve_toolchain):
    mock_resolve_toolchain.return_value = Toolchain(
        ffmpeg="/bin/ffmpeg",
        ffprobe="/bin/ffprobe",
        realesrgan_binary=pathlib.Path("/bin/realesrgan"),
        model_path=pathlib.Path("/tmp/models")
    )
    # Mock that models do NOT exist -> should fallback to 4x
    mock_exists.return_value = False

    args = argparse.Namespace(
        input_video="dummy.mp4",
        model=None,
        type_alias="animation",
        scale=2,
        model_path=None
    )
    
    resolve_model(args)
    assert args.model == "realesrgan-x4plus-anime"

@mock.patch("subprocess.run")
def test_analyze_video_type_anime(mock_run):
    # We will mock the ffmpeg subprocess.run to instead generate a synthetic anime-like frame in the temp dir.
    def mock_subprocess_run(cmd, check, capture_output):
        # find the out_pattern which is the last argument
        out_pattern = cmd[-1]
        temp_dir = pathlib.Path(out_pattern).parent
        
        # create a synthetic image that has very low variance but strong edges
        # flat gray square on flat black background
        img = np.zeros((100, 100), dtype=np.uint8)
        img[20:80, 20:80] = 128
        cv2.imwrite(str(temp_dir / "frame_000.png"), img)
        return mock.MagicMock()

    mock_run.side_effect = mock_subprocess_run
    
    res = analyze_video_type(pathlib.Path("dummy.mp4"), num_samples=1)
    assert res == "animation"

@mock.patch("subprocess.run")
def test_analyze_video_type_reallife(mock_run):
    def mock_subprocess_run(cmd, check, capture_output):
        out_pattern = cmd[-1]
        temp_dir = pathlib.Path(out_pattern).parent
        
        # create a synthetic image that has very high variance (random noise)
        img = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
        cv2.imwrite(str(temp_dir / "frame_000.png"), img)
        return mock.MagicMock()

    mock_run.side_effect = mock_subprocess_run
    
    res = analyze_video_type(pathlib.Path("dummy.mp4"), num_samples=1)
    assert res == "real-life"
