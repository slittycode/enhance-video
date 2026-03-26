import importlib
import os
import stat
import tempfile
import tomllib
from pathlib import Path
from unittest import mock

import toolchain


def test_enhance_video_package_is_importable():
    module = importlib.import_module("enhance_video")
    assert module is not None


def test_enhance_video_submodules_are_importable():
    submodules = [
        "enhance_video.cli",
        "enhance_video.toolchain",
        "enhance_video.validation",
        "enhance_video.ui",
        "enhance_video.telemetry",
        "enhance_video.media",
        "enhance_video.scene_adaptive",
        "enhance_video.pipeline",
    ]

    for module_name in submodules:
        module = importlib.import_module(module_name)
        assert module is not None


def test_pyproject_declares_new_scripts_and_extras():
    pyproject = Path(__file__).resolve().parent.parent / "pyproject.toml"
    payload = tomllib.loads(pyproject.read_text())

    scripts = payload["project"]["scripts"]
    optional_dependencies = payload["project"]["optional-dependencies"]

    assert scripts["enhance-video"]
    assert scripts["enhance-validate"]
    assert "dev" in optional_dependencies
    assert "validation" in optional_dependencies


def test_resolve_realesrgan_binary_uses_vendor_root_env_outside_repo():
    with tempfile.TemporaryDirectory() as temp_dir:
        vendor_root = Path(temp_dir) / "vendor"
        binary_name = toolchain.get_realesrgan_binary_name()
        binary_path = vendor_root / "realesrgan-ncnn-vulkan-v0.2.0-macos" / binary_name
        binary_path.parent.mkdir(parents=True, exist_ok=True)
        binary_path.write_text("#!/bin/sh\nexit 0\n")
        binary_path.chmod(binary_path.stat().st_mode | stat.S_IEXEC)

        with tempfile.TemporaryDirectory() as other_dir:
            with mock.patch.dict(os.environ, {"ENHANCE_AI_VENDOR_ROOT": str(vendor_root)}):
                with mock.patch("shutil.which", return_value=None):
                    with mock.patch("pathlib.Path.cwd", return_value=Path(other_dir)):
                        resolved = toolchain.resolve_realesrgan_binary(None)

    assert resolved == binary_path.resolve()


def test_get_default_calibration_path_has_no_filesystem_side_effects():
    with tempfile.TemporaryDirectory() as temp_dir:
        fake_home = Path(temp_dir) / "home"
        with mock.patch("pathlib.Path.home", return_value=fake_home):
            calibration_path = toolchain.get_default_calibration_path()
            assert calibration_path == fake_home / ".cache" / "enhance-ai" / "runtime_calibration.json"
            assert not calibration_path.parent.exists()
