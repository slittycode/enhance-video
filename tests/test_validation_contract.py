import builtins
import importlib.util
import sys
from pathlib import Path
from unittest import mock


def test_validation_module_imports_without_exiting_when_optional_deps_missing():
    module_path = (
        Path(__file__).resolve().parent.parent / "enhance_video" / "validation.py"
    )
    spec = importlib.util.spec_from_file_location("validation_under_test", module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    original_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name in {"cv2", "numpy"} or name.startswith("skimage"):
            raise ImportError(f"missing optional dependency: {name}")
        return original_import(name, globals, locals, fromlist, level)

    with mock.patch("builtins.__import__", side_effect=fake_import):
        assert spec.loader is not None
        spec.loader.exec_module(module)

    assert module.CV2_AVAILABLE is False
    assert callable(module.main)
