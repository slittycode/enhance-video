"""Compatibility shim for legacy imports."""

from __future__ import annotations

import sys

from enhance_video import pipeline as _pipeline

sys.modules[__name__] = _pipeline

if __name__ == "__main__":
    raise SystemExit(_pipeline.main())
