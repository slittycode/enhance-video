"""Compatibility shim for legacy imports."""

from __future__ import annotations

import sys

from enhance_video import validation as _validation

sys.modules[__name__] = _validation

if __name__ == "__main__":
    raise SystemExit(_validation.main())
