"""Compatibility shim for legacy imports."""

from __future__ import annotations

import sys

from enhance_video import toolchain as _toolchain

sys.modules[__name__] = _toolchain
