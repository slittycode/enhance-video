"""Compatibility shim for legacy imports."""

from __future__ import annotations

import sys

from enhance_video import cli as _cli

sys.modules[__name__] = _cli
