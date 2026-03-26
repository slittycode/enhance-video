"""Enhance Video package namespace."""

from __future__ import annotations

from typing import Optional, Sequence


def main(argv: Optional[Sequence[str]] = None) -> int:
    from enhance_video.pipeline import main as pipeline_main

    return pipeline_main(argv)


__all__ = ["main"]
