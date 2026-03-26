"""Telemetry helpers exposed as part of the package surface."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import enhance_video.pipeline as _pipeline

trace = _pipeline.trace
tracer = _pipeline.tracer

__all__ = ["_traced", "init_tracing", "trace", "tracer"]


def init_tracing() -> None:
    global tracer
    _pipeline.init_tracing()
    tracer = _pipeline.tracer


def _traced(func: Callable[..., Any]) -> Callable[..., Any]:
    return _pipeline._traced(func)


def __getattr__(name: str) -> Any:
    if name in {"trace", "tracer"}:
        return getattr(_pipeline, name)
    raise AttributeError(name)
