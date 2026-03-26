#!/usr/bin/env python3
"""Fail if generated artifacts or vendor archives are tracked in git."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

FORBIDDEN_PREFIXES = (
    "resume/",
    "test_in/",
    "test_out/",
    "test_out_anime/",
    "test_out_v3/",
    "enhance-ai/images/",
)

FORBIDDEN_EXACT = {
    "dummy.jpg",
    "test_frame_2x_direct.png",
    "enhance-ai/realesrgan-macos.zip",
}

ALLOWED_EXACT = {
    "tests/fixtures/tiny_input.mp4",
}


def main() -> int:
    repo_root = Path(__file__).resolve().parent.parent
    tracked = subprocess.run(
        ["git", "ls-files"],
        cwd=repo_root,
        capture_output=True,
        check=True,
        text=True,
    ).stdout.splitlines()

    violations = []
    for path in tracked:
        if path in ALLOWED_EXACT:
            continue
        if path in FORBIDDEN_EXACT:
            violations.append(path)
            continue
        if any(path.startswith(prefix) for prefix in FORBIDDEN_PREFIXES):
            violations.append(path)

    if not violations:
        return 0

    print("Tracked artifact hygiene check failed:", file=sys.stderr)
    for path in violations:
        print(f"  - {path}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
