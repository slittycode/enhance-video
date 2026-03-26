# Development

## Setup
```bash
python -m pip install -e ".[dev,validation,telemetry]"
```

## Daily Commands
```bash
python scripts/check_repo_hygiene.py
ruff check .
pytest -q
```

## Working Rules
- Keep generated media and vendor archives out of git.
- Add new runtime code under `enhance_video/`.
- Keep the legacy top-level modules as thin shims only.
- Prefer returning exit codes from CLI modules and using `SystemExit` only in `__main__` blocks.
