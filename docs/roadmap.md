# Roadmap

## Now
- Finish splitting orchestration out of `enhance_video.pipeline`.
- Replace wrapper-style helper modules with native implementations.
- Keep CI, lint, package metadata, and artifact hygiene green.

## Next
- Add runtime benchmarks and profiling fixtures for scene-adaptive workloads.
- Reduce repeated frame metadata work and other I/O-heavy hotspots.
- Improve `--type auto` heuristics with measured validation sets.

## Later
- Improve UX around vendor/bootstrap setup and diagnostics.
- Add better reporting for runtime guardrails and scene decisions.
- Explore broader packaging or distribution only after the local macOS flow is fully hardened.
