# Session State — Handoff

Last updated: 2026-09-09 (github-issues session: chunks 1-7 implemented + harness protocol lessons + harness recycle complete)

## Where we are

- Github-issues queue executed end-to-end (2026-09-09, user: "quiero que me soluciones puto todo"):
  #5 frame_stride fixed (shared score_frames_with_stride helper, multi-frame frame_data verified on
  fixture: 18 frames @ stride 5 vs 1 before), #7 fail-loud TemplateMismatch + 9 tests, #6
  demote_on_raise opt-in, #10 av1_policy probe (ffprobe + logged fallback), #4 block-delta signal
  (default OFF), #8 encoding config + video_writers.py extraction (pipeline_utils 675→464),
  #11 standalone filter_videos_by_color.py + color_classification.csv. #9 NOT implemented —
  needs design doc first (MagFace Δmean −0.90 / pass −24% trade-off, sidecar semantics question).
- Golden baseline v2 re-captured 2026-09-09 (8 CSVs / 24 sidecars; surface unchanged; hash drift
  by design from multi-frame frame_data). User ratifies via commit.
- Issue reply texts drafted for #4/#6/#8/#10/#11 (close after commit) and #9 (acknowledge, stays
  open); #5/#7 texts with SHAs delivered earlier. User posts comments (no gh CLI).
- Harness protocol lessons codified (see Rules — this session): queue-scoping rule in AGENTS.md,
  exception in refactor-to-objective skill. Harness-arnes-debt recycled same session: session_state
  handoff (this file), always-close-session rule, size gate in validate_harness.py, rule archive
  (docs/HARNESS_RULES.md), cycle_metrics.py log (scripts/, minimal port).
- Dead code removed (user-approved delete): _backpropagate_quality + _QUALITY_PROVENANCE_KEYS
  (quality.py 573→507), FaceQualityAnnotationLogger (pipeline_loggers.py 463→353). Docs
  3-ANNOTATIONS/2-LINEAGE/1-ARCHITECTURE de-staled to the current two-sidecar quality flow.
- God-file ratchet after chunk work: quality.py 507, pipeline_utils.py 464 (writers extracted to
  dardcollect/video_writers.py). Never raise a baseline; lower it when a file shrinks.

## Key decisions so far

- AV1 route: warn/skip, no transcode (FAIR-tracked source bytes; transcode = follow-up).
- Filter demotion: opt-in only (default preserves idempotent-skip semantics).
- Block-delta scene-cut: default OFF pending calibration on production footage.
- Colour filter: standalone, not wired into orchestrator DAG until threshold calibrated (12.0).
- Dead code over feature restoration for #5's backprop family (data lives in
  .magface.json/.ofiq_attr.json with FAIR parent links; viewer reads per-crop).
- Golden drift policy: GPU non-determinism tolerated (informational); intended behavior changes
  re-capture the baseline; user ratifies via commit.

## Open items (need user decision)

- #9 stabilization: design-doc-first (FEATURE_WORKFLOW), user acks via issue reply (stays open).
- Colour threshold 12.0: calibrate on production corpus before enabling --move at scale.
- Platform parity: this session was Windows-only; Linux/WSL pass pending for the queue commits.

## Known quirks (not errors)

- Test runner: `uv run --no-sync python -m pytest ...` (plain `uv run` may re-resolve torch and hang).
- Windows console: no non-ASCII in scripts/scripts output (cp1252); fixture media ASCII-renamed.
- Fake subprocess tools in tests: `.exe` needs a real PE binary (WinError 216) — use `.bat` scripts
  and patch module attrs (`_ffmpeg_exe`) instead of relying on executable names.
- ty reports 7 pre-existing-style diagnostics (benchmark_pipeline ×5, magface ×2, two
  `cv2.VideoWriter_fourcc` false positives from ty's cv2 stubs — same pattern, not real defects).
- Config path resolution is REPO_ROOT-relative everywhere (orchestrator + viewer + stages) —
  pinned by tests; never make it config-dir-relative.