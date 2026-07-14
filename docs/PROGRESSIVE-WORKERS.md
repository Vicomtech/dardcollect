# Defer-launch — load heavy models once, no GPU contention

## Problem
The orchestrator re-launches each stage every `rerun_interval` (default 5 s) while
upstream is still producing, so a model-heavy downstream stage (quality OFIQ,
filter MagFace, transcribe_video Whisper) re-launches 70–150× during a slow root
(`clips`), reloading its models cold each time — hours of GPU wasted on "nothing
new" re-runs.

A **persistent worker** (load models once, loop, hold them) would fix the reloads
but **causes GPU OOM**: it holds multiple model sets concurrently (quality +
filter + transcribe all resident) while the one-shot stages (clips detector,
OCR) also need the GPU. Tried, hit OOM/contention on a single GPU.

## Solution: defer-launch
The orchestrator **defers** a model-heavy stage's first launch until its
dependencies **finish** (not just start), then launches it **once**: the stage
loads its models a single time (deps have freed the GPU), processes all its
now-complete inputs in a single pass, and exits.

- ✅ **Load-once**: no re-launch → no 70× reloads.
- ✅ **Memory-safe**: only 1–2 model sets resident at a time (deps freed GPU
  before the stage loads) — no OOM, like the original one-shot.
- ✅ **Gate-validatable**: the fixture verifies it (no contention to mask).
- ⚠️ **Loses parent→child overlap**: the stage processes after its parent
  finishes, not as the parent streams. Acceptable: these stages' time is small
  vs the `clips` long-pole, and the overlap was what caused the 70× reloads.

`DEFER_UNTIL_DEPS_DONE = {"quality", "filter", "transcribe_video"}` in
`dardcollect/orchestrator_plan.py`; the defer-wait lives in `_stage_worker`
(`scripts/run_pipeline.py`). The stages themselves are unchanged one-shot (the
`_process_one_pass` extraction is just a clean refactor — no `--progressive`).

Not deferred (and why):
- `clips` — root, runs once already (no re-launch waste).
- `face_crops_video`, `masks` — cv2 + sidecar keypoints, no ONNX models (cheap
  re-runs); and `face_crops_video`'s overlap with `clips` is worth keeping.
- `audio_clips` — ffmpeg, no models.
- `transcribe_audio` — Whisper, but root-ish (dep `download` is skipped → no deps
  to defer on); runs once already.

## FAIR / resumability impact
None. Outputs are byte-identical to the plain one-shot path — same scan + same
per-clip processing, just one model load. `.done` resumability unchanged.

## When to upgrade
If, at very large scale, the downstream chain (quality + filter + transcribe)
becomes a **measured** wall-time bottleneck (comparable to `clips`), upgrade to
**persistent workers + a GPU concurrency semaphore** (limit model-heavy stages to
K concurrent, e.g. 1–2): that keeps load-once AND restores parent→child overlap
(quality overlapping the cv2 `face_crops_video`) while bounding VRAM. More
complex (semaphore + lifecycle) and the fixture can't validate the overlap, so
build it only with measured justification, not preemptively.

## Test plan
- **Unit** (`tests/test_run_pipeline_progressive.py`): the one-shot skip/ready
  tests (using non-deferred aliases `frames`).
- **Objective gate (fresh):** `scripts/objective_gate.py` — validates defer-launch
  converges (deferred stages run once) + golden 0 hard-fail. (The fixture must be
  the small baseline-matched set — huge real-data files in `tests/fixtures/media/`
  make the gate slow and produce baseline drift; keep them out of the fixture dir.)