# Defer-launch — load heavy models once, no GPU contention

## Problem
Historically, the orchestrator re-launched stages every `rerun_interval` (default
5 s) while upstream was still producing, so a model-heavy downstream stage
(quality OFIQ, filter MagFace, transcribe_video Whisper) could re-launch 70–150×
during a slow root (`clips`), reloading models cold each time.

Current behavior is stricter: after a successful run, a downstream stage waits
for **real dependency updates** (or dependency finish) before re-launching. This
keeps progressive behavior without empty timeout-only loops.

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

## Local source pre-copy (network-share throughput)
When `person_extraction.preload_source_to_local: true`, the `clips` stage copies each
source video to a **local** cache dir (`local_cache_dir`, default outside `input_dir`)
once, then runs `cv2.VideoCapture` **and** moviepy `extract_clip` against that local copy.

**Why:** network-share sources starve the GPU. `cv2.VideoCapture.read()` pulls frames
one at a time over the network (per-frame latency), and `extract_clip` re-reads the whole
source **once per emitted clip** (3 clips = 3 full network reads of the source). On a 147 s
webm this measured ~9 min/source at **37 % GPU util** (GPU idling on network I/O). Pre-
copying once collapses all of that to a single network read + local-SSD reads; expected
GPU 60–90 %, per-source ~2–4 min.

**Behavior-preserving:** the local copy is only the *read path*. Clip names, the
`source_video` provenance field, `.json` sidecars, `clips_extraction.csv`, and the `.done`
sentinel all still reference the **original** `video_path`. The local copy is deleted
after each source (one file at a time, sequential — `clips` is a single subprocess).
If `local_cache_dir` was explicitly configured, the cache directory is also removed
when it becomes empty.

**Fail-loud:** if the copy fails (disk full, permission, source unreachable), the stage
raises — no silent fallback to network read (per the CLAUDE.md runtime-fallback policy).
`local_cache_dir` MUST be local and outside `input_dir` (it is not scanned by any stage).
Opt-in; default off = zero change for fixture / local-dataset / full Archive.org runs.

## Read-ahead decode (GPU utilization)
After local pre-copy, `clips` GPU util was still only ~23 % on an RTX 4060 (8 GB, ~3 GB
used). The GPU is NOT VRAM-constrained (detector 416² ≈ 2 MB, pose 192×256 ≈ 0.6 MB per
input) — it is **starved by CPU-side gaps between inference calls**: `cv2.VideoCapture.read()`
decodes each webm (VP8/VP9) frame serially in the main thread, blocking the GPU between calls.

**Read-ahead decode** (`readahead_decode`, `readahead_queue_frames`): a daemon producer
thread (`_FrameReader` in `dardcollect/frame_reader.py`) decodes frames ahead into a bounded
`queue.Queue`; the main loop pops `(frame_id, frame)` via the `frame_iter` generator for GPU
inference. Overlaps CPU decode with GPU work. The producer is joined on generator close
(`frame_iter`'s `finally`) so the thread never leaks across sources or on early
return/exception. Resume still works — `load_resume_start` seeks `cap` before the producer
starts. Producer errors are re-raised by the consumer (fail-loud). Opt-in; default off = zero
change for fixture / local / Archive.org runs.

**Measured (147 s `7r1kuyst4` webm, RTX 4060):** read-ahead alone gives a **marginal** gain
(~4 min 50 s vs 5 min 04 s preload-only vs 9 min baseline). Detection reached frame 3652/3660
in ~2 min, but the source then spends ~2 min 45 s in **moviepy clip extraction** (3 clips,
libx264 re-encode, serial) — that is now the dominant cost, and read-ahead does not touch it
(GPU stays ~14–23 %).

**ffmpeg-CLI extraction — attempted, reverted.** Replacing moviepy in `extract_clip` with a
direct `imageio_ffmpeg` subprocess was tried but did **not** deliver a behavior-preserving
speedup: moviepy already uses the same ffmpeg binary for decode, so the Python pipe overhead
is small. `-t <duration>` was ~22 % faster but cut ~17 frames per clip on the VFR webm sources
(not behavior-preserving vs the sidecar frame count); `-frames:v N` preserved the exact frame
count but was ~1.4× **slower** than moviepy; `-t` + `-r <fps>` preserved frames but only
matched moviepy speed. The real cost is VP8/VP9 decode + libx264 encode, inherent to both
paths. Reverted to the moviepy + atomic-write path.

The remaining lever (deferred): **parallelize the N clip extractions in `flush_segments`**
(`ThreadPoolExecutor` over the moviepy/ffmpeg subprocesses) so the 3 clips of a source extract
concurrently — that overlaps the serial extraction time rather than trying to speed one clip.
Batched detection remains gated on TRT dynamic-batch profiles (see above).

## Parallel clip extraction
`flush_segments` now extracts the N clips of a source concurrently when
`parallel_clip_extraction` is set (`max_extraction_workers` bounds the pool). The N clips are
independent — disjoint frame ranges of the same source — so their moviepy/ffmpeg extractions
overlap via a `ThreadPoolExecutor` over `_extract_one_clip` in
[dardcollect/clip_extraction.py](dardcollect/clip_extraction.py). moviepy runs ffmpeg as a
subprocess, releasing the GIL during the encode, so the 3 encodes run in parallel. Results
are drained in segment order (`ex.map`), and the sidecar write (`save_clip_sidecar_json`) +
`clips_extraction.csv` append run **in the main thread after all extractions** — serialized,
ordered, thread-safe, identical to the serial path. Each clip writes a distinct `clip_path`
with the existing atomic temp + `os.replace`, so concurrent writers never collide. The source
sidecar (archive.org id/url) is read once per source. Opt-in; default off = zero change.

**Measured (147 s `7r1kuyst4`, 3 clips):** extraction phase ~2 min 45 s serial → parallel
~N/max_workers wall (the 3 ffmpeg encodes overlap). Per-source wall time drops accordingly.
Behavior-preserving: same 3 clips, 1491/1491/666 frames, audio, sidecar, CSV rows in the same
order, provenance unchanged.

The only remaining lever is **batched detection/pose** (gated on TRT dynamic-batch profiles in
`dardcollect/onnx_utils.py` or a detector ONNX re-export — golden-baseline risk).

**Batched pose / detection — NOT done (TensorRT fixed-batch engines).** The committed ONNX
graphs are dynamic-batch (cigpose `['batch', 3, 256, 192]`), but the TensorRT execution
provider in `dardcollect/onnx_utils.py` builds engines with a fixed `batch=1` profile (no
`trt_profile_min/max_shapes`), so `session.run` with batch>1 raises `INVALID_ARGUMENT: Got: N
Expected: 1`. The `yolox_tiny` ONNX is fixed-batch=1 at the graph level too. Enabling either
requires either re-exporting the detector ONNX with a dynamic batch axis (model-artifact
change, golden-baseline risk) OR adding dynamic-batch optimization profiles to
`get_preferred_providers` (global, rebuilds all TRT engines, risk to other models) — both
deferred. Per-tracklet pose (the current path) stays; a `PoseEstimator._crop_for_bbox` helper
was extracted as a clean refactor of the single-crop path.

**Behavior-preserving:** read-ahead yields the same detections / keypoints / segments /
provenance (the frame stream is unchanged, just decoded in a producer thread). Verified: 3
clips for `7r1kuyst4` (same segments as baseline), `.json` sidecar `source_video` = original
network path (no cache / queue leak), `.done` written.

## Concurrent writer/reader race on the clips dir (atomic writes)
The non-deferred stages `clips`, `audio_clips`, and `face_crops_video` all run
concurrently and share one directory: `clips` *writes* `.mp4` + `.json` into
`extracted_person_clips/`, while `audio_clips` and `face_crops_video` *scan*
that same dir every `rerun_interval` (`rglob("*.mp4")` / `rglob("*.json")`).

On Windows this is a file-lock race: a downstream reader can open a clip `.mp4`
**while `clips` is still writing it**, locking the file so ffmpeg cannot
finalize the `moov` atom. The result is a corrupt clip (ftyp + mdat, **no
moov**) that `audio_clips` rejects (`moov atom not found`), `face_crops_video`
skips (`No sidecar JSON`), and — because the clip never gets a `.done` — `clips`
re-extracts every rerun forever. One such source pins the whole pipeline in an
infinite loop (`finished: 0/N`, deferred stages never launch).

Fix (`dardcollect/pipeline_utils.py`): `extract_clip` and
`save_clip_sidecar_json` write to a sibling `*.partial` temp file and
`os.replace` it into place on success. Readers glob `*.mp4` / `*.json`, which
do **not** match the `.partial` suffix, so they only ever observe a complete
file. ffmpeg is forced to the mp4 muxer via `ffmpeg_params=["-f","mp4"]` since
the temp extension no longer signals the format. `os.replace` also overwrites
any stale corrupt clip left by a prior interrupted run, self-healing the output
dir on the next pass. The existing `.tmp.mp4` pattern in `audio.py`
(`mux_audio_into_face_crop`) is safe without this change because the face-crops
dir is only read by **deferred** stages (quality/filter/masks), which launch
after `face_crops_video` finishes — no concurrent reader there.

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