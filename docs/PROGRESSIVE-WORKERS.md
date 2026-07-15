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

**Fail-loud:** if the copy fails (disk full, permission, source unreachable), the stage
raises — no silent fallback to network read (per the CLAUDE.md runtime-fallback policy).
`local_cache_dir` MUST be local and outside `input_dir` (it is not scanned by any stage).
Opt-in; default off = zero change for fixture / local-dataset / full Archive.org runs.

**Follow-up (not in this chunk):** if GPU util is still <60 % after local pre-copy, the
next lever is a threaded read-ahead decode (producer-consumer queue feeding the GPU),
then batched inference.

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