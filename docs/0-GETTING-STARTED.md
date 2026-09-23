# DARDcollect — Getting Started

## Contents

- [Installation](#installation)
  - [Prerequisites](#prerequisites)
  - [Step 1: Clone & Setup](#step-1-clone--setup)
  - [Step 2: Configure](#step-2-configure)
  - [Step 3: Download Media from Archive.org](#step-3-download-media-from-archiveorg)
  - [Step 4: Process by Modality](#step-4-process-by-modality)
  - [Step 5: Check Outputs](#step-5-check-outputs)
- [Use an Existing Dataset (No Download)](#use-an-existing-dataset-no-download)
- [Next Steps](#next-steps)
- [Troubleshooting](#troubleshooting)
  - ["No GPU detected" or "CUDA not available"](#no-gpu-detected-or-cuda-not-available)
  - ["Config validation failed"](#config-validation-failed)
  - [ImportError on dardcollect modules](#importerror-on-dardcollect-modules)

---

## Installation

### Prerequisites
- **Python 3.12**: Required (other versions untested)
- **uv**: [install](https://docs.astral.sh/uv/getting-started/installation/) — manages Python, the virtualenv, and all dependencies
- **OS**: Linux, macOS, Windows
- **GPU** (optional): NVIDIA GPU with driver 530+ (see table below) — CPU-only mode activates automatically

#### GPU Driver Requirements (CUDA 12.1)

| OS | Minimum Driver | Recommended |
|---|---|---|
| Linux | 530.30.02+ | 535+ (latest R535/R550 series) |
| Windows | 531.14+ | 535+ (latest Game Ready / Studio) |

CUDA runtime and cuDNN are bundled via PyTorch CUDA 12.1 wheels — no separate CUDA toolkit install needed.

### Step 1: Clone & Setup

```bash
git clone https://github.com/Vicomtech/dardcollect.git
cd dardcollect
uv sync   # Includes TensorRT + CUDA 12.1 on Linux/Windows, MPS on macOS
```

`uv sync` creates the virtualenv, installs Python 3.12, and resolves all dependencies:
- **Linux/Windows**: PyTorch CUDA 12.1 wheels + TensorRT (works on CPU-only machines too)
- **macOS**: PyTorch with MPS support (Apple Silicon acceleration)

NVIDIA libraries are auto-preloaded at import — no manual GPU setup required. Falls back to CPU automatically on machines without compatible GPUs.

Run subsequent commands with `uv run python …` or activate the venv first (`source .venv/bin/activate` on Linux/macOS, `.venv\Scripts\activate` on Windows).

### Step 2: Configure

Edit `configs/config.archive_all.yaml` (the general / full Archive.org config) to select media types and customise the search query:
```yaml
media_types: ["video"]          # which modalities to download

media_download:
  video:
    search_query: >
      mediatype:(movies) AND licenseurl:*publicdomain*
      AND language:eng
```

### Step 3: Download Media from Archive.org

```bash
python pipeline/download_media_from_archive.py
```

Outputs: `DARD/archive_org_public_domain/{videos,images,audio,texts}/` + `DARD/archive_org_public_domain/downloads.csv`

### Step 4: Process by Modality

**Video Pipeline** (person clips → audio → face crops → transcriptions):
```bash
python pipeline/extract_person_clips_from_videos.py
python pipeline/extract_audio_from_clips.py
python pipeline/extract_face_crops_from_videos.py
python pipeline/transcribe_video_clips.py
```

**Image Pipeline** (person detection → face crops):
```bash
python pipeline/extract_persons_from_images.py
python pipeline/extract_face_crops_from_images.py
```

**Audio Pipeline** (transcriptions):
```bash
python pipeline/transcribe_audio_files.py
```

**Document Pipeline** (text extraction):
```bash
python pipeline/extract_text_from_doc.py
```

**Quality + Frames + Masks** (all face crops converge here):
```bash
python pipeline/annotate_face_quality.py      # OFIQ 7-dimensional scoring
python pipeline/filter_face_crops_by_quality.py
python pipeline/extract_frames_from_videos.py # PNG frames + per-frame sidecars
python pipeline/generate_face_masks.py        # binary face masks from keypoints
```

These last two write many small files, so on network storage (NFS/GPFS) they are
bound by per-file latency rather than CPU. Both take a thread-count knob —
`frame_extraction.workers` and `face_mask_generation.workers` — defaulting to `1`
(serial). Raising them to 8–16 measured 2.6× and 3.4× on GPFS; outputs are
identical either way, since clips and crops are independent and the only shared
state is the lock-guarded CSV logger. Leave them at `1` on a local SSD, where the
work is CPU-bound and threads only add contention.

### Watch your disk budget

A full run is capacity-hungry, and the single biggest lever is what
`frame_extraction.input_dir` points at. Aimed at `{root}/filtered_video_face_crops`
(the default, and what the DAG implies — `frames` depends on `filter`) it explodes
only the crops that passed the quality threshold. Aimed at
`{root}/extracted_person_clips` it explodes every frame of every clip, including the
ones the filter is about to discard: measured on real data that is 330 frames × 240 KB
per clip, roughly **870 GB for a 103-video run**. Check this before a long run.

Two guards help:

- `min_free_disk_gb` (in `person_extraction`, `face_crop_extraction` and the quality
  filter sections) makes a stage exit loudly *before* it writes into a nearly-full
  filesystem, rather than failing halfway through a file. The `2.0` default is far too
  tight for a shared quota — raise it to tens of GB so you have room to react.
- `scripts/reclaim_processed_sources.py` deletes source videos the clip stage has
  finished with (`.done` sentinel present). Nothing downstream reads them again, and
  `downloads.csv` keeps the `uuid` + `archive_org_identifier`, so provenance survives
  and the file is re-downloadable. Dry-run by default:

  ```bash
  uv run python scripts/reclaim_processed_sources.py --config configs/config.archive_all.yaml
  # unattended: check every 10 min, reclaim only when free space drops under 40 GB
  uv run python scripts/reclaim_processed_sources.py --config configs/config.archive_all.yaml \
      --watch 600 --reclaim-below-gb 40 --target-free-gb 60 --apply
  ```

### Step 5: Check Outputs

```bash
# Each stage writes its traceability CSV alongside its output artifacts:
tail -5 DARD/archive_org_public_domain/downloads.csv
tail -5 DARD/extracted_person_clips/clips_extraction.csv
tail -5 DARD/video_face_crops/video_face_crops_extraction.csv
tail -5 DARD/extracted_image_detections/image_person_detection.csv
tail -5 DARD/audio_transcriptions/audio_transcriptions_extraction.csv
tail -5 DARD/preprocessed_documents/document_text_extraction.csv
tail -5 DARD/filtered_video_face_crops/video_filtered_face_crops.csv
```

---

## Testing with Fixture Media (Fast Verification)

To verify the pipeline end-to-end without running it over the whole dataset, use the fast fixture harness. The fixture is a tiny subset **sampled from your existing `DARD/archive_org_public_domain/` download** (smallest files), so it requires that download to be present. For custom/non-Archive datasets, skip the fixture and use the [existing-dataset flow](#use-an-existing-dataset-no-download) instead.

### Setup (one-time per machine)

**No `DARD/` download yet?** Seed it with the minimal download config first — it fetches a tiny Archive.org slice (≤3 items per media type, 1 GB hard cap, smallest-first sort) exactly where `make_fixture_media.py` expects it:

```powershell
# 0. Seed the dataset (tiny download; PowerShell)
$env:DARDCOLLECT_CONFIG = "configs/config.seed.yaml"
uv run python pipeline/download_media_from_archive.py
```

Then build the fixture:

```bash
# 1. Generate small test media (30s video + sample images/audio/PDFs)
uv run python scripts/make_fixture_media.py
# Outputs: tests/fixtures/media/ (ignore if it already exists — script is idempotent)

# 2. Generate fixture config (redirects DARD paths to tests/fixtures/)
uv run python scripts/make_test_config.py
# Outputs: config.test.yaml (gitignored, per-machine)
```

### Run & Verify

```bash
# 3. Run all 12 stages on fixture (minutes, not hours)
python scripts/run_pipeline.py --config configs/config.test.yaml
# Outputs: DARD_test/ (parallel to DARD/, isolated fixture outputs)

# 4. Verify: all CSVs present, sidecars valid, provenance intact (golden gate)
python scripts/golden_snapshot.py --dard-root DARD_test compare tests/fixtures/golden_manifest.json --validate
```

Expected output: `[compare] 12 match; 26 drift (GPU non-determinism); 0 hard-fail`  
(GPU inference varies run-to-run; hash diffs are expected and informational.)

This is the **objective gate** used in development: it runs in ~1–2 minutes and confirms that all 12 stages complete without regressions.

---

## Production Workflow (Full Dataset)

For production runs on your own dataset, use `configs/config.archive_all.yaml` (no `--config` override):

```bash
# Runs download + all processing stages (hours)
python scripts/run_pipeline.py
```

This automatically:
1. **Downloads** media from archive.org (resumable, skips already-downloaded)
2. **Processes** all 12 stages on `DARD/archive_org_public_domain/` outputs

Both `configs/config.test.yaml` (fixture) and `configs/config.archive_all.yaml` (full) are auto-detected by `run_pipeline.py`:
- Fixture workflow → skips download (media already in `tests/fixtures/media/`)
- Full workflow → includes download as first stage

---

## Use an Existing Dataset (No Download)

If you already have media files on disk and do not want to download from Archive.org, point the config inputs to your dataset and run only non-download stages.

### 1. Minimum dataset layout

Place files under the same modality folders used by the pipeline:

```text
<your_root>/
  videos/   # .mp4, .mov, .mkv ...
  images/   # .jpg, .jpeg, .png ...
  audio/    # .mp3, .wav, .flac ...
  texts/    # .pdf, .txt
```

Then set these paths in your config file:

- `person_extraction.input_dir` -> `<your_root>/videos`
- `image_extraction.input_dir` -> `<your_root>/images`
- `audio_transcription.audio_files_dir` -> `<your_root>/audio`
- `document_preprocessing.input_dir` -> `<your_root>/texts`

Keep output paths (`DARD/extracted_person_clips`, `DARD/video_face_crops`, etc.) as-is or point them to your preferred output root.

### 2. Path templating with `{output_root}` (recommended for custom datasets)

Most pipeline configs repeat the same long prefix in 5–8 `output_dir` fields. Declare the prefix once and reference it where it's used. **The template is only for outputs** — input paths stay literal because they describe a fixed dataset, not a generated artifact:

```yaml
# config.mydata.yaml
output_root: "//my-server/share/dataset/outputs"   # outputs only

person_extraction:
  input_dir: "//my-server/share/dataset/videos"     # input: literal
  output_clips_dir: "{output_root}/extracted_person_clips"
face_crop_extraction:
  input_dir: "{output_root}/extracted_person_clips"
  output_dir: "{output_root}/video_face_crops"
face_quality_filtering:
  input_dir: "{output_root}/video_face_crops"
  output_dir: "{output_root}/filtered_video_face_crops"
# ... etc
```

To relocate every output, change the single `output_root` value. **Mixed roots** (input on a network share, outputs on a local SSD) work by overriding individual `input_dir` / `output_dir` fields with literal paths — the template is just a default. Implementation: [config.py `Path templating`](../dardcollect/config.py).

### 2b. Path templating with `{root}` (Archive.org full runs, `config.archive_all.yaml`)

For full Archive.org runs, `{root}` covers **both inputs and outputs** under one base path. The download stage uses a separate `base_output_dir` key (not `{root}`) — keep it aligned:

```yaml
# config.archive_all.yaml
root: "C:/data/DARD"                                      # template for all {root}/... paths
base_output_dir: "C:/data/DARD/archive_org_public_domain" # download stage writes here
                                                           # must equal {root}/archive_org_public_domain

media_download:
  video:
    output_subdir: "videos"   # relative to base_output_dir → C:/data/DARD/archive_org_public_domain/videos

person_extraction:
  input_dir: "{root}/archive_org_public_domain/videos"    # reads from download output
  output_clips_dir: "{root}/extracted_person_clips"
```

To relocate the dataset: change **both** `root` and `base_output_dir` (keeping `base_output_dir = root + "/archive_org_public_domain"`). All other paths update automatically via `{root}`. `output_subdir` values must be **relative** names (e.g. `"videos"`), not absolute paths.

### 3. Run pipeline without download stage

Set this in your config file (for example `config.mydata.yaml`):

```yaml
run_pipeline:
  skip_download: true
  heartbeat_interval_seconds: 10  # optional: periodic status updates in console
  rerun_interval_seconds: 20      # wait for real dep updates; avoids empty rerun loops
```

Then run the progressive orchestrator:

```bash
python scripts/run_pipeline.py --config config.mydata.yaml
```

This runs the full processing pipeline over your local dataset while skipping Archive.org download.

### 3b. Optional config keys (issues #4, #6, #8, #9, #10)

These keys are all **opt-in with defaults that preserve historical behavior**:

```yaml
download:
  av1_policy: warn        # warn (default) | skip — AV1 sources decode 0 frames in
                          # OpenCV stages on builds without an AV1 decoder; warn
                          # logs a loud WARNING per source, skip deletes + records

encoding:                 # video/audio codec for clip extraction + face-crop rendering
  video_codec: libx264    # h264_nvenc etc. needs a SYSTEM ffmpeg (bundled
  audio_codec: aac        # imageio-ffmpeg lacks NVENC): set FFMPEG_BINARY or
  encoder_threads: 8      # IMAGEIO_FFMPEG_EXE. Missing codec fails loud at startup.

person_extraction:
  # Third scene-cut signal (same-set shot/reverse-shot cuts the global
  # histogram survives). Default OFF — calibrate threshold/fraction first.
  scene_change_block_delta: false
  scene_change_block_delta_threshold: 24.0
  scene_change_block_delta_fraction: 0.5

face_quality_filtering:   # (also image_face_quality_filtering)
  demote_on_raise: false  # true = re-runs re-evaluate already-filtered crops
                          # against the CURRENT threshold and reverse-move those
                          # that no longer pass (raising the threshold takes effect)

face_crop_extraction:
  stabilize_face_crops: false   # corner-only stabilization (issue #9): render each
                                # output frame through the track's median OFIQ quad
                                # (removes sub-keypoint wobble). Design:
                                # docs/DESIGN_crop_stabilization.md. Sidecar corners
                                # stay raw per-frame; toggling requires deleting the
                                # crops' .done sentinels to re-render.
  stabilization_min_frames: 5   # min frames with valid corners to engage per track
```

### 4. Optional provenance manifest for non-Archive sources

If your sources are not Archive.org and you still want `downloads.csv`-schema lineage, register source files first. See [Custom Data Sources](2-LINEAGE.md#15-custom-data-sources-non-archiveorg-workflows) in [docs/2-LINEAGE.md](2-LINEAGE.md).

### 5. Content-based colour filter (standalone, issue #11)

Classify videos colour vs black-and-white by their pixels (archive.org's `color` tag is unreliable):

```bash
python pipeline/filter_videos_by_color.py <video_dir>            # classify → color_classification.csv
python pipeline/filter_videos_by_color.py <video_dir> --move     # also relocate B&W videos to a sibling folder (reversible via the CSV)
```

This stage is standalone (not wired into the orchestrator DAG yet).

---

## Next Steps

- **Architecture & Workflow**: See [docs/1-ARCHITECTURE.md](1-ARCHITECTURE.md)
- **CSV Provenance & Traceability**: See [docs/2-LINEAGE.md](2-LINEAGE.md)
- **Quality Annotations (OFIQ)**: See [docs/3-ANNOTATIONS.md](3-ANNOTATIONS.md)
- **GPU Setup & Development**: See [docs/4-DEVELOPMENT.md](4-DEVELOPMENT.md)

## Troubleshooting

### "No GPU detected" or "CUDA not available"
CPU-only mode will activate automatically. Check [docs/4-DEVELOPMENT.md](4-DEVELOPMENT.md) for GPU setup.

### "Config validation failed"
Run: `python -m dardcollect.config` to validate your `configs/config.archive_all.yaml`.

### ImportError on dardcollect modules
Ensure `.venv` is activated and `pip install -e .` was run.

---

← [Back to README](../README.md)
