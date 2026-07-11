# DARDcollect — Getting Started

## Contents

- [Installation](#installation)
  - [Prerequisites](#prerequisites)
  - [Step 1: Clone & Setup](#step-1-clone--setup)
  - [Step 2: Configure](#step-2-configure)
  - [Step 3: Download Media from Archive.org](#step-3-download-media-from-archiveorg)
  - [Step 4: Process by Modality](#step-4-process-by-modality)
  - [Step 5: Check Outputs](#step-5-check-outputs)
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

Edit `config.yaml` to select media types and customise the search query:
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

**Video Pipeline** (person clips → face crops → transcriptions):
```bash
python pipeline/extract_person_clips_from_videos.py
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

**Quality Annotation** (all face crops):
```bash
python pipeline/annotate_face_quality.py      # OFIQ 7-dimensional scoring
python pipeline/filter_face_crops_by_quality.py
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

To verify the pipeline end-to-end without downloading a full dataset, use the fast fixture harness:

### Setup (one-time per machine)

```bash
# 1. Generate small test media (30s video + sample images/audio/PDFs)
python scripts/make_fixture_media.py
# Outputs: tests/fixtures/media/ (ignore if it already exists — script is idempotent)

# 2. Generate fixture config (redirects DARD paths to tests/fixtures/)
python scripts/make_test_config.py
# Outputs: config.test.yaml (gitignored, per-machine)
```

### Run & Verify

```bash
# 3. Run all 11 stages on fixture (minutes, not hours)
python scripts/run_pipeline.py --config config.test.yaml
# Outputs: DARD_test/ (parallel to DARD/, isolated fixture outputs)

# 4. Verify: all CSVs present, sidecars valid, provenance intact (golden gate)
python scripts/golden_snapshot.py --dard-root DARD_test compare tests/fixtures/golden_manifest.json --validate
```

Expected output: `[compare] 12 match; 26 drift (GPU non-determinism); 0 hard-fail`  
(GPU inference varies run-to-run; hash diffs are expected and informational.)

This is the **objective gate** used in development: it runs in ~1–2 minutes and confirms that all 11 stages complete without regressions.

---

## Production Workflow (Full Dataset)

For production runs on your own dataset, use `config.yaml` (no `--config` override):

```bash
# Runs download + all processing stages (hours)
python scripts/run_pipeline.py
```

This automatically:
1. **Downloads** media from archive.org (resumable, skips already-downloaded)
2. **Processes** all 11 stages on `DARD/archive_org_public_domain/` outputs

Both `config.test.yaml` (fixture) and `config.yaml` (full) are auto-detected by `run_pipeline.py`:
- Fixture workflow → skips download (media already in `tests/fixtures/media/`)
- Full workflow → includes download as first stage

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
Run: `python -m dardcollect.config` to validate your `config.yaml`.

### ImportError on dardcollect modules
Ensure `.venv` is activated and `pip install -e .` was run.

---

← [Back to README](../README.md)
