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
- **uv**: [install](https://docs.astral.sh/uv/getting-started/installation/) — manages Python, the virtualenv, and all dependencies
- **OS**: Linux, macOS, Windows
- **GPU** (optional): NVIDIA GPU with CUDA 12.x driver (550+) — CPU-only mode also supported

### Step 1: Clone & Setup

```bash
git clone https://github.com/Vicomtech/dardcollect.git
cd dardcollect
uv sync
```

`uv sync` creates the virtualenv, installs the correct Python version, and resolves all dependencies (including the PyTorch CUDA index configured in `pyproject.toml`). Run subsequent commands with `uv run python …` or activate the venv first (`source .venv/bin/activate` on Linux/macOS, `.venv\Scripts\activate` on Windows).

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
