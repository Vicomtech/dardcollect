# DARDcollect — Quick Start (5 min)

## Installation

### Prerequisites
- **Python**: 3.9+
- **OS**: Linux, macOS, Windows
- **GPU** (optional): NVIDIA GPU with CUDA 12.x driver (550+) — CPU-only mode also supported

### Step 1: Clone & Setup

```bash
git clone https://github.com/Vicomtech/dardcollect.git
cd dardcollect

# Create virtual environment (Linux/macOS/Windows all support this)
python -m venv .venv

# Activate it
# Linux/macOS:
source .venv/bin/activate
# Windows:
# .venv\Scripts\activate

# Install dependencies
pip install -e .
```

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
python scripts/download_media_from_archive.py
```

Outputs: `DARD/archive_org_public_domain/{videos,images,audio,texts}/` + `DARD/archive_org_public_domain/downloads.csv`

### Step 4: Process by Modality

**Video Pipeline** (person clips → face crops → transcriptions):
```bash
python scripts/extract_person_clips_from_videos.py
python scripts/extract_face_crops_from_videos.py
python scripts/transcribe_video_clips.py
```

**Image Pipeline** (person detection → face crops):
```bash
python scripts/extract_persons_from_images.py
python scripts/extract_face_crops_from_images.py
```

**Audio Pipeline** (transcriptions):
```bash
python scripts/transcribe_audio_files.py
```

**Document Pipeline** (text extraction):
```bash
python scripts/extract_text_from_doc.py
```

**Quality Annotation** (all face crops):
```bash
python scripts/annotate_face_quality.py      # OFIQ 7-dimensional scoring
python scripts/filter_face_crops_by_quality.py
```

### Step 5: Check Outputs

```bash
# Each stage writes its traceability CSV alongside its output artifacts:
tail -5 DARD/archive_org_public_domain/downloads.csv
tail -5 DARD/extracted_person_clips/clips_extraction.csv
tail -5 DARD/face_crops/face_crops_extraction.csv
tail -5 DARD/extracted_image_detections/image_person_detection.csv
tail -5 DARD/audio_transcriptions/audio_transcriptions_extraction.csv
tail -5 DARD/preprocessed_documents/document_text_extraction.csv
tail -5 DARD/filtered_face_crops/filtered_face_crops.csv
```

---

## Next Steps

- **Architecture & Workflow**: See [docs/1-ARCHITECTURE.md](1-ARCHITECTURE.md)
- **CSV Logging & FAIR Compliance**: See [docs/2-LOGGING.md](2-LOGGING.md)
- **Quality Annotations (OFIQ)**: See [docs/3-ANNOTATIONS.md](3-ANNOTATIONS.md)
- **GPU Setup & Development**: See [docs/4-DEVELOPMENT.md](4-DEVELOPMENT.md)

## Troubleshooting

### "No GPU detected" or "CUDA not available"
CPU-only mode will activate automatically. Check [docs/4-DEVELOPMENT.md](4-DEVELOPMENT.md) for GPU setup.

### "Config validation failed"
Run: `python -m persondet.config` to validate your `config.yaml`.

### ImportError on persondet modules
Ensure `.venv` is activated and `pip install -e .` was run.

---

← [Back to README](../README.md)
