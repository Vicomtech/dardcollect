# Development & GPU Setup

## Contents

- [GPU Configuration](#gpu-configuration-optional)
  - [GPU-Enabled Setup](#gpu-enabled-setup)
  - [CPU-Only Setup](#cpu-only-setup)
- [Development Workflow](#development-workflow)
  - [1. Setup Environment](#1-setup-environment)
  - [2. Code Structure](#2-code-structure)
  - [3. Adding a New Script](#3-adding-a-new-script)
  - [4. Testing](#4-testing)
  - [5. Configuration Development](#5-configuration-development)
- [Dependencies & Models](#dependencies--models)
  - [Core Dependencies](#core-dependencies)
  - [Pre-Downloaded Models](#pre-downloaded-models)
- [Logging & Debugging](#logging--debugging)
  - [Enable Verbose Logging](#enable-verbose-logging)
  - [CSV Inspection](#csv-inspection)
  - [Model Diagnostics](#model-diagnostics)
- [Contributing](#contributing)
  - [Reporting Issues](#reporting-issues)
  - [Submitting Changes](#submitting-changes)
  - [Code Style](#code-style)
  - [Documentation Updates](#documentation-updates)
- [Performance Tuning](#performance-tuning)
  - [Memory Usage](#memory-usage)
  - [Speed Optimization](#speed-optimization)
  - [Storage](#storage)
- [References](#references)

---

## GPU Configuration (Optional)

DARDcollect auto-detects GPU availability and preloads NVIDIA libraries at import. No manual configuration needed in most cases.

### GPU-Enabled Setup

**Prerequisites:**
- NVIDIA GPU (compute capability 7.0+; tested on V100, A100, RTX 30/40 series)
- NVIDIA Driver 530+ (Linux: 530.30.02+, Windows: 531.14+)

#### Driver Requirements (CUDA 12.1)

| OS | Minimum Driver | Recommended |
|---|---|---|
| Linux | 530.30.02+ | 535+ (latest R535/R550 series) |
| Windows | 531.14+ | 535+ (latest Game Ready / Studio) |

**Installation:**

```bash
# uv sync installs CUDA 12.1 wheels + TensorRT automatically (Linux/Windows)
uv sync

# Verify GPU availability
python -c "import onnxruntime; print(onnxruntime.get_available_providers())"
# Output should include: CUDAExecutionProvider (and TensorrtExecutionProvider if TRT available)

# Run a script with GPU (auto-detected)
python pipeline/extract_person_clips_from_videos.py
# Logs will show: "Using TensorRT/CUDA execution provider" or "Using CPU execution provider"
```

**GPU acceleration is automatic:** NVIDIA libraries (CUDA, cuDNN, TensorRT) are bundled via PyTorch CUDA 12.1 wheels and auto-preloaded when you `import dardcollect`. Falls back to CPU on machines without compatible GPUs.

**Troubleshooting GPU Issues:**

| Issue | Solution |
|-------|----------|
| "CUDA not found" | Check driver: `nvidia-smi` should show your GPU |
| "No GPU execution provider" | CPU fallback is automatic; check if CUDA 12.1 compatible driver installed |
| Out of Memory (OOM) | Reduce batch size in `config.yaml` or use CPU mode |
| Slow performance on GPU | Check `nvidia-smi` during execution to see utilization |

### CPU-Only Setup

CPU mode activates automatically when no GPU is detected. To force CPU even with a GPU available:

```bash
# Preferred: hide GPUs from the process entirely
export CUDA_VISIBLE_DEVICES=""   # Linux/macOS
set CUDA_VISIBLE_DEVICES=        # Windows cmd
$env:CUDA_VISIBLE_DEVICES=""     # Windows PowerShell

python pipeline/extract_person_clips_from_videos.py
```

Or programmatically when creating ONNX sessions:
```python
from dardcollect.onnx_utils import create_ort_session

# Force CPU execution provider
session = create_ort_session(model_path, providers=["CPUExecutionProvider"])
```

---

## Development Workflow

### 1. Setup Environment

**Installation & setup:** See [docs/0-GETTING-STARTED.md](0-GETTING-STARTED.md) for complete instructions (Python 3.9+, venv, pip install).

Once installed, you have a ready-to-use DARDcollect environment with all dependencies.

### 2. Code Structure

```
dardcollect/
├── dardcollect/              # Main library
│   ├── detector.py         # YOLOX person detection
│   ├── poser.py            # CigPose keypoint estimation
│   ├── face_geometry.py    # OFIQ face crop alignment
│   ├── fair.py             # FAIR metadata generation
│   ├── audio.py            # Whisper transcription
│   ├── ocr.py              # PDF/text extraction
│   ├── pipeline_loggers.py # CSV logging (10 loggers)
│   ├── extraction_logger.py # Legacy clips logger
│   ├── config.py           # Configuration management
│   ├── ingest.py           # register_source_files() for custom data sources
│   ├── gpu_setup.py        # GPU/CPU provider setup
│   └── models/             # Pre-downloaded ONNX models
│
├── pipeline/                # Processing pipelines
│   ├── download_media_from_archive.py
│   ├── extract_person_clips_from_videos.py
│   ├── extract_frames_from_videos.py
│   ├── extract_face_crops_from_videos.py
│   ├── extract_face_crops_from_images.py
│   ├── extract_persons_from_images.py
│   ├── transcribe_video_clips.py
│   ├── transcribe_audio_files.py
│   ├── extract_text_from_doc.py
│   ├── annotate_face_quality.py
│   └── filter_face_crops_by_quality.py
│
├── schemas/                # JSON schemas (sidecars validated at write via dardcollect.fair)
│   ├── person_clip_schema.json
│   ├── face_crop_schema.json   # oneOf: video variant + image variant
│   ├── quality_annotation_schema.json
│   └── transcription_schema.json
│
├── scripts/                # Objective-verification tooling (the dev loop)
│   ├── run_pipeline.py          # Orchestrator: runs the 9 stages in order (--config)
│   ├── golden_snapshot.py       # Golden gate: capture/compare/--validate CSVs+sidecars
│   ├── make_fixture_media.py    # Builds the fast fixture media from the dataset
│   └── make_test_config.py      # Generates config.test.yaml from config.yaml
│
├── docs/                   # Documentation (this folder)
│   ├── 0-GETTING-STARTED.md
│   ├── 1-ARCHITECTURE.md
│   ├── 2-LINEAGE.md        (CSV traceability & provenance)
│   ├── 3-ANNOTATIONS.md    (Sidecar annotation formats)
│   ├── 4-DEVELOPMENT.md    (this file — GPU, dev workflow, objective gate)
│   └── 5-LIBRARY-API.md    (library API)
│
├── tests/                  # CPU-only unit suite (pytest; dev extra)
├── config.yaml             # Main configuration (user-owned source of truth)
├── config.test.yaml        # Generated test config (gitignored; run make_test_config.py)
├── pyproject.toml          # Project metadata + dependencies (+ dev extra: ruff, ty, pytest)
├── uv.lock                 # uv lockfile — pinned transitive deps for reproducible `uv sync` (committed)
├── CLAUDE.md               # Claude Code project context (objective, gates, dev loop)
└── README.md               # Main entry point
```

`uv.lock` is [uv](https://docs.astral.sh/uv/)'s lockfile: it pins the exact versions of every (transitive) dependency so `uv sync` installs the same resolved set on any machine/CI. It is committed; regenerate it with `uv lock` after changing dependencies in `pyproject.toml`. The `scripts/` directory holds the objective-verification tooling — see § 4. Testing for the gate commands, and [CLAUDE.md](../CLAUDE.md) § Objective verification for the full loop.

### 3. Adding a New Script

Template for adding a new extraction or processing script:

```python
#!/usr/bin/env python3
"""Brief description of what this script does."""

import sys
from pathlib import Path
from dardcollect.config import get_log_level
from dardcollect.pipeline_loggers import YourNewLogger  # Your logger

def main():
    logging.getLogger().setLevel(get_log_level(str(CONFIG_PATH)))
    
    # 1. Initialize logger
    logger = YourNewLogger(output_dir=str(output_dir))
    
    # 2. Process items
    for item in items:
        try:
            result = process(item)
            
            # 3. Log every successful processing
            logger.log_extraction(
                id=...,
                source=...,
                output_path=...,
                # ... metadata fields
            )
        except Exception as e:
            logger.logger.error(f"Failed: {e}")
    
    # 4. Print summary
    logger.print_summary()

if __name__ == "__main__":
    main()
```

### 4. Testing

#### Unit Tests
```bash
# Run all tests (CPU-only; pytest is in the dev extra — uv sync --extra dev)
pytest tests/

# Run a specific test module
pytest tests/test_fair.py -v

# Run with coverage
pytest --cov=dardcollect tests/
```

#### Objective gate (fast fixture, ~1–2 min)
The pipeline stages do not take a `--config` CLI flag; instead they read the
`DARDCOLLECT_CONFIG` env var (default `config.yaml`). The orchestrator
`scripts/run_pipeline.py --config <path>` sets that env var for every stage.
See [CLAUDE.md](../CLAUDE.md) § Objective verification for the full setup
(build fixture media + test config once per machine) and gate commands:
```bash
# One-time setup per machine (needs the dataset under DARD/archive_org_public_domain/):
python scripts/make_fixture_media.py
python scripts/make_test_config.py
python scripts/run_pipeline.py --config config.test.yaml
python scripts/golden_snapshot.py --dard-root DARD_test capture tests/fixtures/golden_manifest.json

# Gate (each iteration):
python scripts/run_pipeline.py --config config.test.yaml
python scripts/golden_snapshot.py --dard-root DARD_test compare tests/fixtures/golden_manifest.json --validate

# Verify CSV output (each CSV is co-located with its output dir)
ls DARD/extracted_person_clips/clips_extraction.csv
ls DARD/video_face_crops/video_face_crops_extraction.csv
```

### 5. Configuration Development

Edit `config.yaml` to customize:
- Model paths and sizes
- Detection confidence thresholds
- Quality filter parameters
- Output directories

See `dardcollect/config.py` for schema validation.

---

## Dependencies & Models

### Core Dependencies
- **onnxruntime**: ONNX model inference (GPU or CPU)
- **opencv-python**: Image processing (cv2)
- **numpy**: Numerical operations
- **moviepy**: Video frame extraction
- **pydantic**: Configuration validation
- **torch**: Whisper transcription
- **pdfplumber**: PDF text extraction

### Pre-Downloaded Models
All models are automatically downloaded to `dardcollect/models/`:

| Model | File | Size | Purpose |
|-------|------|------|---------|
| YOLOX-tiny | yolox_tiny_8xb8-300e_humanart.onnx | 6.3 MB | Person detection |
| CigPose-m | cigpose-m_coco-wholebody_256x192.onnx | 22.0 MB | Pose keypoints (133 points) |
| MagFace | magface_iresnet50_norm.onnx | 78.0 MB | Face quality/embedding |
| OFIQ | ofiq_config.jaxn | 0.1 MB | Quality config (7 dimensions) |
| Whisper (small) | openai_whisper_small.pt | 461 MB | Audio transcription |
| BiSeNet | bisenet_400.onnx | 28.0 MB | Face segmentation (optional) |

---

## Logging & Debugging

### Enable Verbose Logging
```bash
# Set log level in config.yaml
logging:
  level: DEBUG
```

### CSV Inspection
```bash
# View first 5 rows of any CSV (each CSV is co-located with its output dir)
head -5 DARD/extracted_person_clips/clips_extraction.csv

# Count total entries (excluding header)
tail -n +2 DARD/extracted_person_clips/clips_extraction.csv | wc -l

# Find entries matching a pattern
grep "my_video" DARD/extracted_person_clips/clips_extraction.csv
```

### Model Diagnostics
```python
import onnxruntime as ort

# Check available execution providers
print(ort.get_available_providers())

# Check model inputs/outputs
sess = ort.InferenceSession("model.onnx")
print(sess.get_inputs())
print(sess.get_outputs())
```

---

## Contributing

### Reporting Issues
1. Check if issue already exists on GitHub
2. Include: OS, Python version, error message, reproduction steps
3. Attach relevant config.yaml and CSV snippets

### Submitting Changes
1. Create feature branch: `git checkout -b feature/your-feature`
2. Make changes with commit messages referencing the architecture
3. Test: `pytest tests/`
4. Submit pull request with description

### Code Style
- Follow PEP 8
- Use type hints where possible
- Document public functions with docstrings
- Keep functions < 50 lines when feasible

### Documentation Updates
- Update relevant `.md` file in `docs/` folder
- Cross-reference with other docs
- Test links before submitting

---

## Performance Tuning

### Memory Usage
- **Reduce frame batch size** in config.yaml
- **Use smaller Whisper model** (base instead of small)
- **Filter by confidence** to reduce downstream processing

### Speed Optimization
- **Enable GPU** (10-50x speedup for detection)
- **Parallel processing** (see `max_workers` in config)
- **Skip intermediate steps** if not needed

### Storage
- **Disable JSON sidecars** for high-volume runs (metadata already in CSV)
- **Compress video clips** to lower bitrate

---

## References

- [ONNX Runtime Docs](https://onnxruntime.ai/)
- [YOLOX Repository](https://github.com/Megvii-BaseDetection/YOLOX)
- [CigPose Repository](https://github.com/IDEA-Research/CigPose)
- [OpenAI Whisper](https://github.com/openai/whisper)
- [FAIR Data Principles](https://www.go-fair.org/fair-principles/)

---

← [Back to README](../README.md)
