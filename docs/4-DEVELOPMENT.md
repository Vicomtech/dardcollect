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

DARDcollect can run on **CPU only** (CPU execution provider in ONNX) or leverage **NVIDIA GPUs** for accelerated inference.

### GPU-Enabled Setup

**Prerequisites:**
- NVIDIA GPU (any compute capability 7.0+; tested on V100, A100, RTX)
- NVIDIA Driver 550+ (includes CUDA 12.4+)

**Installation (automatic with `pip install -e .`):**

```bash
# The setup.py automatically installs CUDA-compatible packages
pip install -e .

# Verify GPU availability
python -c "import onnxruntime; print(onnxruntime.get_available_providers())"
# Output should include: CUDAExecutionProvider

# Run a script with GPU
python pipeline/extract_person_clips_from_videos.py
# Logs will show: "Using CUDA execution provider" or "Using CPU execution provider"
```

**Troubleshooting GPU Issues:**

| Issue | Solution |
|-------|----------|
| "CUDA not found" | Check driver: `nvidia-smi` should show your GPU |
| "No GPU execution provider" | CPU fallback is automatic; check if CUDA 12.x libraries are installed |
| Out of Memory (OOM) | Reduce batch size in `config.yaml` or use CPU mode |
| Slow performance on GPU | Check `nvidia-smi` during execution to see utilization |

### CPU-Only Setup

No additional configuration needed—set in code:
```python
from dardcollect.gpu_setup import setup_gpu_paths
setup_gpu_paths(use_cuda=False)  # Force CPU
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
├── schemas/                # JSON schemas
│   ├── person_clip_schema.json
│   ├── face_crop_schema.json
│   ├── quality_annotation_schema.json
│   └── transcription_schema.json
│
├── docs/                   # Documentation (this folder)
│   ├── 0-GETTING-STARTED.md
│   ├── 1-ARCHITECTURE.md
│   ├── 2-LINEAGE.md        (CSV traceability & provenance)
│   ├── 3-ANNOTATIONS.md    (Quality annotation formats)
│   └── 4-DEVELOPMENT.md    (this file)
│
├── config.yaml             # Main configuration
├── pyproject.toml          # Project metadata + dependencies
└── README.md               # Main entry point
```

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
# Run all tests
pytest tests/

# Run specific test
pytest tests/test_detector.py -v

# Run with coverage
pytest --cov=dardcollect tests/
```

#### Integration Tests
```bash
# Test full pipeline on a sample video
python pipeline/extract_person_clips_from_videos.py --config config.test.yaml

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
