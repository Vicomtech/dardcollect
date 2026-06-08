"""DARDcollect — GPU-accelerated media processing library for historical archives.

Provides modular components for:
    - **Detection & Tracking:** YOLOX person detection, OC-SORT tracking, CIGPose pose estimation
    - **Audio:** openai-whisper transcription for video and audio files
    - **OCR:** Document text extraction (PDF text layer, TXT native, PaddleOCR fallback)
    - **Face Processing:** OFIQ-aligned crop extraction, ISO/IEC 29794-5 quality scoring
    - **Archive.org:** Mass media download with language organization and FAIR metadata
    - **Data Management:** UUID generation, provenance tracking, FAIR compliance

All components follow FAIR data principles and are independently reusable.
See `docs/5-LIBRARY-API.md` for examples of using individual components in custom workflows.

Optional extras:
    - `tensorrt`: Install for TensorRT acceleration (Linux/Windows)
"""

# Preload PyPI-installed NVIDIA libs (CUDA, cuDNN, TensorRT) before any submodule
# triggers `import onnxruntime`. Without this, ONNX Runtime's dlopen of e.g.
# libnvinfer.so.10 fails on systems where LD_LIBRARY_PATH does not include
# site-packages/tensorrt_libs. Config-free and idempotent — no-op on CPU-only
# installs.
from .gpu_setup import auto_preload_pypi_nvidia_libs

auto_preload_pypi_nvidia_libs()

# Core detection/tracking/pose (primary API)
# Archive.org downloads
from .archive import download_item

# Audio transcription
from .audio import (
    AudioTranscriber,
    scan_for_untranscribed_audio,
    scan_for_untranscribed_clips,
)
from .config import DetectorConfig, FaceCropConfig
from .detector import PersonDetector

# Face crop extraction
from .face_crops import process_image, process_video
from .face_geometry import face_crop_corners

# FAIR metadata & traceability
from .fair import add_fair_metadata, generate_uuid, reorganize_for_fair

# Frame extraction
from .frames import extract_frames

# Custom data source ingestion
from .ingest import register_source_files

# Document extraction (OCR)
from .ocr import DocumentExtractor

# Validation utilities
from .pipeline_utils import check_disk_space, check_face_visibility, check_frontal_face
from .poser import PoseEstimator

# Time utilities
from .provenance import now_iso

# Quality assessment (OFIQ 7-dimensional)
from .quality import load_models, score_video
from .tracker import PersonTracker, Tracklet

__all__ = [
    "AudioTranscriber",  # Transcription
    "DetectorConfig",  # Detection & Tracking & Pose
    "DocumentExtractor",  # OCR
    "FaceCropConfig",
    "PersonDetector",  # Detection & Tracking & Pose
    "PersonTracker",
    "PoseEstimator",
    "Tracklet",  # Detection & Tracking & Pose
    "add_fair_metadata",  # FAIR
    "check_disk_space",  # Validation
    "check_face_visibility",
    "check_frontal_face",
    "download_item",  # Archive.org
    "extract_frames",  # Frames
    "face_crop_corners",  # Face crops
    "generate_uuid",  # FAIR
    "load_models",  # Quality (OFIQ)
    "now_iso",  # Time
    "process_image",  # Face crops
    "process_video",
    "register_source_files",  # Custom data source ingestion
    "reorganize_for_fair",  # FAIR
    "scan_for_untranscribed_audio",  # Transcription
    "scan_for_untranscribed_clips",
    "score_video",  # Quality (OFIQ)
]

__version__ = "0.1.0"
