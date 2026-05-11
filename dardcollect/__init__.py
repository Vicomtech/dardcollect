"""
Person detection library for extracting video clips containing people.

This library provides simplified wrappers around:
- YOLOX detection for person detection in images and video frames
- OC-SORT tracking for multi-person tracking across frames
- CIGPose body keypoint estimation for pose analysis

It also includes utilities for face crop extraction, quality assessment,
frame extraction, audio transcription, OCR, and FAIR-compliant data management.
"""

from .config import DetectorConfig, FaceCropConfig
from .detector import PersonDetector
from .poser import PoseEstimator
from .tracker import PersonTracker, Tracklet

__all__ = [
    "DetectorConfig",
    "FaceCropConfig",
    "PersonDetector",
    "PersonTracker",
    "PoseEstimator",
    "Tracklet",
]

__version__ = "0.1.0"
