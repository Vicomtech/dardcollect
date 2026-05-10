"""
Person detection library for extracting video clips containing people.

This library provides simplified wrappers around YOLOX detection,
OC-SORT tracking, and CIGPose body keypoint estimation algorithms.
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
