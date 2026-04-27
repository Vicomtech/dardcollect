"""
Configuration module for the detector library.

Handles model paths, detection thresholds, and other parameters.
"""

import logging
from dataclasses import dataclass
from pathlib import Path

import yaml

# Default path to models directory within the package
DEFAULT_MODELS_PATH = str(Path(__file__).parent / "models")


def get_log_level(yaml_path: str) -> int:
    """Return the logging level integer from the top-level log_level key in config.yaml."""
    with open(yaml_path, encoding="utf-8") as f:
        config_data = yaml.safe_load(f)
    level_name = config_data.get("log_level", "INFO").upper()
    return getattr(logging, level_name, logging.INFO)


@dataclass
class DetectorConfig:
    """Configuration for the person detection library.

    :param models_path: Path to directory containing ONNX models.
    :param detection_threshold: Confidence threshold for person detection.
    :param tracking_score_threshold: Score threshold for tracking association.
    :param tracking_min_hits: Minimum consecutive hits to confirm a track.
    :param tracking_max_time_lost: Max frames before removing a lost track.
    :param pose_keypoint_threshold: Keypoint confidence threshold.
    """

    detection_threshold: float
    tracking_score_threshold: float
    tracking_min_hits: int
    tracking_max_time_lost: int
    pose_keypoint_threshold: float
    models_path: str = DEFAULT_MODELS_PATH
    detection_model_type: int = 0
    pose_model_type: int = 0
    gpu_id: int = 0

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "DetectorConfig":
        """Load configuration from a YAML file.

        :param yaml_path: Path to YAML configuration file.
        :return: DetectorConfig instance.
        :raises ValueError: If required configuration keys are missing.
        """
        with open(yaml_path, encoding="utf-8") as f:
            config_data = yaml.safe_load(f)

        if "person_extraction" not in config_data:
            raise ValueError("Missing 'person_extraction' section in config")

        cfg = config_data["person_extraction"]

        def get_required(key: str):
            if key not in cfg:
                raise ValueError(f"Missing required config key: person_extraction.{key}")
            return cfg[key]

        return cls(
            models_path=cfg.get("models_path", DEFAULT_MODELS_PATH),
            detection_threshold=get_required("detection_threshold"),
            detection_model_type=cfg.get("detection_model_type", 0),
            tracking_score_threshold=get_required("tracking_score_threshold"),
            tracking_min_hits=get_required("tracking_min_hits"),
            tracking_max_time_lost=get_required("tracking_max_time_lost"),
            pose_model_type=cfg.get("pose_model_type", 0),
            pose_keypoint_threshold=get_required("pose_keypoint_threshold"),
            gpu_id=cfg.get("gpu_id", config_data.get("gpu_id", 0)),
        )


@dataclass
class ClipExtractionConfig:
    """Configuration for video clip extraction."""

    input_dir: str
    output_clips_dir: str
    min_clip_duration_seconds: float
    max_clip_duration_seconds: float
    min_consecutive_frames: int
    merge_gap_frames: int
    require_face_visibility: bool
    min_face_size_percent: float
    min_face_visible_frames: int
    min_consecutive_face_frames: int = 5  # Longest unbroken run of face-visible frames required

    models_path: str = DEFAULT_MODELS_PATH
    require_frontal_face: bool = False
    frontal_symmetry_threshold: float = 0.5
    enable_transcription: bool = False
    transcription_model_size: str = "small"
    enable_visual_speaking: bool = False
    scene_change_detection: bool = True
    scene_change_threshold: float = 0.5
    scene_change_bbox_area_ratio: float = 4.0
    min_free_disk_gb: float = 2.0
    max_bbox_area_percent: float = 60.0
    max_detection_aspect_ratio: float = 3.0  # max width/height; landscape = FP
    max_track_overlap_iou: float = 0.5  # suppress duplicate tracklets above this IoU

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "ClipExtractionConfig":
        """Load configuration from a YAML file.

        :raises ValueError: If required configuration keys are missing.
        """
        with open(yaml_path, encoding="utf-8") as f:
            config_data = yaml.safe_load(f)

        if "person_extraction" not in config_data:
            raise ValueError("Missing 'person_extraction' section in config")

        cfg = config_data["person_extraction"]

        def get_required(key: str):
            if key not in cfg:
                raise ValueError(f"Missing required config key: person_extraction.{key}")
            return cfg[key]

        return cls(
            input_dir=get_required("input_dir"),
            output_clips_dir=get_required("output_clips_dir"),
            min_clip_duration_seconds=get_required("min_clip_duration_seconds"),
            max_clip_duration_seconds=get_required("max_clip_duration_seconds"),
            min_consecutive_frames=get_required("min_consecutive_frames"),
            merge_gap_frames=get_required("merge_gap_frames"),
            require_face_visibility=get_required("require_face_visibility"),
            min_face_size_percent=get_required("min_face_size_percent"),
            min_face_visible_frames=get_required("min_face_visible_frames"),
            min_consecutive_face_frames=cfg.get("min_consecutive_face_frames", 5),
            models_path=cfg.get("models_path", DEFAULT_MODELS_PATH),
            require_frontal_face=cfg.get("require_frontal_face", False),
            frontal_symmetry_threshold=cfg.get("frontal_symmetry_threshold", 0.5),
            enable_transcription=cfg.get("enable_transcription", False),
            transcription_model_size=cfg.get("transcription_model_size", "small"),
            enable_visual_speaking=cfg.get("enable_visual_speaking", False),
            scene_change_detection=cfg.get("scene_change_detection", True),
            scene_change_threshold=cfg.get("scene_change_threshold", 0.5),
            scene_change_bbox_area_ratio=cfg.get("scene_change_bbox_area_ratio", 4.0),
            min_free_disk_gb=cfg.get("min_free_disk_gb", 2.0),
            max_bbox_area_percent=cfg.get("max_bbox_area_percent", 60.0),
            max_detection_aspect_ratio=cfg.get("max_detection_aspect_ratio", 3.0),
            max_track_overlap_iou=cfg.get("max_track_overlap_iou", 0.5),
        )


@dataclass
class FaceQualityFilterConfig:
    """Configuration for face crop quality filtering with MagFace."""

    input_dir: str
    output_dir: str
    quality_threshold: float
    gpu_id: int = 0
    min_free_disk_gb: float = 2.0

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "FaceQualityFilterConfig":
        """Load configuration from a YAML file.

        :raises ValueError: If required configuration keys are missing.
        """
        with open(yaml_path, encoding="utf-8") as f:
            config_data = yaml.safe_load(f)

        if "face_quality_filtering" not in config_data:
            raise ValueError("Missing 'face_quality_filtering' section in config")

        cfg = config_data["face_quality_filtering"]

        def get_required(key: str):
            if key not in cfg:
                raise ValueError(f"Missing required config key: face_quality_filtering.{key}")
            return cfg[key]

        return cls(
            input_dir=get_required("input_dir"),
            output_dir=get_required("output_dir"),
            quality_threshold=get_required("quality_threshold"),
            gpu_id=cfg.get("gpu_id", config_data.get("gpu_id", 0)),
            min_free_disk_gb=cfg.get("min_free_disk_gb", 2.0),
        )


@dataclass
class FaceCropConfig:
    """Configuration for face crop extraction.

    extract_face_crops.py always produces two crop formats:
      - arcface/ (112×112): ArcFace-aligned, for MagFace / identity models.
      - ofiq/    (616×616): OFIQ-aligned, for OFIQ quality measures.
    output_dir is the parent of both subdirectories.
    """

    input_dir: str
    output_dir: str
    detection_threshold: float
    pose_keypoint_threshold: float
    min_eye_distance_px: float
    min_track_face_frames: int
    skip_no_face_frames: bool
    gpu_id: int = 0
    models_path: str = DEFAULT_MODELS_PATH
    min_free_disk_gb: float = 2.0
    include_audio: bool = True
    max_overlap_iou: float = 0.3

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "FaceCropConfig":
        """Load configuration from a YAML file.

        :raises ValueError: If required configuration keys are missing.
        """
        with open(yaml_path, encoding="utf-8") as f:
            config_data = yaml.safe_load(f)

        if "face_crop_extraction" not in config_data:
            raise ValueError("Missing 'face_crop_extraction' section in config")

        cfg = config_data["face_crop_extraction"]

        def get_required(key: str):
            if key not in cfg:
                raise ValueError(f"Missing required config key: face_crop_extraction.{key}")
            return cfg[key]

        return cls(
            input_dir=get_required("input_dir"),
            output_dir=get_required("output_dir"),
            detection_threshold=cfg.get("detection_threshold", 0.3),
            pose_keypoint_threshold=cfg.get("pose_keypoint_threshold", 0.3),
            min_eye_distance_px=cfg.get("min_eye_distance_px", 10),
            min_track_face_frames=cfg.get("min_track_face_frames", 10),
            skip_no_face_frames=cfg.get("skip_no_face_frames", False),
            gpu_id=cfg.get("gpu_id", config_data.get("gpu_id", 0)),
            models_path=cfg.get("models_path", DEFAULT_MODELS_PATH),
            min_free_disk_gb=cfg.get("min_free_disk_gb", 2.0),
            include_audio=cfg.get("include_audio", True),
            max_overlap_iou=cfg.get("max_overlap_iou", 0.3),
        )


@dataclass
class FaceQualityAnnotationConfig:
    """Configuration for face quality annotation (annotate_face_quality.py)."""

    input_dir: str
    gpu_id: int = 0
    frame_stride: int = 5
    max_frames: int = 30
    overwrite: bool = False

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "FaceQualityAnnotationConfig":
        """Load configuration from a YAML file.

        :raises ValueError: If required configuration keys are missing.
        """
        with open(yaml_path, encoding="utf-8") as f:
            config_data = yaml.safe_load(f)

        if "face_quality_annotation" not in config_data:
            raise ValueError("Missing 'face_quality_annotation' section in config")

        cfg = config_data["face_quality_annotation"]

        if "input_dir" not in cfg:
            raise ValueError("Missing required config key: face_quality_annotation.input_dir")

        return cls(
            input_dir=cfg["input_dir"],
            gpu_id=cfg.get("gpu_id", config_data.get("gpu_id", 0)),
            frame_stride=cfg.get("frame_stride", 5),
            max_frames=cfg.get("max_frames", 30),
            overwrite=cfg.get("overwrite", False),
        )
