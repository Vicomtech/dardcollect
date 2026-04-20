"""
Configuration module for the detector library.

Handles model paths, detection thresholds, and other parameters.
"""

from dataclasses import dataclass
from pathlib import Path

import yaml

# Default path to models directory within the package
DEFAULT_MODELS_PATH = str(Path(__file__).parent / "models")


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
            gpu_id=cfg.get("gpu_id", 0),
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
class FaceCropConfig:
    """Configuration for face crop extraction."""

    input_dir: str
    output_dir: str
    output_size: int
    detection_threshold: float
    pose_keypoint_threshold: float
    min_eye_distance_px: float
    align_face: bool
    face_padding: float
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
            output_size=cfg.get("output_size", 224),
            detection_threshold=cfg.get("detection_threshold", 0.3),
            pose_keypoint_threshold=cfg.get("pose_keypoint_threshold", 0.3),
            min_eye_distance_px=cfg.get("min_eye_distance_px", 10),
            align_face=cfg.get("align_face", True),
            face_padding=cfg.get("face_padding", 0.4),
            min_track_face_frames=cfg.get("min_track_face_frames", 10),
            skip_no_face_frames=cfg.get("skip_no_face_frames", False),
            gpu_id=cfg.get("gpu_id", 0),
            models_path=cfg.get("models_path", DEFAULT_MODELS_PATH),
            min_free_disk_gb=cfg.get("min_free_disk_gb", 2.0),
            include_audio=cfg.get("include_audio", True),
            max_overlap_iou=cfg.get("max_overlap_iou", 0.3),
        )
