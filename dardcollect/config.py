"""
Configuration dataclasses for all pipeline stages.

Each dataclass has a from_yaml() classmethod that reads from the corresponding
top-level section in config.yaml and raises ValueError for missing required keys.

Path templating
---------------

If the top-level YAML has a ``root`` key (or any other scalar key), every
string value in the loaded config is scanned for ``{root}`` (or ``{key}``)
and substituted. This lets a dataset config declare its base path once
and reference it as ``{root}/extracted_person_clips`` in every section,
instead of repeating the full UNC / local path 8+ times.

Example::

    # config.yaml
    root: "//my-server/share/dataset"   # or any base path

    person_extraction:
      input_dir: "{root}/videos"
      output_clips_dir: "{root}/extracted_person_clips"
    face_crop_extraction:
      input_dir: "{root}/extracted_person_clips"   # ← still just one place to change
      output_dir: "{root}/video_face_crops"

After ``from_yaml()`` runs, the loaded dataclass has fully resolved
strings (e.g. ``"//gpfs-cluster/.../extracted_person_clips"``). The
substitution uses Python's ``str.format_map`` with a SafeDict, so unknown
placeholders (e.g. ``{foo}`` when ``foo`` isn't a top-level key) are left
literal and never raise. To relocate a dataset, change only the ``root``
key at the top of the config — all downstream paths follow.

Configs that don't use templating (e.g. the default ``config.yaml``) work
exactly as before: every string is taken literally.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

# Default path to models directory within the package
DEFAULT_MODELS_PATH = str(Path(__file__).parent / "models")


def _resolve_path_templates(config_data: dict) -> dict:
    """Recursively replace ``{root}`` (and any ``{key}`` from the top-level
    config) in every string value of *config_data*.

    Returns a new dict; does not mutate the input. Substitutable keys are
    taken from the top-level config (any value that is a string/int/float/bool).
    The top-level source-of-truth entries are not themselves templated
    (avoids ``{root}`` being applied to ``root: '...{root}...'``).
    """
    if not isinstance(config_data, dict):
        return config_data
    substitutions = {k: v for k, v in config_data.items() if isinstance(v, (str, int, float, bool))}
    return _apply_substitutions(config_data, substitutions, source_keys=set(substitutions))


def _apply_substitutions(obj: Any, subs: dict, source_keys: set) -> Any:
    """Walk *obj* and return a copy with ``{key}`` interpolated in strings."""
    if isinstance(obj, dict):
        return {
            k: v if k in source_keys else _apply_substitutions(v, subs, source_keys)
            for k, v in obj.items()
        }
    if isinstance(obj, list):
        return [_apply_substitutions(item, subs, source_keys) for item in obj]
    if isinstance(obj, str):
        # Resolve every ``{key}`` that has a known substitution. Leave
        # unknown placeholders literal so configs can mix templated and
        # untemplated strings without one aborting the other.
        class _SafeDict(dict):
            def __missing__(self, key):  # type: ignore[override]
                return "{" + key + "}"

        try:
            return obj.format_map(_SafeDict(**subs))
        except (KeyError, IndexError, ValueError):
            return obj
    return obj


def get_log_level(yaml_path: str) -> int:
    """Return the logging level integer from the top-level ``log_level`` key in config.yaml."""
    with open(yaml_path, encoding="utf-8") as f:
        config_data = yaml.safe_load(f)
    level_name = config_data.get("log_level", "INFO").upper()
    return getattr(logging, level_name, logging.INFO)


@dataclass
class DetectorConfig:
    """Configuration for the person detection library.

    Attributes:
        models_path: Path to directory containing ONNX models.
        detection_threshold: Confidence threshold for person detection.
        tracking_score_threshold: Score threshold for tracking association.
        tracking_min_hits: Minimum consecutive hits to confirm a track.
        tracking_max_time_lost: Max frames before removing a lost track.
        pose_keypoint_threshold: Keypoint confidence threshold.
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

        Args:
            yaml_path: Path to YAML configuration file.

        Returns:
            DetectorConfig: Loaded configuration instance.

        Raises:
            ValueError: If required configuration keys are missing.
        """
        with open(yaml_path, encoding="utf-8") as f:
            config_data = _resolve_path_templates(yaml.safe_load(f))

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
    """Configuration for person-clip extraction (extract_person_clips_from_videos.py)."""

    input_dir: str
    output_clips_dir: str
    min_clip_duration_seconds: float
    max_clip_duration_seconds: float
    min_consecutive_frames: int
    merge_gap_frames: int
    require_face_visibility: bool
    min_face_size_percent: float
    min_face_visible_frames: int
    # Longest unbroken run of face-visible frames required to keep a segment
    min_consecutive_face_frames: int = 5

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
    max_detection_aspect_ratio: float = (
        3.0  # width/height > 3 likely furniture or animal, not person
    )
    max_track_overlap_iou: float = 0.5  # tracklets that overlap above this IoU are suppressed
    # Performance: copy each source video to a LOCAL cache dir before detection/clip
    # extraction, so cv2 + moviepy read from local SSD instead of frame-by-frame over a
    # network share (GPU-starving I/O). Opt-in; default off = unchanged behavior. The cache
    # dir MUST be local and outside input_dir. Copy failure raises (no silent fallback).
    preload_source_to_local: bool = False
    local_cache_dir: str | None = None
    # Performance: decode frames ahead in a producer thread so GPU inference overlaps with
    # cv2 decode (webm VP8/VP9 decode is CPU-bound and starves the GPU otherwise). Opt-in.
    readahead_decode: bool = False
    readahead_queue_frames: int = 32
    # Performance: extract the N clips of a source concurrently (ThreadPoolExecutor over
    # moviepy/ffmpeg subprocesses). The clips are independent (disjoint frame ranges), so the
    # serial per-clip extraction phase overlaps. Sidecar/log writes stay serialized in order in
    # the main thread (no CSV race). Opt-in; behavior-preserving.
    parallel_clip_extraction: bool = False
    max_extraction_workers: int = 3

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "ClipExtractionConfig":
        """Load configuration from a YAML file.

        :raises ValueError: If required configuration keys are missing.
        """
        with open(yaml_path, encoding="utf-8") as f:
            config_data = _resolve_path_templates(yaml.safe_load(f))

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
            preload_source_to_local=cfg.get("preload_source_to_local", False),
            local_cache_dir=cfg.get("local_cache_dir", None),
            readahead_decode=cfg.get("readahead_decode", False),
            readahead_queue_frames=cfg.get("readahead_queue_frames", 32),
            parallel_clip_extraction=cfg.get("parallel_clip_extraction", False),
            max_extraction_workers=cfg.get("max_extraction_workers", 3),
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
    def from_yaml(
        cls, yaml_path: str, section: str = "face_quality_filtering"
    ) -> "FaceQualityFilterConfig":
        """Load configuration from a YAML file.

        :param section: Top-level YAML key to read from (default: 'face_quality_filtering').
                        Pass 'image_face_quality_filtering' for the image pipeline.
        :raises ValueError: If required configuration keys are missing.
        """
        with open(yaml_path, encoding="utf-8") as f:
            config_data = _resolve_path_templates(yaml.safe_load(f))

        if section not in config_data:
            raise ValueError(f"Missing '{section}' section in config")

        cfg = config_data[section]

        def get_required(key: str):
            if key not in cfg:
                raise ValueError(f"Missing required config key: {section}.{key}")
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

    Produces 616×616 OFIQ-aligned crops written flat into output_dir/ (no subdirs).
    ArcFace 112×112 crops are extracted on-the-fly from each OFIQ frame by downstream
    scripts using the constant face_crop_corners_arcface field in the sidecar JSON.
    """

    input_dir: str
    output_dir: str
    detection_threshold: float
    pose_keypoint_threshold: float
    min_eye_distance_px: float
    min_track_face_frames: int
    skip_no_face_frames: bool
    detections_dir: str | None = None  # image pipeline only: dir with detection JSONs
    gpu_id: int = 0
    models_path: str = DEFAULT_MODELS_PATH
    min_free_disk_gb: float = 2.0
    include_audio: bool = True
    max_overlap_iou: float = 0.3

    @classmethod
    def from_yaml(cls, yaml_path: str, section: str = "face_crop_extraction") -> "FaceCropConfig":
        """Load configuration from a YAML file.

        :param section: Top-level YAML key to read from (default: 'face_crop_extraction').
                        Pass 'image_face_crop_extraction' for the image pipeline.
        :raises ValueError: If required configuration keys are missing.
        """
        with open(yaml_path, encoding="utf-8") as f:
            config_data = _resolve_path_templates(yaml.safe_load(f))

        if section not in config_data:
            raise ValueError(f"Missing '{section}' section in config")

        cfg = config_data[section]

        def get_required(key: str):
            if key not in cfg:
                raise ValueError(f"Missing required config key: face_crop_extraction.{key}")
            return cfg[key]

        return cls(
            input_dir=get_required("input_dir"),
            output_dir=get_required("output_dir"),
            detections_dir=cfg.get("detections_dir", None),
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
class FrameExtractionConfig:
    """Configuration for frame extraction (extract_frames_from_videos.py)."""

    input_dir: str
    output_dir: str
    overwrite: bool = False

    @staticmethod
    def _infer_type_from_folder(input_dir: str) -> str:
        """Infer clip type from the input directory name.

        Returns 'filtered_face_crop', 'face_crop', or 'person_clip' (default).
        """
        folder_name = Path(input_dir).name.lower()
        if "filtered" in folder_name:
            return "filtered_face_crop"
        if "face" in folder_name:
            return "face_crop"
        return "person_clip"  # default

    def get_type(self) -> str:
        """Return the inferred clip type (see _infer_type_from_folder)."""
        return self._infer_type_from_folder(self.input_dir)

    @classmethod
    def from_yaml(cls, config_path: str) -> "FrameExtractionConfig":
        """Load configuration from YAML file."""
        with open(config_path, encoding="utf-8") as f:
            config = _resolve_path_templates(yaml.safe_load(f))
        frame_config = config.get("frame_extraction", {})
        return cls(
            input_dir=frame_config.get("input_dir", "DARD/extracted_person_clips"),
            output_dir=frame_config.get("output_dir", "DARD/extracted_frames"),
            overwrite=frame_config.get("overwrite", False),
        )


@dataclass
class AudioTranscriptionConfig:
    """Configuration for audio transcription (transcribe_audio_files.py)."""

    audio_files_dir: str
    output_dir: str
    overwrite: bool = False

    @classmethod
    def from_yaml(cls, config_path: str) -> "AudioTranscriptionConfig":
        """Load configuration from YAML file."""
        with open(config_path, encoding="utf-8") as f:
            config = _resolve_path_templates(yaml.safe_load(f))
        cfg = config.get("audio_transcription", {})
        return cls(
            audio_files_dir=cfg.get("audio_files_dir", "DARD/archive_org_public_domain/audio"),
            output_dir=cfg.get("output_dir", "DARD/audio_transcriptions"),
            overwrite=cfg.get("overwrite", False),
        )


@dataclass
class VideoTranscriptionConfig:
    """Configuration for video clip transcription (transcribe_video_clips.py)."""

    person_clips_dir: str
    overwrite: bool = False

    @classmethod
    def from_yaml(cls, config_path: str) -> "VideoTranscriptionConfig":
        """Load configuration from YAML file."""
        with open(config_path, encoding="utf-8") as f:
            config = _resolve_path_templates(yaml.safe_load(f))
        trans_config = config.get("transcription", {})
        return cls(
            person_clips_dir=trans_config.get("person_clips_dir", "DARD/extracted_person_clips"),
            overwrite=trans_config.get("overwrite", False),
        )


@dataclass
class ImageExtractionConfig:
    """Configuration for image person detection."""

    input_dir: str
    output_detections_dir: str
    overwrite: bool = False
    detection_threshold: float = 0.5
    min_face_size_percent: float = 2.0
    require_face_visibility: bool = True
    require_frontal_face: bool = True
    frontal_symmetry_threshold: float = 0.3
    pose_keypoint_threshold: float = 0.4
    max_bbox_area_percent: float = 60.0
    max_detection_aspect_ratio: float = 2.0

    @classmethod
    def from_yaml(cls, config_path: str) -> "ImageExtractionConfig":
        """Load configuration from YAML file."""
        with open(config_path, encoding="utf-8") as f:
            config = _resolve_path_templates(yaml.safe_load(f))
        img_cfg = config.get("image_extraction", {})
        return cls(
            input_dir=img_cfg.get("input_dir", "DARD/archive_org_public_domain/images"),
            output_detections_dir=img_cfg.get(
                "output_detections_dir", "DARD/extracted_image_detections"
            ),
            overwrite=img_cfg.get("overwrite", False),
            detection_threshold=img_cfg.get("detection_threshold", 0.5),
            min_face_size_percent=img_cfg.get("min_face_size_percent", 2.0),
            require_face_visibility=img_cfg.get("require_face_visibility", True),
            require_frontal_face=img_cfg.get("require_frontal_face", True),
            frontal_symmetry_threshold=img_cfg.get("frontal_symmetry_threshold", 0.3),
            pose_keypoint_threshold=img_cfg.get("pose_keypoint_threshold", 0.4),
            max_bbox_area_percent=img_cfg.get("max_bbox_area_percent", 60.0),
            max_detection_aspect_ratio=img_cfg.get("max_detection_aspect_ratio", 2.0),
        )


@dataclass
class DocumentPreprocessConfig:
    input_dir: str
    output_dir: str
    overwrite: bool = False
    min_text_length: int = 50
    enable_ocr: bool = True
    gpu_id: int = 0
    ocr_languages: list[str] | None = None

    @classmethod
    def from_yaml(cls, config_path: str) -> "DocumentPreprocessConfig":
        with open(config_path, encoding="utf-8") as f:
            config = _resolve_path_templates(yaml.safe_load(f))
        cfg = config.get("document_preprocessing", {})
        gpu_id = config.get("gpu_id", 0)  # Global GPU setting
        return cls(
            input_dir=cfg.get("input_dir", "DARD/archive_org_public_domain/texts"),
            output_dir=cfg.get("output_dir", "DARD/preprocessed_documents"),
            overwrite=cfg.get("overwrite", False),
            min_text_length=cfg.get("min_text_length", 50),
            enable_ocr=cfg.get("enable_ocr", True),
            gpu_id=gpu_id,
            ocr_languages=cfg.get("ocr_languages", None),
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
    def from_yaml(
        cls, yaml_path: str, section: str = "face_quality_annotation"
    ) -> "FaceQualityAnnotationConfig":
        """Load configuration from a YAML file.

        :param section: Top-level YAML key to read from (default: 'face_quality_annotation').
                        Pass 'image_face_quality_annotation' for the image pipeline.
        :raises ValueError: If required configuration keys are missing.
        """
        with open(yaml_path, encoding="utf-8") as f:
            config_data = _resolve_path_templates(yaml.safe_load(f))

        if section not in config_data:
            raise ValueError(f"Missing '{section}' section in config")

        cfg = config_data[section]

        if "input_dir" not in cfg:
            raise ValueError(f"Missing required config key: {section}.input_dir")

        return cls(
            input_dir=cfg["input_dir"],
            gpu_id=cfg.get("gpu_id", config_data.get("gpu_id", 0)),
            frame_stride=cfg.get("frame_stride", 5),
            max_frames=cfg.get("max_frames", 30),
            overwrite=cfg.get("overwrite", False),
        )
