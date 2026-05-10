#!/usr/bin/env python3
"""
Annotate face crop videos and images with OFIQ-based face quality scores.

Reads OFIQ face crops from an output folder of extract_face_crops_from_videos.py or
extract_face_crops_from_images.py (or their upstream filters), computes the following
quality measures following ISO/IEC 29794-5 (OFIQ), and writes a sibling .quality.json
file next to each crop (leaves the extraction-stage sidecar .json untouched):

  unified_score           MagFace IResNet50 magnitude (OFIQ UnifiedQualityScore)
  sharpness               Laplacian/Sobel RTrees (OFIQ Sharpness)
  compression_artifacts   SSIM CNN (OFIQ CompressionArtifacts)
  expression_neutrality   HSEmotion EfficientNet-B0/B2 + AdaBoost (OFIQ ExpressionNeutrality)
  no_head_coverings       BiSeNet face parsing — hat/cloth pixel fraction (OFIQ NoHeadCoverings)
  face_occlusion          FaceOcclusionSegmentation CNN (OFIQ FaceOcclusionPrevention)
  head_pose               MobileNetV1 3DDFAV2 — yaw/pitch/roll angles + cosine² quality scores

Each measure is summarised per crop as {max, mean, p10, p50, p90}.
Head-pose additionally stores the raw angles (degrees, signed) and their quality scores.

The .quality.json carries provenance fields (face_crop_video, face_crop_json,
source_video, annotated_at, annotator) so the origin chain from quality data →
face crop → source video/image is always traceable.

Pass the extract_face_crops_from_videos.py or extract_face_crops_from_images.py
output directory as the input folder (e.g. DARD/face_crops/ or DARD/filtered_face_crops/).
Those crops are 616×616 OFIQ-aligned, matching the format expected by all quality models.

MagFace (unified_score) requires the ArcFace 112×112 format.  Because both crop
formats align to fixed canonical landmark positions, the ArcFace region is always
the same parallelogram within any OFIQ frame.  The script extracts 112×112 crops
from OFIQ frames on-the-fly for any sidecar with crop_format == "ofiq" (all
output of extract_face_crops_from_videos.py and extract_face_crops_from_images.py).
If the field is absent (old-format sidecar), unified_score is omitted from the output JSON.
"""

import json
import logging
import sys
from pathlib import Path

import onnxruntime as ort
from tqdm import tqdm

from dardcollect.pipeline_utils import _TqdmHandler
from dardcollect.quality import (
    load_models,
    score_video,
)

# ── Logging ───────────────────────────────────────────────────────────────────


_handler = _TqdmHandler()
_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
logging.basicConfig(handlers=[_handler], level=logging.INFO, force=True)
logger = logging.getLogger(__name__)

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.yaml"
DEFAULT_MODELS_DIR = Path(__file__).resolve().parent.parent / "dardcollect" / "models"

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dardcollect.config import FaceCropConfig, FaceQualityAnnotationConfig, get_log_level
from dardcollect.gpu_setup import setup_gpu_paths
from dardcollect.onnx_utils import get_preferred_providers
from dardcollect.pipeline_loggers import FaceQualityAnnotationLogger

setup_gpu_paths(str(CONFIG_PATH))


# ── Per-video processing ──────────────────────────────────────────────────────


# ── Back-propagation to person clip sidecars ──────────────────────────────────

_QUALITY_PROVENANCE_KEYS = {
    "face_crop_video",
    "face_crop_json",
    "source_video",
    "annotated_at",
    "annotator",
    "frame_stride",
    "max_frames_sampled",
}


def _backpropagate_quality(face_crop_path: Path, quality_data: dict) -> None:
    """Write a quality summary into the source person clip's sidecar JSON.

    Reads source_video and track_id from the face crop sidecar, then inserts
    face_quality[track_id] into the person clip sidecar so the viewer can show
    per-track quality scores when browsing person clips.
    """
    sidecar_path = face_crop_path.with_suffix(".json")
    if not sidecar_path.exists():
        return

    try:
        with open(sidecar_path, encoding="utf-8") as f:
            sidecar = json.load(f)
    except Exception as exc:
        logger.warning("Cannot read face crop sidecar %s: %s", sidecar_path.name, exc)
        return

    source_video = sidecar.get("source_video", "")
    track_id = sidecar.get("track_id")
    if not source_video or track_id is None:
        logger.debug(
            "No source_video/track_id in %s — skipping back-propagation", sidecar_path.name
        )
        return

    clip_sidecar = Path(source_video).with_suffix(".json")
    if not clip_sidecar.exists():
        logger.debug("Person clip sidecar not found: %s", clip_sidecar)
        return

    try:
        with open(clip_sidecar, encoding="utf-8") as f:
            clip_data = json.load(f)
    except Exception as exc:
        logger.warning("Cannot read clip sidecar %s: %s", clip_sidecar.name, exc)
        return

    summary = {k: v for k, v in quality_data.items() if k not in _QUALITY_PROVENANCE_KEYS}
    summary["face_crop"] = face_crop_path.name

    if "face_quality" not in clip_data:
        clip_data["face_quality"] = {}
    clip_data["face_quality"][str(track_id)] = summary

    try:
        with open(clip_sidecar, "w", encoding="utf-8") as f:
            json.dump(clip_data, f, indent=2)
        logger.debug("Updated face_quality[%d] in %s", track_id, clip_sidecar.name)
    except Exception as exc:
        logger.warning("Cannot write clip sidecar %s: %s", clip_sidecar.name, exc)


# ── Entry point ───────────────────────────────────────────────────────────────


def main(config_path: str | None = None) -> None:
    if config_path is None:
        config_path = str(CONFIG_PATH)

    cfg = FaceQualityAnnotationConfig.from_yaml(config_path)
    face_crop_cfg = FaceCropConfig.from_yaml(config_path)
    logging.getLogger().setLevel(get_log_level(config_path))

    logger.info("=" * 60)
    logger.info("ONNX Runtime available providers: %s", ort.get_available_providers())
    logger.info("=" * 60)

    input_dir = Path(cfg.input_dir)
    if not input_dir.exists():
        logger.error("Input directory does not exist: %s", input_dir)
        sys.exit(1)

    # Find both video and image crops
    crop_files = sorted(input_dir.glob("*_face_*.mp4"))
    crop_files.extend(sorted(input_dir.glob("*_face_*.jpg")))
    crop_files.extend(sorted(input_dir.glob("*_face_*.png")))
    crop_files = sorted(set(crop_files))  # Remove duplicates and re-sort

    if not crop_files:
        logger.error("No face crops (videos or images) found in %s", input_dir)
        sys.exit(1)

    logger.info("Found %d face crop(s) in %s", len(crop_files), input_dir)

    try:
        models = load_models(DEFAULT_MODELS_DIR, cfg.gpu_id)
    except Exception as exc:
        logger.error("Failed to load models: %s", exc)
        sys.exit(1)

    updated = 0
    skipped = 0

    # Initialize quality annotation logger
    face_crops_csv = Path(face_crop_cfg.output_dir) / "face_crops_extraction.csv"
    quality_logger = FaceQualityAnnotationLogger(
        output_dir=str(input_dir), face_crops_csv_path=face_crops_csv
    )

    # Check if TensorRT is enabled for warning
    providers = get_preferred_providers(cfg.gpu_id)
    using_trt = any("TensorrtExecutionProvider" in str(p) for p in providers)

    logger.info("Starting annotation...")
    if using_trt:
        logger.info(
            "⏳ First video may take longer — TensorRT is compiling GPU engines in the background"
        )

    for crop_path in tqdm(crop_files, desc="Annotating quality", unit="crop"):
        logger.info("Processing: %s", crop_path.name)
        try:
            quality_data = score_video(
                crop_path,
                models,
                frame_stride=cfg.frame_stride,
                max_frames=cfg.max_frames,
                overwrite=cfg.overwrite,
            )
            if quality_data is not None:
                updated += 1
                _backpropagate_quality(crop_path, quality_data)

                # Log quality annotation (for traceability) — all values are max over frames
                head_pose = quality_data.get("head_pose", {})
                quality_logger.log_quality_annotation(
                    crop_path=str(crop_path),
                    sharpness=quality_data.get("sharpness", {}).get("max", 0.0),
                    compression_artifacts=quality_data.get("compression_artifacts", {}).get(
                        "max", 0.0
                    ),
                    expression_neutrality=quality_data.get("expression_neutrality", {}).get(
                        "max", 0.0
                    ),
                    no_head_coverings=quality_data.get("no_head_coverings", {}).get("max", 0.0),
                    face_occlusion_prevention=quality_data.get("face_occlusion_prevention", {}).get(
                        "max", 0.0
                    ),
                    unified_score=quality_data.get("unified_score", {}).get("max", 0.0),
                    yaw_quality=head_pose.get("yaw_quality", {}).get("max", 0.0),
                    pitch_quality=head_pose.get("pitch_quality", {}).get("max", 0.0),
                    roll_quality=head_pose.get("roll_quality", {}).get("max", 0.0),
                    passed_filter=True,
                )
            else:
                skipped += 1
        except Exception as exc:
            logger.error("Error processing %s: %s", crop_path.name, exc)

    # Check if TRT engines were created
    if using_trt:
        trt_cache_dir = Path(".cache/trt_engines")
        if trt_cache_dir.exists():
            engine_files = list(trt_cache_dir.glob("*.trt"))
            if engine_files:
                logger.info(
                    "✓ TensorRT engines cached: %d files in %s",
                    len(engine_files),
                    trt_cache_dir,
                )
                logger.info("  Subsequent runs will be much faster (using cached engines)")

    logger.info(
        "Done.  Written: %d  Skipped (already annotated or error): %d",
        updated,
        skipped,
    )
    quality_logger.print_summary()


if __name__ == "__main__":
    config_path = sys.argv[1] if len(sys.argv) > 1 and sys.argv[1].strip() else None
    main(config_path)
