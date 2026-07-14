#!/usr/bin/env python3
"""
Annotate face crop videos and images with OFIQ-based quality measures.

Reads face crops from BOTH unfiltered and filtered directories:
  - Video crops: DARD/video_face_crops + DARD/filtered_video_face_crops
  - Image crops: DARD/image_face_crops + DARD/filtered_image_face_crops

For each crop:
  1. If .magface.json does NOT exist → compute and save it
  2. Compute OFIQ measures (if not already done or if overwrite=True)
  3. Write .ofiq_attr.json with OFIQ scores only

Generated files:
  .magface.json        MagFace IResNet50 per-frame and aggregated scores (computed once, reused)
  .ofiq_attr.json      OFIQ measures: sharpness, compression_artifacts, expression_neutrality,
                       no_head_coverings, face_occlusion, head_pose (per-frame and aggregated)

OFIQ measures (per measure):
  - Each measure: {max, mean, p10, p50, p90}
  - Head-pose: Additionally stores raw angles (yaw/pitch/roll, degrees) and their quality scores
  - All include provenance: face_crop_video, face_crop_json, source_video, annotated_at, annotator

Atomic writes: Computed data is written to temp file, then renamed to final location
to avoid corruption from interruptions.
"""

import argparse
import json
import logging
import os
import sys
import tempfile
from pathlib import Path

import onnxruntime as ort
from tqdm import tqdm

from dardcollect.pipeline_utils import _TqdmHandler
from dardcollect.quality import (
    _score_and_append,
    aggregate_frame_scores,
    load_models,
    score_all_magface_frames,
)

# ── Logging ───────────────────────────────────────────────────────────────────


_handler = _TqdmHandler()
_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
logging.basicConfig(handlers=[_handler], level=logging.INFO, force=True)
logger = logging.getLogger(__name__)

CONFIG_PATH = Path(
    os.environ.get(
        "DARDCOLLECT_CONFIG",
        Path(__file__).resolve().parent.parent / "configs" / "config.archive_all.yaml",
    )
)
DEFAULT_MODELS_DIR = Path(__file__).resolve().parent.parent / "dardcollect" / "models"

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dardcollect.config import FaceCropConfig, FaceQualityAnnotationConfig, get_log_level
from dardcollect.gpu_setup import setup_gpu_paths
from dardcollect.onnx_utils import get_preferred_providers

setup_gpu_paths(str(CONFIG_PATH))


# ── Helper functions ──────────────────────────────────────────────────────────


def _write_atomically(data: dict, output_path: Path) -> bool:
    """Write JSON data atomically: temp file → rename.

    Returns True if successful, False otherwise.
    Avoids partial writes from interruptions.
    """
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".json",
            dir=output_path.parent,
            delete=False,
            encoding="utf-8",
        ) as tf:
            temp_path = Path(tf.name)
            json.dump(data, tf, indent=2)
        temp_path.replace(output_path)
        return True
    except Exception as exc:
        logger.error("Failed to write %s: %s", output_path.name, exc)
        if temp_path and temp_path.exists():
            try:
                temp_path.unlink()
            except Exception:
                pass
        return False


def _ensure_magface_json(crop_path: Path, models, video_cfg) -> bool:
    """Ensure .magface.json exists. If missing, compute and save it.

    Returns True if .magface.json now exists (newly created or already existed).
    Returns False if computation failed.
    """
    magface_path = crop_path.with_suffix(".magface.json")

    # If exists and valid, skip
    if magface_path.exists():
        try:
            with open(magface_path, encoding="utf-8") as f:
                json.load(f)
            logger.debug("  .magface.json already exists: %s", crop_path.name)
            return True
        except Exception as exc:
            logger.warning("  .magface.json corrupted, will recompute: %s", exc)

    # Compute MagFace scores
    logger.info("  → Computing MagFace scores...")
    try:
        magface_data = score_all_magface_frames(crop_path, models.magface)
        if magface_data is None:
            logger.error("  Failed to compute MagFace for %s", crop_path.name)
            return False

        # Write atomically
        success = _write_atomically(magface_data, magface_path)
        if success:
            logger.info("  ✓ Saved .magface.json")
        return success
    except Exception as exc:
        logger.error("  Error computing MagFace: %s", exc)
        return False


def _generate_ofiq_attr_json(crop_path: Path, models, cfg) -> bool:
    """Compute OFIQ measures and save to .ofiq_attr.json atomically.

    Returns True if .ofiq_attr.json was written, False otherwise.
    """
    from dardcollect.face_geometry import arcface_from_ofiq_frame
    from dardcollect.fair import add_fair_metadata, reorganize_for_fair, validate_against_schema
    from dardcollect.pipeline_utils import _get_frames_from_crop
    from dardcollect.provenance import now_iso

    ofiq_attr_path = crop_path.with_suffix(".ofiq_attr.json")

    # Check if already done (unless overwrite=True)
    if not cfg.overwrite and ofiq_attr_path.exists():
        try:
            with open(ofiq_attr_path, encoding="utf-8") as f:
                json.load(f)
            logger.debug("  .ofiq_attr.json already exists, skipping: %s", crop_path.name)
            return False  # Already done, nothing to update
        except Exception as exc:
            logger.warning("  .ofiq_attr.json corrupted, will recompute: %s", exc)

    logger.info("  → Computing OFIQ measures...")

    # Read sidecar for provenance; OFIQ quality sidecars require parent_crop.
    sidecar_path = crop_path.with_suffix(".json")
    source_video = ""
    has_arcface_annotation = False
    parent_uuid = None
    sidecar_data = {}
    if not sidecar_path.exists():
        logger.info("  Missing sidecar JSON for %s — skipping OFIQ annotation", crop_path.name)
        return False

    try:
        with open(sidecar_path, encoding="utf-8") as f:
            sidecar_data = json.load(f)
        source_video = sidecar_data.get("source_video", "")
        parent_uuid = sidecar_data.get("uuid")
        has_arcface_annotation = (
            sidecar_data.get("crop_format") == "ofiq"
            or "arcface_crop_corners_in_ofiq" in sidecar_data
        )
    except Exception as exc:
        logger.warning("  Could not read sidecar for %s: %s", crop_path.name, exc)
        return False

    if not isinstance(parent_uuid, str) or not parent_uuid:
        logger.warning(
            "  Sidecar missing parent UUID for %s — skipping OFIQ annotation",
            crop_path.name,
        )
        return False

    # Get frames
    frames = _get_frames_from_crop(crop_path)
    if not frames:
        logger.warning("  Cannot read frames from %s", crop_path.name)
        return False

    # Score frames
    frame_scores: list[dict] = []
    frame_idx = 0

    for ofiq_frame in frames:
        arcface_frame = arcface_from_ofiq_frame(ofiq_frame) if has_arcface_annotation else None
        if frame_idx % cfg.frame_stride == 0:
            _score_and_append(
                ofiq_frame, arcface_frame, frame_idx, crop_path.name, models, frame_scores
            )
            if len(frame_scores) % 10 == 0:
                logger.info("    (sampled %d frames so far...)", len(frame_scores))
            frame_idx += 1
            if cfg.max_frames > 0 and len(frame_scores) >= cfg.max_frames:
                break

    if not frame_scores:
        logger.warning("  No frames scored for %s", crop_path.name)
        return False

    # Build OFIQ-only data (no MagFace)
    ofiq_data: dict = {
        "face_crop_video": crop_path.name,
        "face_crop_json": sidecar_path.name,
        "source_video": source_video,
        "annotated_at": now_iso(),
        "annotator": "pipeline/annotate_face_quality.py",
        "frame_stride": cfg.frame_stride,
        "max_frames_sampled": cfg.max_frames,
        "frame_data": frame_scores,
        **aggregate_frame_scores(frame_scores),
    }

    # Add FAIR metadata
    try:
        add_fair_metadata(
            ofiq_data,
            schema_type="quality_annotation",
            parent_uuid=parent_uuid,
            parent_file=sidecar_path.name,
        )
        ofiq_data = reorganize_for_fair(ofiq_data, schema_type="quality_annotation")
    except Exception as exc:
        logger.warning("  Could not add FAIR metadata: %s", exc)

    # Validate the FAIR sidecar against the ratified schema before write
    # (per the project's "validate at write" contract).
    try:
        validate_against_schema(ofiq_data, "quality_annotation")
    except Exception as exc:
        logger.error("  OFIQ sidecar failed schema validation for %s: %s", crop_path.name, exc)
        return False

    # Write atomically
    success = _write_atomically(ofiq_data, ofiq_attr_path)
    if success:
        logger.info("  ✓ Saved .ofiq_attr.json")
    return success


# ── Entry point ───────────────────────────────────────────────────────────────


def _load_annotation_configs(
    config_path: str,
) -> tuple[list[tuple[str, FaceQualityAnnotationConfig, FaceCropConfig, str]], int]:
    """Load face-quality-annotation configs for whichever modalities are present.

    Either video or image may be absent in a single-modality config
    (media_types = video-only or image-only); the missing modality is skipped
    with an info log. A missing *section* is the single-modality case; any other
    ValueError (e.g. a missing required key in a present section) is a real
    config error and re-raised.

    Returns ``(configs, gpu_id)``. Exits 1 if neither modality is configured.
    """
    configs: list[tuple[str, FaceQualityAnnotationConfig, FaceCropConfig, str]] = []
    gpu_id: int | None = None

    try:
        video_cfg = FaceQualityAnnotationConfig.from_yaml(config_path)
        video_face_crop_cfg = FaceCropConfig.from_yaml(config_path)
        configs.append(("video", video_cfg, video_face_crop_cfg, "video_face_crops_extraction.csv"))
        gpu_id = video_cfg.gpu_id
    except ValueError as exc:
        msg = str(exc)
        if "section in config" in msg and (
            "'face_quality_annotation'" in msg or "'face_crop_extraction'" in msg
        ):
            logger.info(
                "Video modality config not found in %s — "
                "skipping video face quality annotation (image-only config)",
                config_path,
            )
        else:
            raise

    try:
        image_cfg = FaceQualityAnnotationConfig.from_yaml(
            config_path, section="image_face_quality_annotation"
        )
        image_face_crop_cfg = FaceCropConfig.from_yaml(
            config_path, section="image_face_crop_extraction"
        )
        configs.append(("image", image_cfg, image_face_crop_cfg, "image_face_crops_extraction.csv"))
        if gpu_id is None:
            gpu_id = image_cfg.gpu_id
    except ValueError as exc:
        msg = str(exc)
        if "section in config" in msg and "image_" in msg:
            logger.info(
                "Image modality config not found in %s — "
                "skipping image face quality annotation (video-only config)",
                config_path,
            )
        else:
            raise

    if not configs:
        logger.error(
            "No face quality annotation config found in %s "
            "(neither video nor image sections present)",
            config_path,
        )
        sys.exit(1)

    assert gpu_id is not None  # configs non-empty ⇒ at least one modality set gpu_id
    return configs, gpu_id


def _process_one_pass(configs, models) -> int:
    """Scan all configured face crops and annotate any pending ones (idempotent:
    already-annotated crops are skipped inside _generate_ofiq_attr_json). Returns
    the number of crops processed (MagFace computed) this pass. Used by both the
    one-shot path (single call) and the progressive worker (looped).
    """
    processed = 0
    for modality, cfg, face_crop_cfg, crops_csv_name in configs:
        # Determine input directories
        if modality == "video":
            input_dirs = [
                Path(cfg.input_dir),
                Path(cfg.input_dir).parent / "filtered_video_face_crops",
            ]
        else:
            input_dirs = [
                Path(cfg.input_dir),
                Path(cfg.input_dir).parent / "filtered_image_face_crops",
            ]

        # Find all crops from both directories (recursively: face crops may
        # live in per-source-subdir trees produced by extract_face_crops_from_videos)
        crop_files = []
        for input_dir in input_dirs:
            if input_dir.exists():
                crop_files.extend(sorted(input_dir.rglob("*_face_*.mp4")))
                crop_files.extend(sorted(input_dir.rglob("*_face_*.jpg")))
                crop_files.extend(sorted(input_dir.rglob("*_face_*.png")))
        crop_files = sorted(set(crop_files))
        crop_files = [p for p in crop_files if not p.name.endswith("_mask.png")]

        if not crop_files:
            logger.info("[%s] No face crops found", modality)
            continue

        logger.info("[%s] Found %d face crop(s)", modality, len(crop_files))

        magface_created = 0
        ofiq_created = 0
        ofiq_skipped = 0
        errors = 0

        for crop_path in tqdm(crop_files, desc=f"Annotating OFIQ ({modality})", unit="crop"):
            logger.info("[%s] Processing: %s", modality, crop_path.name)

            try:
                # Step 1: Ensure .magface.json exists
                if not _ensure_magface_json(crop_path, models, cfg):
                    logger.warning("  Skipping OFIQ annotation (MagFace failed)")
                    errors += 1
                    continue
                magface_created += 1

                # Step 2: Compute and save .ofiq_attr.json
                if _generate_ofiq_attr_json(crop_path, models, cfg):
                    ofiq_created += 1
                else:
                    ofiq_skipped += 1

            except Exception as exc:
                logger.error("[%s] Error processing %s: %s", modality, crop_path.name, exc)
                errors += 1

        logger.info(
            "[%s] Done. MagFace: %d  OFIQ: %d created, %d skipped  Errors: %d",
            modality,
            magface_created,
            ofiq_created,
            ofiq_skipped,
            errors,
        )
        processed += magface_created

    return processed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "config_path",
        nargs="?",
        default=str(CONFIG_PATH),
        help="Path to config.yaml (default: project root config.yaml)",
    )
    args = parser.parse_args()

    config_path = args.config_path
    logging.getLogger().setLevel(get_log_level(config_path))

    logger.info("=" * 60)
    logger.info("ONNX Runtime available providers: %s", ort.get_available_providers())
    logger.info("=" * 60)

    # Load configs for whichever modalities are present (video and/or image).
    # A single-modality config skips the missing one; see _load_annotation_configs.
    configs, gpu_id = _load_annotation_configs(config_path)

    try:
        models = load_models(DEFAULT_MODELS_DIR, gpu_id)
    except Exception as exc:
        logger.error("Failed to load models: %s", exc)
        sys.exit(1)

    providers = get_preferred_providers(gpu_id)
    using_trt = any("TensorrtExecutionProvider" in str(p) for p in providers)

    logger.info("Starting OFIQ annotation...")
    if using_trt:
        logger.info(
            "⏳ First crop may take longer — TensorRT is compiling GPU engines in the background"
        )

    # Models are loaded ONCE above. The orchestrator defers this stage's launch
    # until its deps finish (DEFER_UNTIL_DEPS_DONE), so this one-shot call processes
    # all available crops in a single pass — one model load, no re-launch waste.
    _process_one_pass(configs, models)

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
                logger.info("  Subsequent runs will be much faster")


if __name__ == "__main__":
    main()
