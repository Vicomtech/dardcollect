"""
OFIQ-based face quality scoring primitives.

Implements the quality measures following ISO/IEC 29794-5 (OFIQ):
  unified_score           MagFace IResNet50 magnitude (OFIQ UnifiedQualityScore)
  sharpness               Laplacian/Sobel RTrees (OFIQ Sharpness)
  compression_artifacts   SSIM CNN (OFIQ CompressionArtifacts)
  expression_neutrality   HSEmotion EfficientNet-B0/B2 + AdaBoost (OFIQ ExpressionNeutrality)
  no_head_coverings       BiSeNet face parsing — hat/cloth pixel fraction (OFIQ NoHeadCoverings)
  face_occlusion          FaceOcclusionSegmentation CNN (OFIQ FaceOcclusionPrevention)
  head_pose               MobileNetV1 3DDFAV2 — yaw/pitch/roll angles + cosine² quality scores
"""

import gzip
import json
import logging
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

from dardcollect.quality_measures import (
    _angle_to_quality,
    _compression_score,
    _expression_neutrality_score,
    _face_occlusion_score,
    _head_pose_angles,
    _no_head_coverings_score,
    _sharpness_score,
)

logger = logging.getLogger(__name__)


# ── Model bundle ──────────────────────────────────────────────────────────────


@dataclass
class QualityModels:
    magface: ort.InferenceSession
    rtrees: cv2.ml.RTrees  # sharpness
    n_trees: int  # RTrees termination count (numTrees)
    compression: ort.InferenceSession  # ssim_248_model
    enet_b0: ort.InferenceSession  # expression neutrality CNN1
    enet_b2: ort.InferenceSession  # expression neutrality CNN2
    adaboost: cv2.ml.Boost  # expression neutrality classifier
    bisenet: ort.InferenceSession  # face parsing (head coverings)
    occlusion: ort.InferenceSession  # face occlusion segmentation
    headpose: ort.InferenceSession  # head pose


def load_models(models_dir: Path, gpu_id: int) -> QualityModels:
    from dardcollect.magface import load_magface
    from dardcollect.onnx_utils import create_ort_session, get_preferred_providers

    providers = get_preferred_providers(gpu_id)
    logger.info("Loading quality models from %s (GPU %d)...", models_dir, gpu_id)
    logger.info("  Using execution providers: %s", providers)

    # Check if TensorRT is being used
    using_trt = any("TensorrtExecutionProvider" in str(p) for p in providers)
    if using_trt:
        logger.info(
            "  ⚠️  TensorRT is enabled — first run will compile ONNX models to TensorRT engines"
        )
        logger.info(
            "     (this is slow but happens only once; subsequent runs will be much faster)"
        )

    def _onnx(name: str) -> ort.InferenceSession:
        p = models_dir / name
        if not p.exists():
            raise FileNotFoundError(f"Model not found: {p}")
        logger.info("  Loading %s...", name)
        sess = create_ort_session(p, providers)
        actual_providers = sess.get_providers()
        provider = actual_providers[0] if actual_providers else "CPU"
        logger.info("  ✓ Loaded %s (using: %s)", name, provider)
        return sess

    def _load_gz_opencv(name: str, loader):
        p = models_dir / name
        if not p.exists():
            raise FileNotFoundError(f"Model not found: {p}")
        logger.info("  Loading %s...", name)
        with gzip.open(p, "rb") as f:
            xml_bytes = f.read()
        with tempfile.NamedTemporaryFile(suffix=".xml", delete=False) as tf:
            tf.write(xml_bytes)
            tpath = tf.name
        try:
            model = loader(tpath)
        finally:
            os.unlink(tpath)
        logger.info("  ✓ Loaded %s", name)
        return model

    logger.info("  Loading MagFace...")
    magface = load_magface(gpu_id)
    logger.info("  ✓ Loaded MagFace")

    rtrees = _load_gz_opencv("face_sharpness_rtree.xml.gz", cv2.ml.RTrees_load)  # type: ignore
    n_trees = rtrees.getTermCriteria()[1]

    compression = _onnx("ssim_248_model.onnx")
    enet_b0 = _onnx("enet_b0_8_best_vgaf_embed_zeroed.onnx")
    enet_b2 = _onnx("enet_b2_8_embed_zeroed.onnx")
    adaboost = _load_gz_opencv("hse_1_2_C_adaboost.yml.gz", cv2.ml.Boost_load)  # type: ignore
    bisenet = _onnx("bisenet_400.onnx")
    occlusion = _onnx("face_occlusion_segmentation_ort.onnx")
    headpose = _onnx("mb1_120x120.onnx")

    logger.info("All models loaded successfully!")

    return QualityModels(
        magface=magface,
        rtrees=rtrees,
        n_trees=n_trees,
        compression=compression,
        enet_b0=enet_b0,
        enet_b2=enet_b2,
        adaboost=adaboost,
        bisenet=bisenet,
        occlusion=occlusion,
        headpose=headpose,
    )


# ── Per-frame scoring ─────────────────────────────────────────────────────────


def score_frame_all(
    ofiq_frame: np.ndarray,
    models: QualityModels,
    arcface_frame: np.ndarray | None = None,
) -> dict:
    """Run all quality measures on a single frame and return a dict of scores.

    :param ofiq_frame: 616×616 OFIQ-aligned frame for all quality measures except MagFace.
    :param models: Loaded quality models.
    :param arcface_frame: 112×112 ArcFace-aligned frame for MagFace.  If None,
        unified_score is omitted from the result.
    """
    from dardcollect.magface import score_frame as magface_score_frame

    yaw, pitch, roll = _head_pose_angles(ofiq_frame, models.headpose)
    result: dict = {
        "sharpness": _sharpness_score(ofiq_frame, models.rtrees, models.n_trees),
        "compression_artifacts": _compression_score(ofiq_frame, models.compression),
        "expression_neutrality": _expression_neutrality_score(
            ofiq_frame, models.enet_b0, models.enet_b2, models.adaboost
        ),
        "no_head_coverings": _no_head_coverings_score(ofiq_frame, models.bisenet),
        "face_occlusion_prevention": _face_occlusion_score(ofiq_frame, models.occlusion),
        "head_pose": {
            "yaw_deg": yaw,
            "pitch_deg": pitch,
            "roll_deg": roll,
            "yaw_quality": _angle_to_quality(yaw),
            "pitch_quality": _angle_to_quality(pitch),
            "roll_quality": _angle_to_quality(roll),
        },
    }
    if arcface_frame is not None:
        result["unified_score"] = magface_score_frame(models.magface, arcface_frame)
    return result


# ── Aggregation ───────────────────────────────────────────────────────────────


def _pct_stats(values: list[float]) -> dict:
    """Return {max, mean, p10, p50, p90} for a list of scalar quality scores."""
    arr = np.array(values, dtype=np.float64)
    return {
        "max": round(float(arr.max()), 3),
        "mean": round(float(arr.mean()), 3),
        "p10": round(float(np.percentile(arr, 10)), 3),
        "p50": round(float(np.percentile(arr, 50)), 3),
        "p90": round(float(np.percentile(arr, 90)), 3),
    }


def aggregate_frame_scores(frame_scores: list[dict]) -> dict:
    """Aggregate per-frame score dicts into a summary dict for the JSON."""
    scalar_keys = [
        "unified_score",
        "sharpness",
        "compression_artifacts",
        "expression_neutrality",
        "no_head_coverings",
        "face_occlusion_prevention",
    ]
    agg: dict = {"frames_scored": len(frame_scores)}
    for key in scalar_keys:
        vals = [s[key] for s in frame_scores if key in s]
        if vals:
            agg[key] = _pct_stats(vals)

    yaws = [s["head_pose"]["yaw_deg"] for s in frame_scores]
    pitches = [s["head_pose"]["pitch_deg"] for s in frame_scores]
    rolls = [s["head_pose"]["roll_deg"] for s in frame_scores]
    yaw_q = [s["head_pose"]["yaw_quality"] for s in frame_scores]
    pitch_q = [s["head_pose"]["pitch_quality"] for s in frame_scores]
    roll_q = [s["head_pose"]["roll_quality"] for s in frame_scores]

    agg["head_pose"] = {
        "yaw_deg": {
            "mean": round(float(np.mean(yaws)), 2),
            "abs_mean": round(float(np.mean(np.abs(yaws))), 2),
        },
        "pitch_deg": {
            "mean": round(float(np.mean(pitches)), 2),
            "abs_mean": round(float(np.mean(np.abs(pitches))), 2),
        },
        "roll_deg": {
            "mean": round(float(np.mean(rolls)), 2),
            "abs_mean": round(float(np.mean(np.abs(rolls))), 2),
        },
        "yaw_quality": _pct_stats(yaw_q),
        "pitch_quality": _pct_stats(pitch_q),
        "roll_quality": _pct_stats(roll_q),
    }
    return agg


# ── Per-crop scoring (moved from pipeline) ─────────────────────────────────────

# Track if we've logged the actual provider being used during inference
_provider_logged = False


def _score_and_append(
    ofiq_frame: np.ndarray,
    arcface_frame: np.ndarray | None,
    frame_idx: int,
    video_name: str,
    models: QualityModels,
    out: list,
) -> None:
    global _provider_logged
    try:
        frame_scores = score_frame_all(ofiq_frame, models, arcface_frame)
        frame_scores["frame_index"] = frame_idx
        out.append(frame_scores)

        # Log actual execution provider on first frame
        if not _provider_logged:
            _provider_logged = True
            providers = models.magface.get_providers()
            if providers:
                logger.info("  Actual execution provider during inference: %s", providers[0])
    except Exception as exc:
        logger.debug("Error scoring frame %d of %s: %s", frame_idx, video_name, exc)


def score_video(
    crop_path: Path,
    models: QualityModels,
    frame_stride: int,
    max_frames: int,
    overwrite: bool,
) -> dict | None:
    """Score a single OFIQ face crop (video or image) and write a sibling .quality.json file.

    Computes OFIQ measures (sharpness, expression, etc.) on all frames. MagFace
    (unified_score) is read from .magface.json if available (written by
    filter_face_crops_by_quality.py), or computed here as fallback if .magface.json
    is absent and crop_format == "ofiq".

    Returns the quality data dict if written, None if skipped or failed.
    """
    from dardcollect.face_geometry import arcface_from_ofiq_frame
    from dardcollect.fair import add_fair_metadata, reorganize_for_fair
    from dardcollect.pipeline_utils import _get_frames_from_crop
    from dardcollect.provenance import now_iso

    quality_path = crop_path.with_suffix(".quality.json")
    if not overwrite and quality_path.exists():
        logger.debug("Already annotated, skipping: %s", crop_path.name)
        return None

    sidecar_path = crop_path.with_suffix(".json")
    magface_path = crop_path.with_suffix(".magface.json")
    source_video = ""
    has_arcface_annotation = False
    sidecar_data: dict | None = None
    if sidecar_path.exists():
        try:
            with open(sidecar_path, encoding="utf-8") as f:
                sidecar_data = json.load(f)
            source_video = sidecar_data.get("source_video", "")
            has_arcface_annotation = (
                sidecar_data.get("crop_format") == "ofiq"
                or "arcface_crop_corners_in_ofiq" in sidecar_data  # legacy
            )
        except Exception as exc:
            logger.warning("Failed to read sidecar %s: %s", sidecar_path.name, exc)
    else:
        logger.warning(
            "No sidecar JSON alongside %s — provenance will be incomplete", crop_path.name
        )

    # Try to read pre-computed MagFace scores from .magface.json
    magface_unified_score: dict | None = None
    if magface_path.exists():
        try:
            with open(magface_path, encoding="utf-8") as f:
                magface_data = json.load(f)
            magface_unified_score = magface_data.get("unified_score")
            logger.debug("Read MagFace scores from %s", magface_path.name)
        except Exception as exc:
            logger.warning("Failed to read %s: %s", magface_path.name, exc)

    if not has_arcface_annotation:
        if magface_unified_score is None:
            logger.warning(
                "No crop_format in sidecar for %s and no .magface.json — "
                "unified_score will be omitted (re-run "
                "extract_face_crops_from_videos.py or extract_face_crops_from_images.py to fix)",
                crop_path.name,
            )

    frames = _get_frames_from_crop(crop_path)
    if not frames:
        logger.warning("Cannot read frames from %s", crop_path.name)
        return None

    frame_scores: list[dict] = []
    frame_idx = 0

    logger.info("  → Reading frames and computing quality scores...")
    for ofiq_frame in frames:
        arcface_frame: np.ndarray | None = (
            arcface_from_ofiq_frame(ofiq_frame) if has_arcface_annotation else None
        )
        if frame_idx % frame_stride == 0:
            _score_and_append(
                ofiq_frame, arcface_frame, frame_idx, crop_path.name, models, frame_scores
            )
            # Log progress every 10 frames sampled
            if len(frame_scores) % 10 == 0:
                logger.info("    (sampled %d frames so far...)", len(frame_scores))
            frame_idx += 1
            if max_frames > 0 and len(frame_scores) >= max_frames:
                break

    if not frame_scores:
        logger.warning("No frames scored for %s", crop_path.name)
        return None

    quality_data: dict = {
        "face_crop_video": crop_path.name,
        "face_crop_json": sidecar_path.name,
        "source_video": source_video,
        "annotated_at": now_iso(),
        "annotator": "pipeline/annotate_face_quality.py",
        "frame_stride": frame_stride,
        "max_frames_sampled": max_frames,
        "frame_data": frame_scores,  # Per-frame quality scores
        **aggregate_frame_scores(frame_scores),
    }

    # Include MagFace unified_score if available
    if magface_unified_score:
        quality_data["unified_score"] = magface_unified_score

    # Add FAIR metadata (UUID, schema version, parent crop link)
    parent_crop_uuid = sidecar_data.get("uuid") if sidecar_data else None
    quality_data = add_fair_metadata(
        quality_data,
        schema_type="quality_annotation",
        parent_uuid=parent_crop_uuid,
        parent_file=crop_path.name,
    )

    quality_data = reorganize_for_fair(quality_data, "quality_annotation")
    with open(quality_path, "w", encoding="utf-8") as f:
        json.dump(quality_data, f, indent=2)

    us_max = quality_data.get("unified_score", {}).get("max")
    logger.info(
        "  %s  → %d frames scored%s → %s",
        crop_path.name,
        len(frame_scores),
        f", unified_score max={us_max:.1f}" if us_max is not None else "",
        quality_path.name,
    )
    return quality_data


def score_all_magface_frames(
    crop_path: Path,
    session: ort.InferenceSession,
) -> dict:
    """Score all frames from a crop (video or image) with MagFace.

    Reads OFIQ 616×616 frames/image, extracts a 112×112 ArcFace crop from each using
    the precomputed constant region, then scores with MagFace. Returns aggregated
    statistics (min, max, mean, percentiles) plus per-frame scores.

    Args:
        crop_path: Path to the OFIQ face crop (.mp4 video or .jpg/.png image).
        session: Loaded MagFace ONNX session.

    Returns:
        dict with keys:
            frame_scores: List of per-frame MagFace scores (float)
            min: Minimum score
            max: Maximum score
            mean: Average score
            p10: 10th percentile
            p50: 50th percentile (median)
            p90: 90th percentile
            num_frames: Total frames scored
    """
    from dardcollect.face_geometry import arcface_from_ofiq_frame
    from dardcollect.magface import score_frame
    from dardcollect.pipeline_utils import _get_frames_from_crop

    global _provider_logged

    frames = _get_frames_from_crop(crop_path)
    if not frames:
        logger.warning("No frames from %s — skipping", crop_path.name)
        return {}

    frame_scores = []
    for ofiq_frame in frames:
        arcface_frame = arcface_from_ofiq_frame(ofiq_frame)
        score = score_frame(session, arcface_frame)

        # Log actual execution provider on first frame
        if not _provider_logged:
            _provider_logged = True
            providers = session.get_providers()
            if providers:
                logger.info("  Actual execution provider during inference: %s", providers[0])

        frame_scores.append(float(score))

    if not frame_scores:
        return {}

    scores_array = np.array(frame_scores)
    return {
        "frame_scores": frame_scores,
        "min": float(scores_array.min()),
        "max": float(scores_array.max()),
        "mean": float(scores_array.mean()),
        "p10": float(np.percentile(scores_array, 10)),
        "p50": float(np.percentile(scores_array, 50)),
        "p90": float(np.percentile(scores_array, 90)),
        "num_frames": len(frame_scores),
    }


def _passes_quality(
    crop_path: Path,
    session: ort.InferenceSession,
    threshold: float,
) -> tuple[bool, float]:
    """Score frames from a crop (video or image) and exit as soon as one meets the threshold.

    Reads OFIQ 616×616 frames/image, extracts a 112×112 ArcFace crop from each using
    the precomputed constant region, then scores with MagFace.

    The returned max_score is the highest score seen up to the passing frame —
    a lower bound on the crop's true peak quality, but sufficient for filtering
    and for relative comparison between crops.

    Args:
        crop_path: Path to the OFIQ face crop (.mp4 video or .jpg/.png image).
        session: Loaded MagFace ONNX session.
        threshold: Minimum quality score required to pass.

    Returns:
        tuple: (passes, max_score) where passes is True if any frame met the threshold,
            and max_score is the highest score seen.
    """
    magface_data = score_all_magface_frames(crop_path, session)
    if not magface_data:
        return False, 0.0

    max_score = magface_data["max"]
    return max_score >= threshold, max_score


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
