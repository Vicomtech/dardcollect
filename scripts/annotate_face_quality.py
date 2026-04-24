#!/usr/bin/env python3
"""
Annotate face crop videos with OFIQ-based face quality scores.

Reads face crop videos from an output folder of extract_face_crops.py or
filter_face_crops_by_quality.py, computes the following quality measures
following ISO/IEC 29794-5 (OFIQ), and writes a sibling .quality.json file
next to each video (leaves the extraction-stage sidecar .json untouched):

  unified_score           MagFace IResNet50 magnitude (OFIQ UnifiedQualityScore)
  sharpness               Laplacian/Sobel RTrees (OFIQ Sharpness)
  compression_artifacts   SSIM CNN (OFIQ CompressionArtifacts)
  expression_neutrality   HSEmotion EfficientNet-B0/B2 + AdaBoost (OFIQ ExpressionNeutrality)
  no_head_coverings       BiSeNet face parsing — hat/cloth pixel fraction (OFIQ NoHeadCoverings)
  face_occlusion          FaceOcclusionSegmentation CNN (OFIQ FaceOcclusionPrevention)
  head_pose               MobileNetV1 3DDFAV2 — yaw/pitch/roll angles + cosine² quality scores

Each measure is summarised per video as {max, mean, p10, p50, p90}.
Head-pose additionally stores the raw angles (degrees, signed) and their quality scores.

The .quality.json carries provenance fields (face_crop_video, face_crop_json,
source_video, annotated_at, annotator) so the origin chain from quality data →
face crop → source video is always traceable.

NOTE: The models were designed for OFIQ's internal 616×616 aligned-face format.
This script feeds them the ArcFace-aligned face crops produced by extract_face_crops.py
(224×224 or 112×112).  Preprocessing is adapted accordingly; scores are interpretable
and comparable across clips but will differ slightly from full-OFIQ values.
"""

import argparse
import gzip
import json
import logging
import os
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import cv2
import numpy as np
import onnxruntime as ort
from tqdm import tqdm

# ── Logging ───────────────────────────────────────────────────────────────────


class _TqdmHandler(logging.StreamHandler):
    def emit(self, record: logging.LogRecord) -> None:
        tqdm.write(self.format(record))


_handler = _TqdmHandler()
_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
logging.basicConfig(handlers=[_handler], level=logging.DEBUG, force=True)
logger = logging.getLogger(__name__)

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.yaml"
DEFAULT_MODELS_DIR = Path(__file__).resolve().parent.parent / "persondet" / "models"

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from persondet.gpu_setup import setup_gpu_paths
from persondet.magface import load_magface
from persondet.magface import score_frame as magface_score_frame
from persondet.onnx_utils import get_preferred_providers
from persondet.provenance import now_iso

setup_gpu_paths(str(CONFIG_PATH))


# ── OFIQ general sigmoid calibration: Q(x) = h * (a + s * sigmoid(x, x0, w)) ─


def _ofiq_sigmoid(x: float, h: float, a: float, s: float, x0: float, w: float) -> float:
    sig = 1.0 / (1.0 + np.exp((x0 - x) / w))
    score = h * (a + s * sig)
    return float(np.clip(score, 0.0, 100.0))


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
    providers = get_preferred_providers(gpu_id)
    logger.info("Loading quality models from %s (GPU %d)", models_dir, gpu_id)

    def _onnx(name: str) -> ort.InferenceSession:
        p = models_dir / name
        if not p.exists():
            raise FileNotFoundError(f"Model not found: {p}")
        sess = ort.InferenceSession(str(p), providers=providers)
        logger.debug("  Loaded %s", name)
        return sess

    def _load_gz_opencv(name: str, loader):
        p = models_dir / name
        if not p.exists():
            raise FileNotFoundError(f"Model not found: {p}")
        with gzip.open(p, "rb") as f:
            xml_bytes = f.read()
        with tempfile.NamedTemporaryFile(suffix=".xml", delete=False) as tf:
            tf.write(xml_bytes)
            tpath = tf.name
        try:
            model = loader(tpath)
        finally:
            os.unlink(tpath)
        logger.debug("  Loaded %s", name)
        return model

    magface = load_magface("", gpu_id)
    rtrees = _load_gz_opencv("face_sharpness_rtree.xml.gz", cv2.ml.RTrees_load)  # type: ignore
    n_trees = rtrees.getTermCriteria()[1]

    return QualityModels(
        magface=magface,
        rtrees=rtrees,
        n_trees=n_trees,
        compression=_onnx("ssim_248_model.onnx"),
        enet_b0=_onnx("enet_b0_8_best_vgaf_embed_zeroed.onnx"),
        enet_b2=_onnx("enet_b2_8_embed_zeroed.onnx"),
        adaboost=_load_gz_opencv("hse_1_2_C_adaboost.yml.gz", cv2.ml.Boost_load),  # type: ignore
        bisenet=_onnx("bisenet_400.onnx"),
        occlusion=_onnx("face_occlusion_segmentation_ort.onnx"),
        headpose=_onnx("mb1_120x120.onnx"),
    )


# ── Per-frame inference ───────────────────────────────────────────────────────

# Head-pose denormalisation constants from OFIQ HeadPose3DDFAV2.cpp
_HP_MEAN = np.array(
    [
        3.4926363e-04,
        2.5279013e-07,
        -6.8751979e-07,
        6.0167957e01,
        -6.2955132e-07,
        5.7572004e-04,
        -5.0853912e-05,
    ],
    dtype=np.float32,
)
_HP_STD = np.array(
    [
        1.76321526e-04,
        6.73794348e-05,
        4.47084894e-04,
        2.65502319e01,
        1.23137695e-04,
        4.49302170e-05,
        7.92367064e-05,
    ],
    dtype=np.float32,
)

# BiSeNet class indices for head-covering pixels
_BISENET_CLOTH_IDX = 16
_BISENET_HAT_IDX = 18

# ImageNet mean/std (BGR order for OpenCV)
_IMAGENET_MEAN_BGR = np.array([0.406, 0.456, 0.485], dtype=np.float32) * 255.0
_IMAGENET_STD_BGR = np.array([0.225, 0.224, 0.229], dtype=np.float32) * 255.0


def _sharpness_score(frame_bgr: np.ndarray, rtrees: cv2.ml.RTrees, n_trees: int) -> float:
    """Compute OFIQ Sharpness raw score from feature vector, then calibrate."""
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    mask = np.ones_like(gray, dtype=np.uint8)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)

    features: list[float] = []

    # Laplacian features (kernel sizes 1, 3, 5, 7, 9)
    for k in (1, 3, 5, 7, 9):
        lap = cv2.Laplacian(blurred, cv2.CV_64F, ksize=k)
        m, s = cv2.meanStdDev(np.abs(lap), mask=mask)
        features.extend([float(m[0, 0]), float(s[0, 0])])

    # MeanDiff features (kernel sizes 3, 5, 7)
    for k in (3, 5, 7):
        b = cv2.blur(gray, (k, k))
        diff = cv2.absdiff(gray.astype(np.float64), b.astype(np.float64))
        m, s = cv2.meanStdDev(diff, mask=mask)
        features.extend([float(m[0, 0]), float(s[0, 0])])

    # Sobel features (kernel sizes 1, 3, 5, 7, 9)
    for k in (1, 3, 5, 7, 9):
        sob = cv2.Sobel(gray.astype(np.float64), cv2.CV_64F, 1, 1, ksize=k)
        m, s = cv2.meanStdDev(np.abs(sob), mask=mask)
        features.extend([float(m[0, 0]), float(s[0, 0])])

    feat = np.array(features, dtype=np.float32).reshape(1, -1)
    pred_result = np.zeros((1, 1), dtype=np.float32)
    rtrees.predict(feat, pred_result, flags=cv2.ml.StatModel_RAW_OUTPUT)
    raw = float(n_trees) - float(pred_result[0, 0])
    # OFIQ sigmoid: h=1, a=-14.0, s=115.0, x0=-20.0, w=15.0
    return _ofiq_sigmoid(raw, h=1.0, a=-14.0, s=115.0, x0=-20.0, w=15.0)


def _compression_score(frame_bgr: np.ndarray, session: ort.InferenceSession) -> float:
    """Compute OFIQ CompressionArtifacts score."""
    img = cv2.resize(frame_bgr, (248, 248), interpolation=cv2.INTER_LINEAR)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
    img_rgb -= np.array([123.7, 116.3, 103.5], dtype=np.float32)
    img_rgb /= np.array([58.4, 57.1, 57.4], dtype=np.float32)
    blob = cv2.dnn.blobFromImage(img_rgb, scalefactor=1.0, size=(248, 248), mean=0, swapRB=False)
    output = cast(list[np.ndarray], session.run(None, {session.get_inputs()[0].name: blob}))
    raw = float(output[0][0, 0])
    # OFIQ sigmoid: h=1, a=-0.0278, s=103.0, x0=0.3308, w=0.092
    return _ofiq_sigmoid(raw, h=1.0, a=-0.0278, s=103.0, x0=0.3308, w=0.092)


def _expression_neutrality_score(
    frame_bgr: np.ndarray,
    enet_b0: ort.InferenceSession,
    enet_b2: ort.InferenceSession,
    adaboost: cv2.ml.Boost,
) -> float:
    """Compute OFIQ ExpressionNeutrality score."""
    img_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    img_norm = (img_rgb - np.array([0.485, 0.456, 0.406], dtype=np.float32)) / np.array(
        [0.229, 0.224, 0.225], dtype=np.float32
    )

    # B0: 224×224
    img_b0 = cv2.resize(img_norm, (224, 224), interpolation=cv2.INTER_LINEAR)
    blob_b0 = cv2.dnn.blobFromImage(img_b0, scalefactor=1.0, size=(224, 224), mean=0, swapRB=False)
    feat1 = cast(
        np.ndarray, enet_b0.run(None, {enet_b0.get_inputs()[0].name: blob_b0})[0]
    )  # (1, 1280)

    # B2: 260×260
    img_b2 = cv2.resize(img_norm, (260, 260), interpolation=cv2.INTER_LINEAR)
    blob_b2 = cv2.dnn.blobFromImage(img_b2, scalefactor=1.0, size=(260, 260), mean=0, swapRB=False)
    feat2 = cast(
        np.ndarray, enet_b2.run(None, {enet_b2.get_inputs()[0].name: blob_b2})[0]
    )  # (1, 1408)

    features = np.concatenate([feat1, feat2], axis=1).astype(np.float32)  # (1, 2688)
    pred_result = np.zeros((1, 1), dtype=np.float32)
    adaboost.predict(features, pred_result, flags=cv2.ml.DTrees_PREDICT_SUM)
    raw = float(pred_result[0, 0])
    # OFIQ sigmoid: h=100, x0=-5000.0, w=5000.0, a=0 (default), s=1 (default)
    return _ofiq_sigmoid(raw, h=100.0, a=0.0, s=1.0, x0=-5000.0, w=5000.0)


def _no_head_coverings_score(frame_bgr: np.ndarray, session: ort.InferenceSession) -> float:
    """Compute OFIQ NoHeadCoverings score via BiSeNet face parsing."""
    img = cv2.resize(frame_bgr, (400, 400), interpolation=cv2.INTER_LINEAR)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
    img_norm = (img_rgb - _IMAGENET_MEAN_BGR[[2, 1, 0]]) / _IMAGENET_STD_BGR[[2, 1, 0]]
    blob = cv2.dnn.blobFromImage(img_norm, scalefactor=1.0, size=(400, 400), mean=0, swapRB=False)
    logits = cast(
        np.ndarray, session.run(None, {session.get_inputs()[0].name: blob})[0]
    )  # (1, 19, 400, 400)

    seg_map = np.argmax(logits[0], axis=0).astype(np.uint8)  # (400, 400)

    # OFIQ crops the bottom 204 rows before counting; apply proportional crop (51%)
    crop_bottom = round(204 / 400 * seg_map.shape[0])
    seg_cropped = seg_map[: seg_map.shape[0] - crop_bottom, :]

    total = seg_cropped.size
    cloth_px = int(np.sum(seg_cropped == _BISENET_CLOTH_IDX))
    hat_px = int(np.sum(seg_cropped == _BISENET_HAT_IDX))
    raw = (cloth_px + hat_px) / total if total > 0 else 0.0

    # Piecewise quality mapping from OFIQ (T0=0.0, T1=0.95, x0=0.02, w=0.1)
    T0, T1, x0, w = 0.0, 0.95, 0.02, 0.1
    if raw <= T0:
        return 100.0
    if raw >= T1:
        return 0.0
    s = 1.0 / (1.0 + np.exp((x0 - raw) / w))
    s0 = 1.0 / (1.0 + np.exp((x0 - T0) / w))
    s1 = 1.0 / (1.0 + np.exp((x0 - T1) / w))
    q = (s1 - s) / (s1 - s0)
    return float(np.clip(round(100.0 * q), 0.0, 100.0))


def _face_occlusion_score(frame_bgr: np.ndarray, session: ort.InferenceSession) -> float:
    """Compute OFIQ FaceOcclusionPrevention score.

    Returns 100 when no occlusion is detected, 0 for fully occluded.
    """
    img = cv2.resize(frame_bgr, (224, 224), interpolation=cv2.INTER_LINEAR)
    # swapRB=True converts BGR→RGB; normalize [0, 1]
    blob = cv2.dnn.blobFromImage(img, scalefactor=1.0 / 255.0, size=(224, 224), mean=0, swapRB=True)
    out = cast(list[np.ndarray], session.run(None, {session.get_inputs()[0].name: blob}))
    # OFIQ: negate output, then THRESH_BINARY_INV(thresh=0) gives 1 when original ≥ 0.
    # So raw_output ≥ 0 → non-occluded face; raw_output < 0 → occluded / background.
    raw_map = out[-1][0, 0]  # (H, W)
    non_occluded_frac = float(np.sum(raw_map >= 0)) / raw_map.size
    return float(np.clip(round(100.0 * non_occluded_frac), 0.0, 100.0))


def _head_pose_angles(
    frame_bgr: np.ndarray, session: ort.InferenceSession
) -> tuple[float, float, float]:
    """Return (yaw, pitch, roll) Euler angles in degrees using 3DDFAV2.

    Preprocessing and decoding from OFIQ HeadPose3DDFAV2.cpp.
    """
    img = cv2.resize(frame_bgr, (120, 120), interpolation=cv2.INTER_LINEAR).astype(np.float32)
    img = (img - 127.5) / 128.0
    # HWC → NCHW (BGR channel order, matching C++ code)
    blob = cv2.dnn.blobFromImage(img, scalefactor=1.0, size=(120, 120), mean=0, swapRB=False)
    raw = cast(np.ndarray, session.run(None, {session.get_inputs()[0].name: blob})[0])  # (1, 62)

    # Denormalise first 7 parameters
    params = raw[0, :7] * _HP_STD + _HP_MEAN

    # Build rotation matrix from two row vectors
    r0 = params[:3]
    r1 = params[4:]
    norm0 = np.linalg.norm(r0)
    norm1 = np.linalg.norm(r1)
    if norm0 < 1e-8 or norm1 < 1e-8:
        return 0.0, 0.0, 0.0
    r0 = r0 / norm0
    r1 = r1 / norm1
    r2 = np.cross(r0, r1)
    R = np.stack([r0, r1, r2]).T  # 3×3, transposed as in C++ code

    # Euler angles (ZYX convention as in OFIQ)
    THRES = 0.9975
    r31 = float(R[2, 0])
    if -THRES <= r31 <= THRES:
        pitch = np.arcsin(r31)
        c = np.cos(pitch)
        yaw = -np.arctan2(R[2, 1] / c, R[2, 2] / c)
        roll = -np.arctan2(R[1, 0] / c, R[0, 0] / c)
    elif r31 < -THRES:
        pitch = -np.pi / 2
        yaw = -np.arctan2(R[0, 1], R[0, 2])
        roll = 0.0
    else:
        pitch = np.pi / 2
        yaw = np.arctan2(R[0, 1], R[0, 2])
        roll = 0.0

    return (
        float(yaw * 180.0 / np.pi),
        float(pitch * 180.0 / np.pi),
        float(roll * 180.0 / np.pi),
    )


def _angle_to_quality(angle_deg: float) -> float:
    """OFIQ HeadPose quality: round(100 * max(0, cos(angle))²)."""
    c = np.cos(angle_deg * np.pi / 180.0)
    c = max(0.0, float(c))
    return float(round(100.0 * c * c))


# ── Per-frame scoring ─────────────────────────────────────────────────────────


def score_frame_all(frame_bgr: np.ndarray, models: QualityModels) -> dict:
    """Run all quality measures on a single frame and return a dict of scores."""
    yaw, pitch, roll = _head_pose_angles(frame_bgr, models.headpose)
    return {
        "unified_score": magface_score_frame(models.magface, frame_bgr),
        "sharpness": _sharpness_score(frame_bgr, models.rtrees, models.n_trees),
        "compression_artifacts": _compression_score(frame_bgr, models.compression),
        "expression_neutrality": _expression_neutrality_score(
            frame_bgr, models.enet_b0, models.enet_b2, models.adaboost
        ),
        "no_head_coverings": _no_head_coverings_score(frame_bgr, models.bisenet),
        "face_occlusion_prevention": _face_occlusion_score(frame_bgr, models.occlusion),
        "head_pose": {
            "yaw_deg": yaw,
            "pitch_deg": pitch,
            "roll_deg": roll,
            "yaw_quality": _angle_to_quality(yaw),
            "pitch_quality": _angle_to_quality(pitch),
            "roll_quality": _angle_to_quality(roll),
        },
    }


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
        vals = [s[key] for s in frame_scores]
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


# ── Per-video processing ──────────────────────────────────────────────────────


def _score_and_append(
    frame: np.ndarray,
    frame_idx: int,
    video_name: str,
    models: QualityModels,
    out: list,
) -> None:
    try:
        out.append(score_frame_all(frame, models))
    except Exception as exc:
        logger.debug("Error scoring frame %d of %s: %s", frame_idx, video_name, exc)


def score_video(
    video_path: Path,
    models: QualityModels,
    frame_stride: int,
    max_frames: int,
    overwrite: bool,
) -> bool:
    """Score a single face crop video and write a sibling .quality.json file.

    Returns True if the quality file was written, False if skipped.
    """
    quality_path = video_path.with_suffix(".quality.json")
    if not overwrite and quality_path.exists():
        logger.debug("Already annotated, skipping: %s", video_path.name)
        return False

    sidecar_path = video_path.with_suffix(".json")
    source_video = ""
    if sidecar_path.exists():
        try:
            with open(sidecar_path, encoding="utf-8") as f:
                source_video = json.load(f).get("source_video", "")
        except Exception:
            pass
    else:
        logger.warning(
            "No sidecar JSON alongside %s — provenance will be incomplete", video_path.name
        )

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logger.warning("Cannot open video: %s", video_path.name)
        return False

    frame_scores: list[dict] = []
    frame_idx = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % frame_stride == 0:
                _score_and_append(frame, frame_idx, video_path.name, models, frame_scores)
            frame_idx += 1
            if max_frames > 0 and len(frame_scores) >= max_frames:
                break
    finally:
        cap.release()

    if not frame_scores:
        logger.warning("No frames scored for %s", video_path.name)
        return False

    quality_data: dict = {
        "face_crop_video": video_path.name,
        "face_crop_json": sidecar_path.name,
        "source_video": source_video,
        "annotated_at": now_iso(),
        "annotator": "scripts/annotate_face_quality.py",
        "frame_stride": frame_stride,
        "max_frames_sampled": max_frames,
        **aggregate_frame_scores(frame_scores),
    }

    with open(quality_path, "w", encoding="utf-8") as f:
        json.dump(quality_data, f, indent=2)

    logger.info(
        "  %s  → %d frames scored, unified_score max=%.1f → %s",
        video_path.name,
        len(frame_scores),
        quality_data["unified_score"]["max"],
        quality_path.name,
    )
    return True


# ── Entry point ───────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Annotate face crop sidecar JSONs with OFIQ face quality scores.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "input_dir",
        type=Path,
        help="Folder containing face crop .mp4 files and their sidecar .json files "
        "(output of extract_face_crops.py or filter_face_crops_by_quality.py).",
    )
    parser.add_argument(
        "--gpu-id",
        type=int,
        default=0,
        metavar="ID",
        help="CUDA device ID for ONNX Runtime (use -1 to force CPU).",
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=DEFAULT_MODELS_DIR,
        metavar="DIR",
        help="Directory containing all OFIQ ONNX and OpenCV model files.",
    )
    parser.add_argument(
        "--frame-stride",
        type=int,
        default=5,
        metavar="N",
        help="Score every N-th frame (1 = all frames).",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=30,
        metavar="N",
        help="Maximum frames to score per video (0 = no limit).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-annotate videos that already have a sibling .quality.json file.",
    )
    args = parser.parse_args()

    if not args.input_dir.exists():
        logger.error("Input directory does not exist: %s", args.input_dir)
        sys.exit(1)

    video_files = sorted(args.input_dir.glob("*_face_*.mp4"))
    if not video_files:
        logger.error("No face crop videos (*_face_*.mp4) found in %s", args.input_dir)
        sys.exit(1)

    logger.info("Found %d face crop video(s) in %s", len(video_files), args.input_dir)

    try:
        models = load_models(args.models_dir, args.gpu_id)
    except Exception as exc:
        logger.error("Failed to load models: %s", exc)
        sys.exit(1)

    updated = 0
    skipped = 0

    for video_path in tqdm(video_files, desc="Annotating quality", unit="video"):
        try:
            result = score_video(
                video_path,
                models,
                frame_stride=args.frame_stride,
                max_frames=args.max_frames,
                overwrite=args.overwrite,
            )
            if result:
                updated += 1
            else:
                skipped += 1
        except Exception as exc:
            logger.error("Error processing %s: %s", video_path.name, exc)

    logger.info(
        "Done.  Written: %d  Skipped (already annotated or error): %d",
        updated,
        skipped,
    )


if __name__ == "__main__":
    main()
