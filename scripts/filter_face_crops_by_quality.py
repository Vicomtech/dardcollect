#!/usr/bin/env python3
"""
Filter face crop videos by facial image quality using the OFIQ unified quality score.

The quality metric follows ISO/IEC 29794-5 (OFIQ): the unified quality score is
the output magnitude of MagFace (IResNet50), which measures how confidently a
face recognition model can embed a given crop — a standard proxy for biometric
sample quality defined in the OFIQ reference implementation
(https://github.com/BSI-OFIQ/OFIQ-Project).

Rather than running the full OFIQ pipeline (which includes internal SSD face
detection and ADNet landmark estimation), this script applies the MagFace model
directly to the already-aligned face crop videos produced by extract_face_crops.py.
The face crops are CIGPose-aligned (eyes horizontal, face centred) and normalised
before being passed to MagFace, so re-detecting the face is unnecessary.

Quality score: raw positive float output of magface_iresnet50_norm.onnx (higher =
better). Not sigmoid-calibrated to [0, 100] — set the threshold empirically by
inspecting score distributions on your data.

Preprocessing matches MagFace/ArcFace expectations:
  resize to 112×112, BGR→RGB, normalise to [-1, 1] (pixel/127.5 − 1).

All parameters are read from config.yaml under the 'face_quality_filtering' key.
"""

import logging
import shutil
import sys
from dataclasses import asdict
from pathlib import Path
from typing import cast

import numpy as np
from tqdm import tqdm


class _TqdmHandler(logging.StreamHandler):
    def emit(self, record: logging.LogRecord) -> None:
        tqdm.write(self.format(record))


_handler = _TqdmHandler()
_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
logging.basicConfig(handlers=[_handler], level=logging.DEBUG, force=True)
logger = logging.getLogger(__name__)

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.yaml"
_DEFAULT_MAGFACE_MODEL = (
    Path(__file__).resolve().parent.parent / "persondet" / "models" / "magface_iresnet50_norm.onnx"
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from persondet.gpu_setup import setup_gpu_paths

setup_gpu_paths(str(CONFIG_PATH))

import cv2
import onnxruntime as ort

import persondet
from persondet.config import FaceQualityFilterConfig
from persondet.provenance import PROVENANCE_FILENAME, now_iso, record_stage

_MAGFACE_INPUT_SIZE = 112  # IResNet50 input resolution


# ── MagFace helpers ───────────────────────────────────────────────────────────


def _load_magface(model_path: str, gpu_id: int) -> ort.InferenceSession:
    """Load the MagFace ONNX model onto the requested GPU (or CPU as fallback).

    :param model_path: Path to magface_iresnet50_norm.onnx, or "" for default.
    :param gpu_id: CUDA device ID.
    :return: ONNX Runtime inference session.
    """
    path = Path(model_path) if model_path else _DEFAULT_MAGFACE_MODEL
    if not path.exists():
        raise FileNotFoundError(
            f"MagFace model not found at {path}.\n"
            f"Expected at persondet/models/magface_iresnet50_norm.onnx.\n"
            f"Obtain it from the OFIQ project: https://github.com/BSI-OFIQ/OFIQ-Project"
        )
    providers = [("CUDAExecutionProvider", {"device_id": gpu_id}), "CPUExecutionProvider"]
    session = ort.InferenceSession(str(path), providers=providers)
    logger.info("Loaded MagFace from %s", path)
    return session


def _preprocess(frame_bgr) -> np.ndarray:
    """Resize and normalise a BGR face crop for MagFace.

    :param frame_bgr: BGR uint8 numpy array (any square size).
    :return: Float32 array of shape (1, 3, 112, 112), values in [-1, 1].
    """
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(
        rgb, (_MAGFACE_INPUT_SIZE, _MAGFACE_INPUT_SIZE), interpolation=cv2.INTER_LINEAR
    )
    normalized = (resized.astype(np.float32) - 127.5) / 128.0
    return normalized.transpose(2, 0, 1)[np.newaxis]  # HWC → NCHW


def _score_frame(session: ort.InferenceSession, frame_bgr) -> float:
    """Return the MagFace quality score for a single face crop frame.

    :param session: Loaded MagFace ONNX session.
    :param frame_bgr: BGR uint8 numpy array.
    :return: Quality score (higher = better). Returns 0.0 on failure.
    """
    try:
        inp = _preprocess(frame_bgr)
        outputs = cast(list[np.ndarray], session.run(None, {session.get_inputs()[0].name: inp}))
        return float(outputs[0][0])
    except Exception:
        return 0.0


# ── Disk-space guard ──────────────────────────────────────────────────────────


def _check_disk_space(path: Path, min_gb: float) -> None:
    usage = shutil.disk_usage(path)
    free_gb = usage.free / (1024**3)
    if free_gb < min_gb:
        raise RuntimeError(f"Only {free_gb:.1f} GB free on {path} (minimum {min_gb} GB required)")


# ── Per-video quality assessment ──────────────────────────────────────────────


def _passes_quality(
    video_path: Path,
    session: ort.InferenceSession,
    threshold: float,
    num_sample_frames: int,
) -> tuple[bool, float]:
    """Score a fixed number of uniformly sampled frames and check against threshold.

    Sampling a fixed count rather than every Nth frame keeps per-video cost
    constant regardless of clip length.

    :param video_path: Path to the face crop .mp4 file.
    :param session: Loaded MagFace ONNX session.
    :param threshold: Minimum quality score required to pass.
    :param num_sample_frames: How many frames to sample from the video.
    :return: (passes, max_score) tuple.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logger.warning("Cannot open %s — skipping", video_path.name)
        return False, 0.0

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        return False, 0.0

    n = min(num_sample_frames, total_frames)
    sample_indices = np.linspace(0, total_frames - 1, n, dtype=int)

    max_score = 0.0
    try:
        for idx in sample_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ret, frame = cap.read()
            if not ret:
                continue
            score = _score_frame(session, frame)
            if score > max_score:
                max_score = score
            if max_score >= threshold:
                return True, max_score
    finally:
        cap.release()

    return max_score >= threshold, max_score


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    cfg = FaceQualityFilterConfig.from_yaml(str(CONFIG_PATH))

    input_dir = Path(cfg.input_dir)
    output_dir = Path(cfg.output_dir)

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    _check_disk_space(output_dir, cfg.min_free_disk_gb)

    video_files = sorted(input_dir.glob("*_face_*.mp4"))
    if not video_files:
        logger.info("No face crop videos found in %s", input_dir)
        return

    logger.info(
        "Found %d face crop videos in %s — quality threshold %.2f, %d frames/video",
        len(video_files),
        input_dir,
        cfg.quality_threshold,
        cfg.num_sample_frames,
    )

    session = _load_magface(cfg.magface_model_path, cfg.gpu_id)

    started_at = now_iso()
    videos_assessed = 0
    videos_passed = 0
    videos_skipped = 0
    all_scores: list[float] = []

    for video_path in tqdm(video_files, desc="Quality filtering", unit="video"):
        sidecar_path = video_path.with_suffix(".json")
        dest_video = output_dir / video_path.name
        dest_sidecar = output_dir / sidecar_path.name

        # Idempotency: already moved
        if dest_video.exists():
            logger.debug("Already in output dir, skipping: %s", video_path.name)
            videos_skipped += 1
            continue

        if not sidecar_path.exists():
            logger.warning("Missing sidecar JSON for %s — skipping", video_path.name)
            continue

        _check_disk_space(output_dir, cfg.min_free_disk_gb)

        try:
            passes, max_score = _passes_quality(
                video_path, session, cfg.quality_threshold, cfg.num_sample_frames
            )
        except Exception as exc:
            logger.error("Error assessing %s: %s", video_path.name, exc)
            continue

        videos_assessed += 1
        all_scores.append(max_score)
        status = f"score={max_score:.4f}"

        if passes:
            shutil.move(str(video_path), dest_video)
            shutil.move(str(sidecar_path), dest_sidecar)
            videos_passed += 1
            logger.info("PASS %s (%s) → %s", video_path.name, status, output_dir)
        else:
            logger.debug("FAIL %s (%s)", video_path.name, status)

    logger.info(
        "Done. Assessed: %d  Passed: %d  Skipped (already done): %d",
        videos_assessed,
        videos_passed,
        videos_skipped,
    )

    if all_scores:
        arr = np.array(all_scores)
        logger.info(
            "Score distribution (n=%d): min=%.2f  p10=%.2f  p25=%.2f  median=%.2f"
            "  p75=%.2f  p90=%.2f  max=%.2f  (threshold=%.2f)",
            len(arr),
            arr.min(),
            np.percentile(arr, 10),
            np.percentile(arr, 25),
            np.percentile(arr, 50),
            np.percentile(arr, 75),
            np.percentile(arr, 90),
            arr.max(),
            cfg.quality_threshold,
        )

    record_stage(
        output_dir.parent / PROVENANCE_FILENAME,
        {
            "stage": "filter_face_crops_by_quality",
            "started_at": started_at,
            "completed_at": now_iso(),
            "software": {
                "script": "scripts/filter_face_crops_by_quality.py",
                "persondet_version": persondet.__version__,
            },
            "config": asdict(cfg),
            "stats": {
                "videos_assessed": videos_assessed,
                "videos_passed": videos_passed,
                "videos_skipped_already_done": videos_skipped,
                **(
                    {
                        "score_min": float(np.min(all_scores)),
                        "score_p10": float(np.percentile(all_scores, 10)),
                        "score_p25": float(np.percentile(all_scores, 25)),
                        "score_median": float(np.percentile(all_scores, 50)),
                        "score_p75": float(np.percentile(all_scores, 75)),
                        "score_p90": float(np.percentile(all_scores, 90)),
                        "score_max": float(np.max(all_scores)),
                    }
                    if all_scores
                    else {}
                ),
            },
        },
    )


if __name__ == "__main__":
    main()
