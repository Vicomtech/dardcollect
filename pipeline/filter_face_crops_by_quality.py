#!/usr/bin/env python3
"""
Filter face crop videos and images by MagFace quality and compute all scores.

Computes MagFace (IResNet50) quality scores for all frames and saves per-frame and
aggregated scores to a .magface.json sidecar alongside each crop. Crops passing
the quality threshold are moved to output_dir/. This script is responsible for MagFace
scoring only; other OFIQ measures (sharpness, expression, etc.) are computed by
annotate_face_quality.py.

MagFace measures how confidently a face recognition model can embed a crop —
a standard proxy for biometric sample quality defined in the OFIQ reference
implementation (https://github.com/BSI-OFIQ/OFIQ-Project).

Input directory (produced by extract_face_crops_from_videos.py or
extract_face_crops_from_images.py):
  input_dir/  — 616×616 OFIQ-aligned crops (flat layout, no subdirs)

Supported crop formats:
  - Videos: .mp4 files (from extract_face_crops_from_videos.py) with per-frame quality assessment
  - Images: .jpg/.png files (from extract_face_crops_from_images.py) with single-frame assessment

Output files:
  (For crops that PASS the threshold)
  output_dir/
    crop.mp4/.jpg/.png          — Original crop
    crop.json                    — Original sidecar (keypoints, bbox, etc.)
    crop.magface.json            — MagFace scores (per-frame and aggregated stats)

  (For crops that FAIL the threshold)
  input_dir/
    crop.mp4/.jpg/.png          — Original crop (stays in place)
    crop.json                    — Original sidecar (stays in place)
    crop.magface.json            — MagFace scores (always computed and saved)

All crops, whether they pass or fail the quality threshold, get a .magface.json
file with complete per-frame and aggregated MagFace scores. This enables
post-filtering decisions and comparative analysis in annotate_face_quality.py.

The sidecar JSON alongside each crop contains: keypoints, bbox, score, and
face_crop_corners_arcface. MagFace scoring extracts 112×112 ArcFace crops from
OFIQ frames on-the-fly using the constant region defined in dardcollect/face_geometry.py.

All parameters are read from config.yaml under the 'face_quality_filtering' key.
"""

import argparse
import json
import logging
import os
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import onnxruntime as ort
from tqdm import tqdm

from dardcollect.pipeline_utils import _check_disk_space, _TqdmHandler
from dardcollect.quality import score_all_magface_frames

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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dardcollect.gpu_setup import setup_gpu_paths

setup_gpu_paths(str(CONFIG_PATH))


from dardcollect.config import FaceQualityFilterConfig, get_log_level
from dardcollect.magface import load_magface
from dardcollect.pipeline_loggers import FilteredFaceCropsLogger
from dardcollect.provenance import now_iso

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


def _refresh_image_parent_uuid(sidecar_path: Path, input_dir: Path) -> None:
    """Refresh parent UUID in an image crop sidecar from current detections.

    Image detections may be regenerated with new UUIDs across reruns. This keeps
    filtered image crop provenance linked to the active detection sidecar.
    """
    if not sidecar_path.exists():
        return

    try:
        with open(sidecar_path, encoding="utf-8") as f:
            sidecar = json.load(f)
    except Exception as exc:
        logger.warning("Failed to read sidecar %s: %s", sidecar_path.name, exc)
        return

    parent = sidecar.get("parent_clip")
    if not isinstance(parent, dict):
        return

    parent_file = parent.get("file")
    if not isinstance(parent_file, str) or not parent_file:
        return

    detection_sidecar = input_dir.parent / "extracted_image_detections" / parent_file
    if not detection_sidecar.exists():
        return

    try:
        with open(detection_sidecar, encoding="utf-8") as f:
            detection_data = json.load(f)
    except Exception as exc:
        logger.warning("Failed to read detection sidecar %s: %s", detection_sidecar.name, exc)
        return

    detection_uuid = detection_data.get("uuid")
    if not isinstance(detection_uuid, str) or not detection_uuid:
        return

    if parent.get("uuid") == detection_uuid:
        return

    parent["uuid"] = detection_uuid
    sidecar["parent_clip"] = parent
    _write_atomically(sidecar, sidecar_path)


# ── Main ──────────────────────────────────────────────────────────────────────


def _get_max_score(
    crop_path: Path,
    sidecar_path: Path,
    magface_path: Path,
    output_dir: Path,
    cfg: FaceQualityFilterConfig,
    session: ort.InferenceSession,
) -> float | None:
    """Return the max MagFace score for a crop.

    Reuses the existing ``.magface.json`` when it has a valid ``unified_score.max``;
    otherwise computes fresh scores (saving them atomically) and returns the max.
    Returns ``None`` on a scoring or write failure so the caller can skip the crop.
    """
    # Reuse existing .magface.json when possible, but still evaluate threshold.
    if magface_path.exists():
        try:
            with open(magface_path, encoding="utf-8") as f:
                existing_magface = json.load(f)
            unified = existing_magface.get("unified_score", {})
            if isinstance(unified, dict) and "max" in unified:
                logger.debug("Reusing existing MagFace for %s", crop_path.name)
                return float(unified["max"])
            logger.warning(
                "%s exists but has no unified_score.max, will recompute", magface_path.name
            )
        except Exception as exc:
            logger.warning(
                "%s exists but is corrupted (%s), will recompute", magface_path.name, exc
            )

    _check_disk_space(output_dir, cfg.min_free_disk_gb)
    try:
        magface_data = score_all_magface_frames(crop_path, session)
    except Exception as exc:
        logger.error("Error assessing %s: %s", crop_path.name, exc)
        return None

    if not magface_data:
        logger.warning("Failed to score frames for %s", crop_path.name)
        return None

    max_score = float(magface_data["max"])

    # Read sidecar for provenance
    source_video = ""
    try:
        with open(sidecar_path, encoding="utf-8") as f:
            sidecar = json.load(f)
        source_video = sidecar.get("source_video", "")
    except Exception:
        pass

    # Save MagFace scores to .magface.json (unified_score: per-frame + aggregated stats)
    magface_json = {
        "face_crop_video": crop_path.name,
        "source_video": source_video,
        "annotated_at": now_iso(),
        "annotator": "pipeline/filter_face_crops_by_quality.py",
        "unified_score": magface_data,
    }
    if not _write_atomically(magface_json, magface_path):
        logger.error("Failed to save .magface.json for %s, skipping", crop_path.name)
        return None

    return max_score


def _process_crop(
    crop_path: Path,
    modality: str,
    input_dir: Path,
    output_dir: Path,
    cfg: FaceQualityFilterConfig,
    session: ort.InferenceSession,
    filter_logger: FilteredFaceCropsLogger,
) -> tuple[str, float | None]:
    """Score one crop with MagFace and move it to output_dir if it passes the threshold.

    Returns ``(status, score)`` where status is one of:
    ``"skipped"`` (already moved), ``"skipped_nosidecar"`` (missing sidecar),
    ``"error"`` (scoring/write failure), ``"assessed_pass"`` or ``"assessed_fail"``.
    ``score`` is the max MagFace score when assessed, else ``None``.
    """
    sidecar_path = crop_path.with_suffix(".json")
    magface_path = crop_path.with_suffix(".magface.json")
    # Preserve the source subdirectory in the destination so the layout
    # mirrors what extract_face_crops_from_videos.py wrote. Crops in
    # video_face_crops/0c9460bf-.../foo_face_1.mp4 land in
    # filtered_video_face_crops/0c9460bf-.../foo_face_1.mp4.
    try:
        rel_parent = crop_path.relative_to(input_dir).parent
    except ValueError:
        rel_parent = Path()
    dest_crop = output_dir / rel_parent / crop_path.name
    dest_sidecar = output_dir / rel_parent / sidecar_path.name
    dest_magface = output_dir / rel_parent / magface_path.name
    dest_crop.parent.mkdir(parents=True, exist_ok=True)

    # Idempotency: already moved
    if dest_crop.exists():
        if modality == "image" and dest_sidecar.exists():
            _refresh_image_parent_uuid(dest_sidecar, input_dir)
        logger.debug("Already in output dir, skipping: %s", crop_path.name)
        return "skipped", None

    if not sidecar_path.exists():
        logger.info("Missing sidecar JSON for %s — skipping", crop_path.name)
        return "skipped_nosidecar", None

    if modality == "image":
        _refresh_image_parent_uuid(sidecar_path, input_dir)

    max_score = _get_max_score(crop_path, sidecar_path, magface_path, output_dir, cfg, session)
    if max_score is None:
        return "error", None

    # Check if it passes the quality threshold
    if max_score >= cfg.quality_threshold:
        shutil.move(str(crop_path), dest_crop)
        shutil.move(str(sidecar_path), dest_sidecar)
        shutil.move(str(magface_path), dest_magface)
        filter_logger.log_filtered_crop(
            source_crop_path=str(crop_path),
            magface_score=float(max_score),
            filter_threshold=float(cfg.quality_threshold),
            output_path=str(dest_crop),
        )
        logger.info("PASS %s (score=%.4f) → %s", crop_path.name, max_score, output_dir)
        return "assessed_pass", max_score

    logger.debug(
        "FAIL %s (score=%.4f) — .magface.json retained for annotation",
        crop_path.name,
        max_score,
    )
    return "assessed_fail", max_score


def _log_score_distribution(modality: str, scores: list[float]) -> None:
    """Log a calibrated [0, 100] quality-score distribution summary."""
    scores_array = np.array(scores)
    logger.info(
        "[%s] Quality score distribution (calibrated [0, 100]):\n"
        "  Min: %.2f  |  Max: %.2f  |  Mean: %.2f  |  Median: %.2f\n"
        "  P10: %.2f  |  P25: %.2f  |  P50: %.2f  |  P75: %.2f  |  P90: %.2f",
        modality,
        scores_array.min(),
        scores_array.max(),
        scores_array.mean(),
        np.median(scores_array),
        np.percentile(scores_array, 10),
        np.percentile(scores_array, 25),
        np.percentile(scores_array, 50),
        np.percentile(scores_array, 75),
        np.percentile(scores_array, 90),
    )


def _process_modality(
    modality: str,
    cfg: FaceQualityFilterConfig,
    input_dir: Path,
    session: ort.InferenceSession,
) -> tuple[int, int, int, list[float]]:
    """Score and filter all face crops for one modality.

    Returns ``(assessed, passed, skipped, scores)``.
    """
    output_dir = Path(cfg.output_dir)

    if not input_dir.exists():
        logger.warning("Input directory does not exist: %s — skipping", input_dir)
        return 0, 0, 0, []

    output_dir.mkdir(parents=True, exist_ok=True)
    _check_disk_space(output_dir, cfg.min_free_disk_gb)

    # Find crops (dedup + drop existing masks)
    crop_files = sorted(
        {
            *sorted(input_dir.rglob("*_face_*.mp4")),
            *sorted(input_dir.rglob("*_face_*.jpg")),
            *sorted(input_dir.rglob("*_face_*.png")),
        }
    )
    crop_files = [p for p in crop_files if not p.name.endswith("_mask.png")]

    if not crop_files:
        logger.info("No face crops found in %s", input_dir)
        return 0, 0, 0, []

    logger.info(
        "Processing %s: %d OFIQ face crops — quality threshold %.2f",
        modality,
        len(crop_files),
        cfg.quality_threshold,
    )

    # Filtered crops logger — use whichever extraction CSV exists
    face_crops_csv = input_dir / "image_face_crops_extraction.csv"
    if not face_crops_csv.exists():
        face_crops_csv = input_dir / "video_face_crops_extraction.csv"
    filter_logger = FilteredFaceCropsLogger(
        output_dir=str(output_dir), face_crops_csv_path=face_crops_csv
    )

    assessed = 0
    passed = 0
    skipped = 0
    all_scores: list[float] = []

    for crop_path in tqdm(crop_files, desc=f"Quality filtering ({modality})", unit="crop"):
        status, score = _process_crop(
            crop_path, modality, input_dir, output_dir, cfg, session, filter_logger
        )
        if status == "skipped":
            skipped += 1
        elif status in ("assessed_pass", "assessed_fail"):
            assessed += 1
            all_scores.append(score or 0.0)
            if status == "assessed_pass":
                passed += 1

    logger.info(
        "[%s] Done. Assessed: %d  Passed: %d  Skipped (already done): %d",
        modality,
        assessed,
        passed,
        skipped,
    )
    if all_scores:
        _log_score_distribution(modality, all_scores)
    filter_logger.print_summary()
    return assessed, passed, skipped, all_scores


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

    # Load configs for whichever modalities are present. Either may be absent
    # in a single-modality config (media_types = video-only or image-only): the
    # missing modality is skipped with an info log. A missing *section* is the
    # single-modality case; any other ValueError (e.g. a missing required key in
    # a present section) is a real config error and re-raised.
    configs: list[tuple[str, FaceQualityFilterConfig, Path]] = []
    gpu_id: int | None = None

    try:
        cfg_video = FaceQualityFilterConfig.from_yaml(config_path, section="face_quality_filtering")
        input_dir_video = Path(cfg_video.input_dir)
        if input_dir_video.exists():
            configs.append(("video", cfg_video, input_dir_video))
        gpu_id = cfg_video.gpu_id
    except ValueError as exc:
        if "Missing 'face_quality_filtering' section in config" in str(exc):
            logger.info(
                "face_quality_filtering section not found in %s — "
                "skipping video modality (image-only config)",
                config_path,
            )
        else:
            raise

    try:
        cfg_image = FaceQualityFilterConfig.from_yaml(
            config_path, section="image_face_quality_filtering"
        )
        input_dir_image = Path(cfg_image.input_dir)
        if input_dir_image.exists():
            configs.append(("image", cfg_image, input_dir_image))
        if gpu_id is None:
            gpu_id = cfg_image.gpu_id
    except ValueError as exc:
        if "Missing 'image_face_quality_filtering' section in config" in str(exc):
            logger.info(
                "image_face_quality_filtering section not found in %s — "
                "skipping image modality (video-only config)",
                config_path,
            )
        else:
            raise

    if not configs:
        raise FileNotFoundError(
            "No input directories found in config. "
            "Run extract_face_crops_from_videos.py or extract_face_crops_from_images.py first."
        )

    logging.getLogger().setLevel(get_log_level(config_path))

    # Load MagFace session once (shared across modalities) — loaded ONCE.
    assert gpu_id is not None  # configs non-empty ⇒ at least one modality set gpu_id
    session = load_magface(gpu_id)

    # The orchestrator defers this stage's launch until its deps finish
    # (DEFER_UNTIL_DEPS_DONE), so this one-shot pass filters all available crops in
    # a single run — one MagFace load, no re-launch waste.
    for modality, cfg, input_dir in configs:
        _process_modality(modality, cfg, input_dir, session)


if __name__ == "__main__":
    main()
