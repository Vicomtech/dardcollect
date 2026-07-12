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
from tqdm import tqdm

from dardcollect.pipeline_utils import _check_disk_space, _TqdmHandler
from dardcollect.quality import score_all_magface_frames

_handler = _TqdmHandler()
_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
logging.basicConfig(handlers=[_handler], level=logging.INFO, force=True)
logger = logging.getLogger(__name__)

CONFIG_PATH = Path(
    os.environ.get("DARDCOLLECT_CONFIG", Path(__file__).resolve().parent.parent / "config.yaml")
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
    except Exception:
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
    except Exception:
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

    # Load both configurations
    cfg_video = FaceQualityFilterConfig.from_yaml(config_path, section="face_quality_filtering")
    cfg_image = FaceQualityFilterConfig.from_yaml(
        config_path, section="image_face_quality_filtering"
    )

    input_dir_video = Path(cfg_video.input_dir)
    input_dir_image = Path(cfg_image.input_dir)

    # Process both video and image crops if they exist
    configs = []
    if input_dir_video.exists():
        configs.append(("video", cfg_video, input_dir_video))
    if input_dir_image.exists():
        configs.append(("image", cfg_image, input_dir_image))

    if not configs:
        raise FileNotFoundError(
            "No input directories found in config. "
            "Run extract_face_crops_from_videos.py or extract_face_crops_from_images.py first."
        )

    logging.getLogger().setLevel(get_log_level(config_path))

    # Load MagFace session once (used for all modalities)
    session = load_magface(cfg_video.gpu_id)

    # Process each modality
    for modality, cfg, input_dir in configs:
        output_dir = Path(cfg.output_dir)

        if not input_dir.exists():
            logger.warning("Input directory does not exist: %s — skipping", input_dir)
            continue

        output_dir.mkdir(parents=True, exist_ok=True)
        _check_disk_space(output_dir, cfg.min_free_disk_gb)

        # Find crops
        crop_files = sorted(input_dir.glob("*_face_*.mp4"))
        crop_files.extend(sorted(input_dir.glob("*_face_*.jpg")))
        crop_files.extend(sorted(input_dir.glob("*_face_*.png")))
        crop_files = sorted(set(crop_files))  # Remove duplicates and re-sort
        crop_files = [p for p in crop_files if not p.name.endswith("_mask.png")]

        if not crop_files:
            logger.info("No face crops found in %s", input_dir)
            continue

        logger.info(
            "Processing %s: %d OFIQ face crops — quality threshold %.2f",
            modality,
            len(crop_files),
            cfg.quality_threshold,
        )

        videos_assessed = 0
        videos_passed = 0
        videos_skipped = 0
        all_scores = []

        # Initialize filtered crops logger — use whichever extraction CSV exists
        face_crops_csv = input_dir / "image_face_crops_extraction.csv"
        if not face_crops_csv.exists():
            face_crops_csv = input_dir / "video_face_crops_extraction.csv"
        filter_logger = FilteredFaceCropsLogger(
            output_dir=str(output_dir), face_crops_csv_path=face_crops_csv
        )

        for crop_path in tqdm(crop_files, desc=f"Quality filtering ({modality})", unit="crop"):
            sidecar_path = crop_path.with_suffix(".json")
            magface_path = crop_path.with_suffix(".magface.json")
            dest_crop = output_dir / crop_path.name
            dest_sidecar = output_dir / sidecar_path.name
            dest_magface = output_dir / magface_path.name
            max_score: float | None = None

            # Idempotency: already moved
            if dest_crop.exists():
                if modality == "image" and dest_sidecar.exists():
                    _refresh_image_parent_uuid(dest_sidecar, input_dir)
                logger.debug("Already in output dir, skipping: %s", crop_path.name)
                videos_skipped += 1
                continue

            if not sidecar_path.exists():
                logger.info("Missing sidecar JSON for %s — skipping", crop_path.name)
                continue

            if modality == "image":
                _refresh_image_parent_uuid(sidecar_path, input_dir)

            # Reuse existing .magface.json when possible, but still evaluate threshold.
            if magface_path.exists():
                try:
                    with open(magface_path, encoding="utf-8") as f:
                        existing_magface = json.load(f)
                    unified = existing_magface.get("unified_score", {})
                    if isinstance(unified, dict) and "max" in unified:
                        max_score = float(unified["max"])
                        logger.debug("Reusing existing MagFace for %s", crop_path.name)
                    else:
                        logger.warning(
                            "%s exists but has no unified_score.max, will recompute",
                            magface_path.name,
                        )
                except Exception as exc:
                    logger.warning(
                        "%s exists but is corrupted (%s), will recompute", magface_path.name, exc
                    )

            if max_score is None:
                _check_disk_space(output_dir, cfg.min_free_disk_gb)

                try:
                    magface_data = score_all_magface_frames(crop_path, session)
                except Exception as exc:
                    logger.error("Error assessing %s: %s", crop_path.name, exc)
                    continue

                if not magface_data:
                    logger.warning("Failed to score frames for %s", crop_path.name)
                    continue

                max_score = float(magface_data["max"])

                # Read sidecar for provenance
                source_video = ""
                try:
                    with open(sidecar_path, encoding="utf-8") as f:
                        sidecar = json.load(f)
                    source_video = sidecar.get("source_video", "")
                except Exception:
                    pass

                # Save MagFace scores to .magface.json
                magface_json = {
                    "face_crop_video": crop_path.name,
                    "source_video": source_video,
                    "annotated_at": now_iso(),
                    "annotator": "pipeline/filter_face_crops_by_quality.py",
                    # unified_score contains per-frame scores + aggregated stats
                    "unified_score": magface_data,
                }

                if not _write_atomically(magface_json, magface_path):
                    logger.error("Failed to save .magface.json for %s, skipping", crop_path.name)
                    continue

            videos_assessed += 1
            all_scores.append(max_score)

            # Check if passes quality threshold
            passes = max_score >= cfg.quality_threshold

            if passes:
                shutil.move(str(crop_path), dest_crop)
                shutil.move(str(sidecar_path), dest_sidecar)
                shutil.move(str(magface_path), dest_magface)
                videos_passed += 1

                # Log filtered crop (for traceability)
                filter_logger.log_filtered_crop(
                    source_crop_path=str(crop_path),
                    magface_score=float(max_score),
                    filter_threshold=float(cfg.quality_threshold),
                    output_path=str(dest_crop),
                )

                logger.info("PASS %s (score=%.4f) → %s", crop_path.name, max_score, output_dir)
            else:
                logger.debug(
                    "FAIL %s (score=%.4f) — .magface.json retained for annotation",
                    crop_path.name,
                    max_score,
                )

        logger.info(
            "[%s] Done. Assessed: %d  Passed: %d  Skipped (already done): %d",
            modality,
            videos_assessed,
            videos_passed,
            videos_skipped,
        )

        # Print quality score distribution
        if all_scores:
            scores_array = np.array(all_scores)
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

        # Print filtered crops summary
        filter_logger.print_summary()


if __name__ == "__main__":
    main()
