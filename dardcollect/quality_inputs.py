"""Input reading + sidecar assembly for the per-crop quality annotation path.

Split out of `dardcollect/quality.py` (2026-09-23) so that file stays under its
god-file baseline. `quality.score_video` (the public entry point) calls these.
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class StrideSampling:
    """Stride-sampling policy: score every ``frame_stride``-th frame, cap at ``max_frames``."""

    frame_stride: int
    max_frames: int


@dataclass
class QualityInputs:
    """Resolved inputs for one crop's quality annotation, bundled for low arity."""

    crop_path: Path
    sidecar_path: Path
    sidecar_data: dict | None
    source_video: str
    magface_unified_score: dict | None
    frame_stride: int
    max_frames: int


def read_quality_inputs(
    crop_path: Path,
    sidecar_path: Path,
    magface_path: Path,
) -> tuple[dict | None, str, bool, dict | None]:
    """Read the crop sidecar and any cached MagFace scores.

    Returns ``(sidecar_data, source_video, has_arcface_annotation,
    magface_unified_score)``; each missing/unreadable input is warned and yields
    its default.
    """
    source_video = ""
    has_arcface_annotation = False
    sidecar_data: dict | None = None
    if sidecar_path.exists():
        try:
            with open(sidecar_path, encoding="utf-8") as f:
                sidecar_data = json.load(f)
            source_video = sidecar_data.get("source_video", "")
            has_arcface_annotation = sidecar_data.get("crop_format") == "ofiq"
        except Exception as exc:
            logger.warning("Failed to read sidecar %s: %s", sidecar_path.name, exc)
    else:
        logger.warning(
            "No sidecar JSON alongside %s — provenance will be incomplete", crop_path.name
        )

    magface_unified_score: dict | None = None
    if magface_path.exists():
        try:
            with open(magface_path, encoding="utf-8") as f:
                magface_data = json.load(f)
            magface_unified_score = magface_data.get("unified_score")
            logger.debug("Read MagFace scores from %s", magface_path.name)
        except Exception as exc:
            logger.warning("Failed to read %s: %s", magface_path.name, exc)
    return sidecar_data, source_video, has_arcface_annotation, magface_unified_score


def build_quality_data(inputs: QualityInputs, frame_scores: list, aggregate) -> dict:
    """Assemble the quality-annotation dict (FAIR metadata + provenance links).

    *aggregate* is `quality.aggregate_frame_scores`, passed in to avoid a circular
    import between this module and `quality.py`.
    """
    from dardcollect.fair import Provenance, add_fair_metadata
    from dardcollect.provenance import now_iso

    quality_data: dict = {
        "face_crop_video": inputs.crop_path.name,
        "face_crop_json": inputs.sidecar_path.name,
        "source_video": inputs.source_video,
        "annotated_at": now_iso(),
        "annotator": "pipeline/annotate_face_quality.py",
        "frame_stride": inputs.frame_stride,
        "max_frames_sampled": inputs.max_frames,
        "frame_data": frame_scores,  # Per-frame quality scores
        **aggregate(frame_scores),
    }
    if inputs.magface_unified_score:
        quality_data["unified_score"] = inputs.magface_unified_score
    return add_fair_metadata(
        quality_data,
        schema_type="quality_annotation",
        provenance=Provenance(
            parent_uuid=inputs.sidecar_data.get("uuid") if inputs.sidecar_data else None,
            parent_file=inputs.crop_path.name,
        ),
    )
