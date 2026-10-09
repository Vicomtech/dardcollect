"""Input reading and sidecar writing for the per-crop OFIQ quality annotation.

Single implementation shared by the library entry point (`quality.score_video`)
and the pipeline stage (`pipeline/annotate_face_quality.py`), so both write the
same `<crop>.ofiq_attr.json`, validated against `schemas/quality_annotation_schema.json`.
MagFace (`unified_score`) is not repeated here: it lives in `<crop>.magface.json`.
"""

import json
import logging
import tempfile
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

OFIQ_ATTR_SUFFIX = ".ofiq_attr.json"

# Values the schema's `annotator` enum accepts; each writer names itself.
ANNOTATOR_PIPELINE = "pipeline/annotate_face_quality.py"
ANNOTATOR_LIBRARY = "dardcollect/quality.py"


@dataclass
class StrideSampling:
    """Stride-sampling policy: score every ``frame_stride``-th frame, cap at ``max_frames``."""

    frame_stride: int
    max_frames: int


@dataclass(frozen=True)
class CropProvenance:
    """What an OFIQ sidecar needs from the crop's own sidecar."""

    sidecar_data: dict
    source_video: str
    parent_uuid: str
    has_arcface_annotation: bool


def read_crop_provenance(crop_path: Path) -> CropProvenance | None:
    """Read the crop sidecar (``<crop>.json``); None when it cannot anchor an annotation.

    The OFIQ sidecar requires ``parent_crop``, so a crop without a sidecar or
    without a UUID is skipped with a warning rather than written unlinked.
    """
    sidecar_path = crop_path.with_suffix(".json")
    if not sidecar_path.exists():
        logger.warning("No sidecar JSON for %s; OFIQ annotation skipped", crop_path.name)
        return None
    try:
        with open(sidecar_path, encoding="utf-8") as f:
            sidecar_data = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Could not read sidecar for %s: %s", crop_path.name, exc)
        return None
    parent_uuid = sidecar_data.get("uuid")
    if not isinstance(parent_uuid, str) or not parent_uuid:
        logger.warning("Sidecar of %s has no UUID; OFIQ annotation skipped", crop_path.name)
        return None
    return CropProvenance(
        sidecar_data=sidecar_data,
        source_video=sidecar_data.get("source_video", ""),
        parent_uuid=parent_uuid,
        has_arcface_annotation=sidecar_data.get("crop_format") == "ofiq",
    )


@dataclass(frozen=True)
class OfiqAttrRequest:
    """Everything one OFIQ sidecar is built from."""

    crop_path: Path
    sidecar_path: Path
    provenance: CropProvenance
    frame_scores: list
    sampling: StrideSampling
    annotator: str


def build_ofiq_attr(req: OfiqAttrRequest, aggregate) -> dict:
    """Assemble the FAIR, schema-validated OFIQ sidecar for one crop.

    *aggregate* is `quality.aggregate_frame_scores`, passed in to avoid a circular
    import. Raises ``ValueError`` if the result violates the schema.
    """
    crop_path, provenance, frame_scores = req.crop_path, req.provenance, req.frame_scores
    from dardcollect.fair import (
        Provenance,
        add_fair_metadata,
        reorganize_for_fair,
        validate_against_schema,
    )
    from dardcollect.provenance import now_iso

    data: dict = {
        "face_crop_video": crop_path.name,
        "face_crop_json": req.sidecar_path.name,
        "source_video": provenance.source_video,
        "annotated_at": now_iso(),
        "annotator": req.annotator,
        "frame_stride": req.sampling.frame_stride,
        "max_frames_sampled": req.sampling.max_frames,
        "frame_data": frame_scores,
        **aggregate(frame_scores),
    }
    add_fair_metadata(
        data,
        schema_type="quality_annotation",
        provenance=Provenance(
            parent_uuid=provenance.parent_uuid, parent_file=req.sidecar_path.name
        ),
    )
    data = reorganize_for_fair(data)
    try:
        validate_against_schema(data, "quality_annotation")
    except Exception as exc:
        raise ValueError(f"OFIQ sidecar for {crop_path.name} violates the schema: {exc}") from exc
    return data


def write_json_atomically(data: dict, output_path: Path) -> bool:
    """Write JSON via a temporary file and rename: an interruption leaves no partial file."""
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", dir=output_path.parent, delete=False, encoding="utf-8"
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
            except OSError:
                pass
        return False
