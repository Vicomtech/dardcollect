"""person_clips_flush.py — merge/filter/split/write path for clip extraction.

Split from person_clips_run.py (god-file ratchet): the frame loop reads and
tracks, this module writes. ``_flush_batch`` merges candidate segments,
filters, splits over-long ones, smooths keypoints, and writes clip videos +
JSON sidecars.
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from dardcollect.clip_extraction import ClipBatchContext, extract_clips
from dardcollect.config import ClipExtractionConfig, FaceCropConfig
from dardcollect.extraction_logger import ClipRecord, ExtractionLogger
from dardcollect.face_geometry import _annotate_face_crop_corners
from dardcollect.fair import reorganize_for_fair
from dardcollect.person_clips_helpers import apply_duration_split
from dardcollect.pipeline_utils import check_disk_space
from dardcollect.poser import PoseEstimator
from dardcollect.tracker import Segment, merge_segments, smooth_segment_keypoints
from dardcollect.video_writers import save_clip_sidecar_json

if TYPE_CHECKING:
    from dardcollect.encoding_config import EncodingConfig

logger = logging.getLogger(__name__)


@dataclass
class _FlushBatch:
    """One flush call: the segments plus the context they are written with."""

    segments: list[Segment]
    fps: float
    clip_config: ClipExtractionConfig
    poser: PoseEstimator | None
    face_crop_cfg: FaceCropConfig | None
    video_path: Path
    output_dir: Path
    video_info: dict
    clip_logger: ExtractionLogger | None
    source_path: Path
    encoding: "EncodingConfig | None" = None


def _filter_candidate_segments(
    merged: list[Segment],
    clip_config: ClipExtractionConfig,
    fps: float,
    poser: PoseEstimator | None,
) -> list[Segment]:
    """Apply the duration and (opt-in) face-visibility filters to merged segments."""
    filtered = [
        s
        for s in merged
        if s.frame_count >= clip_config.min_consecutive_frames
        and s.duration_seconds(fps) >= clip_config.min_clip_duration_seconds
    ]
    if clip_config.require_face_visibility and poser is not None:
        filtered = [
            s
            for s in filtered
            if s.face_visible_frames >= clip_config.min_face_visible_frames
            and s.max_consecutive_face_frames >= clip_config.min_consecutive_face_frames
        ]
    return filtered


def _read_source_identity(video_path: Path) -> tuple[str | None, str | None]:
    """Read (archive_org_id, archive_org_url) from the source sidecar, if present.

    The sidecar is the same for all clips of a source, so this is read once per
    flush rather than per clip.
    """
    try:
        sidecar = video_path.with_suffix(".json")
        if sidecar.exists():
            with open(sidecar, encoding="utf-8") as f:
                sidecar_data = json.load(f)
            return sidecar_data.get("identifier"), sidecar_data.get("url")
    except Exception as exc:
        logger.warning("Failed to read source sidecar %s: %s", sidecar.name, exc)
    return None, None


def _avg_detection_confidence(seg: Segment) -> float:
    """Mean detection score across the segment's frame_data (0.5 when none)."""
    all_scores = [
        d.get("score", 0.5)
        for detections in seg.frame_data.values()
        for d in detections
        if isinstance(d, dict)
    ]
    return sum(all_scores) / len(all_scores) if all_scores else 0.5


def _write_clip_result(
    r: dict,
    video_path: Path,
    fps: float,
    clip_logger: ExtractionLogger | None,
) -> dict:
    """Write one extracted clip's sidecar + CSV row (serialized, segment order).

    Returns the clip metadata dict (with the schema-consistency ``transcription``
    field always present).
    """
    seg: Segment = r["seg"]
    clip_path: Path = r["clip_path"]
    meta: dict = r["meta"]

    # Transcription is handled by transcribe_video_clips.py; keep field for schema consistency
    meta["transcription"] = ""

    if not r["success"]:
        return meta

    meta = reorganize_for_fair(meta)
    save_clip_sidecar_json(clip_path, meta)

    if clip_logger is not None:
        clip_logger.log_extraction(
            ClipRecord(
                source_video=video_path.name,
                fps=fps,
                start_frame=seg.start_frame,
                end_frame=seg.end_frame,
                start_seconds=r["start_sec"],
                duration_seconds=seg.duration_seconds(fps),
                max_persons_per_frame=seg.max_persons,
                detector_model="yolox-tiny",
                detector_confidence=_avg_detection_confidence(seg),
                output_path=str(clip_path),
            )
        )
    return meta


def _flush_batch(batch: _FlushBatch) -> list[dict]:
    """Merge, filter, split, smooth, and write a batch of candidate segments.

    The pipeline: merge adjacent segments → apply duration/face-visibility filters →
    split over-long segments → smooth keypoints per track → write clip videos and
    JSON sidecars. Returns clip metadata dicts for all successfully extracted clips.
    """
    if not batch.segments:
        return []
    # Read path for extract_clip: the local pre-copy if preloaded, else the original.
    # video_path stays the provenance/clip-name source (unchanged).
    read_path = batch.source_path if batch.source_path is not None else batch.video_path

    # Merge compatible segments within this batch
    merged = merge_segments(batch.segments, batch.clip_config.merge_gap_frames)

    filtered = _filter_candidate_segments(merged, batch.clip_config, batch.fps, batch.poser)

    # Handle max duration splitting
    filtered = apply_duration_split(
        filtered, batch.fps, batch.clip_config.max_clip_duration_seconds
    )

    for seg in filtered:
        smooth_segment_keypoints(seg, batch.fps)

    if batch.face_crop_cfg is not None:
        for seg in filtered:
            _annotate_face_crop_corners(seg, batch.face_crop_cfg)

    archive_org_id, archive_org_url = _read_source_identity(batch.video_path)

    check_disk_space(batch.output_dir, batch.clip_config.min_free_disk_gb)

    # Extract all clips (parallel when configured + >1 clip; results in segment order).
    results = extract_clips(
        filtered,
        ClipBatchContext(
            read_path=read_path,
            output_dir=batch.output_dir,
            fps=batch.fps,
            video_path=batch.video_path,
            video_info=batch.video_info,
            archive_org_id=archive_org_id,
            archive_org_url=archive_org_url,
            encoding=batch.encoding,
        ),
        batch.clip_config,
    )

    # Serialize the sidecar write + CSV log in segment order (thread-safe + deterministic,
    # identical to the old serial path). The heavy extraction already ran above.
    return [_write_clip_result(r, batch.video_path, batch.fps, batch.clip_logger) for r in results]
