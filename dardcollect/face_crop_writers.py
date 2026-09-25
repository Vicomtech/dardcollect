"""Per-track OFIQ face-crop writers for `face_crops.process_video`.

Split out of `dardcollect/face_crops.py` (2026-09-23) when that file crossed the
600-line god-file cap. Holds the frame-collection helpers and the per-track
video/sidecar writer; `face_crops.py` keeps detection, accumulation and
`process_video`.

The library never imports pipeline stage scripts; this is the allowed direction
(pipeline -> library).
"""

import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from dardcollect.audio import _mux_audio
from dardcollect.config import FaceCropConfig
from dardcollect.face_geometry import (
    OFIQ_SIZE,
    _corners_to_warp,
    _get_or_compute_corners,
    _transform_bbox,
    _transform_keypoints,
    compute_track_mean_corners,
)
from dardcollect.fair import (
    Provenance,
    add_fair_metadata,
    reorganize_for_fair,
    validate_against_schema,
)
from dardcollect.pipeline_loggers import FaceCropsExtractionLogger
from dardcollect.pipeline_utils import check_disk_space
from dardcollect.video_writers import (
    _cleanup_files,
    _write_video_with_moviepy,
)

if TYPE_CHECKING:
    from dardcollect.encoding_config import EncodingConfig

logger = logging.getLogger(__name__)


@dataclass
class _CropWriteContext:
    """Per-video state for writing track crops, bundled to keep arity low."""

    video_path: Path
    clip_data: dict
    frame_data_orig: dict
    start_frame: int
    face_config: FaceCropConfig
    output_dir: Path
    fps: float
    encoding: "EncodingConfig | None"
    face_crops_logger: FaceCropsExtractionLogger | None
    track_corners: dict
    black_ofiq: np.ndarray
    arcface_corners_json: list


def _build_track_frame_entry(
    det: dict, tid: int, face_config: FaceCropConfig, arcface_corners_json: list
) -> dict | None:
    """Build a frame_data entry for a track's detection, or None if it has no
    usable keypoints/corners. Shared by both skip-no-face and keep-all paths."""
    kpts = det.get("keypoints", [])
    scores = det.get("keypoint_scores", [])
    corners = _get_or_compute_corners(det, face_config)
    if corners is None or not kpts or not scores:
        return None
    kpts_array = np.array(kpts, dtype=np.float32)
    scores_array = np.array(scores, dtype=np.float32)
    transformed_kpts, _, M = _transform_keypoints(kpts, scores, kpts_array, scores_array)
    entry: dict = {
        "track_id": tid,
        "score": det.get("score"),
        "keypoints": transformed_kpts,
        "keypoint_scores": scores,
        "face_crop_corners_arcface": arcface_corners_json,
    }
    if M is not None and det.get("bbox"):
        entry["bbox"] = _transform_bbox(det["bbox"], M)
    return entry


def _frame_data_entry_for(
    ctx: _CropWriteContext, abs_frame: int, tid: int, output_frame_idx: int, out: dict
) -> None:
    """Record this track's detection for *abs_frame* as frame_data[output_frame_idx]."""
    for det in ctx.frame_data_orig.get(str(abs_frame), []):
        if det.get("track_id") == tid:
            entry = _build_track_frame_entry(det, tid, ctx.face_config, ctx.arcface_corners_json)
            if entry is not None:
                out[str(output_frame_idx)] = [entry]
            break


def _collect_track_frames_skip_no_face(
    ctx: _CropWriteContext,
    tid: int,
    valid_frames: list,
) -> tuple[list, dict]:
    """Collect OFIQ frames + frame_data, dropping frames with no face (skip mode)."""
    frames_to_write: list = []
    frame_data: dict = {}
    output_frame_idx = 0
    for fid, oc in valid_frames:
        if oc is not None:
            frames_to_write.append(oc)
            _frame_data_entry_for(ctx, ctx.start_frame + fid, tid, output_frame_idx, frame_data)
            output_frame_idx += 1
    return frames_to_write, frame_data


def _collect_track_frames_keep_all(
    ctx: _CropWriteContext,
    tid: int,
    frames: list,
) -> tuple[list, dict]:
    """Collect OFIQ frames + frame_data, filling gaps with the last seen frame
    (keep-all mode — output video spans the full track, audio stays in sync)."""
    first_fid = frames[0][0]
    last_fid = frames[-1][0]
    ofiq_dict = {fid: oc for fid, oc in frames if oc is not None}
    last_ofiq = ctx.black_ofiq
    frames_to_write: list = []
    frame_data: dict = {}
    output_frame_idx = 0
    for fid in range(first_fid, last_fid + 1):
        if fid in ofiq_dict:
            last_ofiq = ofiq_dict[fid]
        frames_to_write.append(last_ofiq)
        _frame_data_entry_for(ctx, ctx.start_frame + fid, tid, output_frame_idx, frame_data)
        output_frame_idx += 1
    return frames_to_write, frame_data


def _log_face_crop(
    face_crops_logger: FaceCropsExtractionLogger | None,
    video_path: Path,
    frame_data: dict,
    ofiq_path: Path,
) -> None:
    """Log a face-crop extraction to the traceability CSV (best-effort confidence
    from the first output frame's first detection)."""
    if face_crops_logger is None:
        return
    avg_confidence = 0.5
    if frame_data and frame_data.get("0"):
        detections = frame_data.get("0", [])
        if detections and isinstance(detections[0], dict):
            score = detections[0].get("score", 0.5)
            avg_confidence = float(score) if score else 0.5
    face_crops_logger.log_face_crop_extraction(
        source_type="person_clip",
        source_path=str(video_path),
        face_bbox=f"0,0,{OFIQ_SIZE},{OFIQ_SIZE}",  # Full OFIQ frame
        confidence=avg_confidence,
        output_path=str(ofiq_path),
    )


def _stabilized_frames_for_track(
    tid: int,
    frames_to_write: list,
    track_corners: dict,
    face_config: FaceCropConfig,
) -> list:
    """Issue #9 (opt-in): re-render source frames through the track-median OFIQ
    quad when stabilization is on and enough stable frames exist. Falls back to
    the collected (per-frame-rendered) frames otherwise."""
    if not face_config.stabilize_face_crops:
        return frames_to_write
    median_corners = compute_track_mean_corners(
        track_corners.get(tid, []), face_config.stabilization_min_frames
    )
    stable_n = len([c for c in track_corners.get(tid, []) if c is not None])
    if median_corners is None:
        logger.info(
            "  Track %d: stabilization requested but < %d stable corners — per-frame fallback",
            tid,
            face_config.stabilization_min_frames,
        )
        return frames_to_write
    logger.info("  Track %d: stabilization engaged (%d stable frames)", tid, stable_n)
    return [_corners_to_warp(f, median_corners, OFIQ_SIZE) for f in frames_to_write]


def _build_face_crop_meta(
    ctx: _CropWriteContext,
    tid: int,
    frame_data: dict,
    frames: list,
    valid_frames: list,
) -> dict:
    """Build the FAIR face-crop sidecar dict for one track."""
    first_fid, last_fid = frames[0][0], frames[-1][0]
    duration_seconds = round((last_fid - first_fid + 1) / ctx.fps, 3) if ctx.fps > 0 else 0
    meta = {
        "source_video": str(ctx.video_path),
        "track_id": tid,
        "start_frame": 0,
        "end_frame": last_fid - first_fid,
        "start_seconds": 0.0,
        "end_seconds": duration_seconds,
        "duration_seconds": duration_seconds,
        "video_info": {
            "fps": round(ctx.fps, 3),
            "width": OFIQ_SIZE,
            "height": OFIQ_SIZE,
            "duration_seconds": duration_seconds,
        },
        "valid_face_frames": len(valid_frames),
        "crop_format": "ofiq",
        "output_size": OFIQ_SIZE,
        "frame_data": frame_data,
    }
    return add_fair_metadata(
        meta,
        schema_type="face_crop",
        provenance=Provenance(
            parent_uuid=ctx.clip_data.get("uuid"),
            parent_file=ctx.video_path.name,
        ),
    )


def _collect_track_frames_for_write(
    ctx: _CropWriteContext,
    tid: int,
    frames: list,
    valid_frames: list,
) -> tuple[list, dict]:
    """Collect the frames + frame_data to write for one track (either policy)."""
    if ctx.face_config.skip_no_face_frames:
        return _collect_track_frames_skip_no_face(ctx, tid, valid_frames)
    return _collect_track_frames_keep_all(ctx, tid, frames)


def _write_track_sidecar(
    meta: dict,
    ofiq_path: Path,
    ofiq_sidecar: Path,
) -> None:
    """Validate and write one track's FAIR sidecar; exit loudly on write failure."""
    try:
        meta = reorganize_for_fair(meta)
        # Validate the FAIR sidecar against the ratified schema before write
        # (per the project's "validate at write" contract). A ValidationError
        # propagates (not an OSError) — fail loudly rather than persist a
        # non-conforming sidecar.
        validate_against_schema(meta, "face_crop")
        with open(ofiq_sidecar, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
    except OSError as e:
        logger.error("Cannot write %s (%s) — stopping.", ofiq_sidecar.name, e)
        _cleanup_files(ofiq_path, ofiq_sidecar)
        sys.exit(1)


def _write_track_crop(
    ctx: _CropWriteContext,
    tid: int,
    frames: list,
    valid_frames: list,
) -> bool:
    """Write one track's OFIQ crop video + sidecar. Returns True if written.

    Returns False when the track is skipped (too few valid frames or already
    complete); exits the process on a write failure (fail loudly, matching the
    original behavior).
    """
    face_config = ctx.face_config
    if len(valid_frames) < face_config.min_track_face_frames:
        logger.debug(
            "  Track %d: only %d valid face frame(s), skipping",
            tid,
            len(valid_frames),
        )
        return False

    stem = f"{ctx.video_path.stem}_face_{tid}"
    ofiq_path = ctx.output_dir / f"{stem}.mp4"
    ofiq_sidecar = ofiq_path.with_suffix(".json")

    if ofiq_path.exists() and ofiq_sidecar.exists():
        logger.info("  SKIP (already complete): %s", stem)
        return False

    if ofiq_path.exists():
        logger.info("  Incomplete write detected, cleaning up: %s", stem)
        _cleanup_files(ofiq_path, ofiq_sidecar)

    check_disk_space(ctx.output_dir, face_config.min_free_disk_gb)

    frames_to_write, frame_data = _collect_track_frames_for_write(ctx, tid, frames, valid_frames)

    # Issue #9 (opt-in): when stabilization collected SOURCE frames, this
    # re-renders them through the track-median quad (same order/count, so
    # frame_data alignment is preserved).
    frames_to_write = _stabilized_frames_for_track(
        tid, frames_to_write, ctx.track_corners, face_config
    )

    # Write video using moviepy (encoding config: issue #8, defaults = libx264)
    if not _write_video_with_moviepy(frames_to_write, ofiq_path, ctx.fps, ctx.encoding):
        logger.error("Failed to write video for %s — stopping.", stem)
        _cleanup_files(ofiq_path)
        sys.exit(1)

    first_fid, last_fid = frames[0][0], frames[-1][0]
    if face_config.include_audio and not face_config.skip_no_face_frames:
        _mux_audio(ctx.video_path, ofiq_path, first_fid / ctx.fps, (last_fid + 1) / ctx.fps)

    meta = _build_face_crop_meta(ctx, tid, frame_data, frames, valid_frames)
    _write_track_sidecar(meta, ofiq_path, ofiq_sidecar)

    _log_face_crop(ctx.face_crops_logger, ctx.video_path, frame_data, ofiq_path)

    logger.info(
        "  Wrote %s  (%d valid face frames / %d span frames)",
        stem,
        len(valid_frames),
        last_fid - first_fid + 1,
    )
    return True
