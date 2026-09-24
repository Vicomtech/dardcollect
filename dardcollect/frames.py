"""Frame extraction from video clips.

Extracts individual PNG frames from video files, with per-frame JSON sidecars
containing FAIR metadata (UUID, timestamp, detection data). Supports resumable
extraction — skips frames that already have both .png and .json files.

Intended for use after person clip extraction or face crop filtering, where
frame-level data is needed for downstream tasks (pose estimation, quality
annotation, etc.).
"""

import json
import logging
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import cv2

from dardcollect.fair import add_fair_metadata, generate_uuid, reorganize_for_fair
from dardcollect.pipeline_loggers import FramesExtractionLogger
from dardcollect.pipeline_utils import make_tqdm

logger = logging.getLogger(__name__)


def _frame_has_face(frame_detections: object) -> bool:
    """Return whether frame detections contain a usable face annotation."""
    if not isinstance(frame_detections, list) or not frame_detections:
        return False

    for det in frame_detections:
        if not isinstance(det, Mapping):
            continue
        det_map = cast(Mapping[str, Any], det)
        keypoints = det_map.get("keypoints")
        if isinstance(keypoints, list) and keypoints:
            return True
    return False


def _relist_existing_frame(frame_json: Path, frame_number: int, manifest: dict) -> None:
    """Append an already-extracted frame to the manifest.

    The manifest is rebuilt each call, so a resumed frame must be re-listed even
    though nothing is written this pass. Deliberately NOT logged to
    frames_extraction.csv (append-only; its row came from the first pass).
    """
    try:
        existing = json.loads(frame_json.read_text(encoding="utf-8"))
        manifest["frames"].append(
            {
                "frame_number": frame_number,
                "uuid": existing.get("uuid"),
                "timestamp": existing.get("timestamp", 0.0),
            }
        )
    except Exception as e:
        logger.warning("Cannot re-list existing frame %s: %s", frame_json.name, e)


def _remove_frame_outputs(output_dir: Path, frame_number: int) -> None:
    """Delete the png/json/mask outputs of a frame (used when overwriting skips)."""
    for suffix in (".png", ".json", "_mask.png"):
        path = output_dir / f"frame_{frame_number:06d}{suffix}"
        if path.exists():
            path.unlink()


@dataclass
class _FrameContext:
    """Per-video frame-writing context, bundled to keep helpers low-arity."""

    fps: float
    clip_type: str
    clip_start_frame: int
    frame_data_dict: dict
    parent_uuid: str | None
    parent_file: str
    video_path: Path


def _build_frame_meta(frame_number: int, frame_uuid: str, ctx: _FrameContext) -> dict:
    """Build a frame sidecar dict with FAIR metadata and its frame-specific UUID."""
    frame_key = str(ctx.clip_start_frame + frame_number)
    frame_detections = (
        ctx.frame_data_dict.get(frame_key, []) if isinstance(ctx.frame_data_dict, dict) else []
    )
    frame_meta = {
        "frame_number": frame_number,
        "timestamp": frame_number / ctx.fps if ctx.fps > 0 else 0.0,
        "detections": frame_detections,
    }
    schema = "face_crop" if "face" in ctx.clip_type else "person_clip"
    frame_meta = add_fair_metadata(
        frame_meta,
        schema_type=schema,
        parent_uuid=ctx.parent_uuid,
        parent_file=ctx.parent_file,
    )
    frame_meta["uuid"] = frame_uuid  # override with frame-specific UUID
    return reorganize_for_fair(frame_meta)


@dataclass
class _FrameRunContext:
    """Per-run state for the frame loop, bundled to keep the loop helper low-arity."""

    ctx: _FrameContext
    output_dir: Path
    overwrite: bool
    frames_logger: FramesExtractionLogger | None
    frame_manifest: dict


def _process_one_frame(run: _FrameRunContext, frame, frame_number: int) -> str:
    """Handle one decoded frame; returns a status string.

    Statuses: ``"written"`` (new sidecar written), ``"relisted"`` (resumed,
    already present), ``"skipped"`` (face-crop frame with no face), or an
    ``"error_*"`` code (write failed; logged, frame skipped).
    """
    ctx = run.ctx
    frame_png = run.output_dir / f"frame_{frame_number:06d}.png"
    frame_json = run.output_dir / f"frame_{frame_number:06d}.json"

    if frame_png.exists() and frame_json.exists() and not run.overwrite:
        _relist_existing_frame(frame_json, frame_number, run.frame_manifest)
        return "relisted"

    frame_detections = (
        ctx.frame_data_dict.get(str(ctx.clip_start_frame + frame_number), [])
        if isinstance(ctx.frame_data_dict, dict)
        else []
    )
    # For face-crop clips, keep only frames that still have face annotations.
    if "face" in ctx.clip_type and not _frame_has_face(frame_detections):
        if run.overwrite:
            _remove_frame_outputs(run.output_dir, frame_number)
        return "skipped"

    frame_uuid = generate_uuid()
    frame_meta = _build_frame_meta(frame_number, frame_uuid, ctx)

    try:
        cv2.imwrite(str(frame_png), frame)
    except Exception as e:
        logger.error("Cannot write frame PNG %s: %s", frame_png.name, e)
        return "error_png"

    try:
        with open(frame_json, "w", encoding="utf-8") as f:
            json.dump(frame_meta, f, indent=2)
    except Exception as e:
        logger.error("Cannot write frame JSON %s: %s", frame_json.name, e)
        return "error_json"

    run.frame_manifest["frames"].append(
        {
            "frame_number": frame_number,
            "uuid": frame_uuid,
            "timestamp": frame_meta.get("timestamp", 0.0),
        }
    )

    if run.frames_logger is not None:
        run.frames_logger.log_frame_extraction(
            source_clip_path=str(ctx.video_path),
            frame_number=frame_number,
            timestamp_seconds=frame_number / ctx.fps if ctx.fps > 0 else 0.0,
            output_path=str(frame_png),
        )
    return "written"


@dataclass
class _FrameSidecarInfo:
    """Provenance read from a clip sidecar, bundled for the frame extractor."""

    parent_uuid: str | None
    parent_file: str
    frame_data_dict: dict
    clip_start_frame: int


def _read_frame_sidecar(video_path: Path, sidecar_path: Path) -> _FrameSidecarInfo | None:
    """Read the clip sidecar; None (logged) when missing or unreadable.

    ``clip_start_frame`` matters: person-clip sidecars key ``frame_data`` by
    ABSOLUTE source-video frame number (a clip cut at 28m56s starts at key
    "43411"), while face-crop sidecars key it 0-based with ``start_frame == 0``.
    Offsetting the per-frame lookup by ``start_frame`` resolves both — without it
    every lookup misses and each frame sidecar is written with
    ``"detections": []``, silently stripping detection provenance and leaving
    generate_face_masks.py with nothing to draw.
    """
    if not sidecar_path.exists():
        logger.warning("No sidecar for %s, skipping", video_path.name)
        return None
    try:
        with open(sidecar_path, encoding="utf-8") as f:
            sidecar_data = json.load(f)
    except Exception as e:
        logger.error("Cannot read sidecar %s: %s", sidecar_path.name, e)
        return None

    try:
        clip_start_frame = int(sidecar_data.get("start_frame") or 0)
    except (TypeError, ValueError):
        clip_start_frame = 0
    return _FrameSidecarInfo(
        parent_uuid=sidecar_data.get("uuid"),
        parent_file=video_path.name,
        frame_data_dict=sidecar_data.get("frame_data", {}),
        clip_start_frame=clip_start_frame,
    )


def _write_frame_manifest(manifest_path: Path, frame_manifest: dict) -> bool:
    """Write the frames manifest; True on success, False on write failure."""
    try:
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(frame_manifest, f, indent=2)
    except Exception as e:
        logger.error("Cannot write manifest %s: %s", manifest_path.name, e)
        return False
    return True


def extract_frames(
    video_path: Path,
    sidecar_path: Path,
    output_dir: Path,
    clip_type: str,
    overwrite: bool = False,
    frames_logger: FramesExtractionLogger | None = None,
) -> dict | None:
    """Extract all frames from a video as PNG images with per-frame JSON sidecars.

    Reads detection data from the sidecar JSON and embeds it in each frame's
    metadata. Resumable: skips frames whose .png and .json already exist unless
    *overwrite* is True.

    Args:
        video_path: Path to the source video file.
        sidecar_path: Path to the JSON sidecar with FAIR metadata and frame_data.
        output_dir: Directory where frame PNGs and JSONs will be written.
        clip_type: Tag for FAIR schema selection. Use 'person_clip' for general
            clips, 'face_crop' or 'filtered_face_crop' for face crops.
        overwrite: If True, re-extract frames even if they already exist.
        frames_logger: Optional logger for frame extraction events.

    Returns:
        dict: Manifest with source info and list of all extracted frames with UUIDs,
            or None if the sidecar is missing, video cannot be opened, or write fails.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    info = _read_frame_sidecar(video_path, sidecar_path)
    if info is None:
        return None

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logger.error("Cannot open video: %s", video_path.name)
        return None

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    logger.info("  %s  %d frames @ %.1f fps", video_path.name, total_frames, fps)

    frame_manifest = {
        "source_video": str(video_path),
        "parent_uuid": info.parent_uuid,
        "parent_file": info.parent_file,
        "source_sidecar": sidecar_path.name,
        "clip_type": clip_type,
        "total_frames": total_frames,
        "fps": fps,
        "frames": [],
    }

    frame_ctx = _FrameContext(
        fps=fps,
        clip_type=clip_type,
        clip_start_frame=info.clip_start_frame,
        frame_data_dict=info.frame_data_dict,
        parent_uuid=info.parent_uuid,
        parent_file=info.parent_file,
        video_path=video_path,
    )

    frame_count = 0
    pbar = make_tqdm(
        total=total_frames, unit="frame", desc=video_path.stem[:40], dynamic_ncols=True
    )
    run = _FrameRunContext(
        ctx=frame_ctx,
        output_dir=output_dir,
        overwrite=overwrite,
        frames_logger=frames_logger,
        frame_manifest=frame_manifest,
    )

    try:
        frame_number = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if _process_one_frame(run, frame, frame_number) == "written":
                frame_count += 1
            pbar.update(1)
            frame_number += 1
    finally:
        cap.release()
        pbar.close()

    if not _write_frame_manifest(output_dir / "frames_manifest.json", frame_manifest):
        return None

    logger.info("  Extracted %d frames → %s", frame_count, output_dir.name)
    return frame_manifest
