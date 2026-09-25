"""Per-clip extraction dispatch (serial vs parallel) for the clips stage.

``flush_segments`` builds a batch of candidate segments; the N clips of a source are
independent (disjoint frame ranges of the same source), so their moviepy/ffmpeg extractions
can run concurrently. ``ThreadPoolExecutor`` overlaps the N ffmpeg encode subprocesses
(moviepy runs ffmpeg as a subprocess, releasing the GIL during the encode). Results are
returned in segment order so the caller's sidecar/CSV writes stay deterministic + identical
to the serial path.
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING

from dardcollect.config import ClipExtractionConfig
from dardcollect.fair import Provenance, add_fair_metadata
from dardcollect.tracker import Segment
from dardcollect.video_writers import ClipSpec, extract_clip

if TYPE_CHECKING:
    from dardcollect.encoding_config import EncodingConfig

logger = logging.getLogger(__name__)


@dataclass
class ClipBatchContext:
    """Per-source state shared by every clip extraction in a batch.

    Bundled so ``_extract_one_clip`` stays low-arity; ``read_path`` is the local
    pre-copy when preloading (the N concurrent reads hit local SSD) while
    ``video_path`` (provenance + clip name) stays the original source path.
    """

    read_path: Path
    output_dir: Path
    fps: float
    video_path: Path
    video_info: dict
    archive_org_id: str | None
    archive_org_url: str | None
    encoding: EncodingConfig | None = None


def _extract_one_clip(seg: Segment, ctx: ClipBatchContext) -> dict:
    """Build the clip metadata + extract one clip (the heavy moviepy/ffmpeg call).

    Returns ``{seg, clip_path, meta, success, start_sec}``. The sidecar write +
    ``clip_logger`` CSV append are NOT done here — the caller serializes them in segment order
    (thread-safe + deterministic) so the parallel path produces the same artifacts/CSV rows as
    serial.
    """
    fps = ctx.fps
    start_sec = seg.start_frame / fps
    end_sec = seg.end_frame / fps
    start_str = f"{int(start_sec // 60):02d}m{int(start_sec % 60):02d}s"
    end_str = f"{int(end_sec // 60):02d}m{int(end_sec % 60):02d}s"
    # Clip name derives from the source video stem (unique across input_dir subtree).
    clip_name = f"{ctx.video_path.stem}_{start_str}-{end_str}.mp4"
    clip_path = ctx.output_dir / clip_name

    meta = {
        "source_video": ctx.video_path.as_posix(),
        "start_frame": seg.start_frame,
        "end_frame": seg.end_frame,
        "start_seconds": round(start_sec, 2),
        "end_seconds": round(end_sec, 2),
        "duration_seconds": round(seg.duration_seconds(fps), 2),
        "max_persons": seg.max_persons,
        "unique_tracks": len(seg.track_ids),
        "track_ids": sorted(seg.track_ids),
        "video_info": ctx.video_info,
        "face_visible_frames": seg.face_visible_frames,
        "max_consecutive_face_frames": seg.max_consecutive_face_frames,
        "mouth_open_frames": seg.mouth_open_frames,
        "frame_data": seg.frame_data,
    }
    meta = add_fair_metadata(
        meta,
        schema_type="person_clip",
        provenance=Provenance(
            archive_org_id=ctx.archive_org_id,
            archive_org_url=ctx.archive_org_url,
        ),
    )

    logger.info("  Extracting: %s (%.1fs)", clip_name, meta["duration_seconds"])
    t0 = time.time()
    success = extract_clip(
        ClipSpec(
            input_path=ctx.read_path,
            output_path=clip_path,
            start_frame=seg.start_frame,
            end_frame=seg.end_frame,
            fps=fps,
            encoding=ctx.encoding,
        )
    )
    elapsed = time.time() - t0
    if success:
        logger.info("  Extraction took %.2fs", elapsed)
        meta["clip_path"] = clip_path.as_posix()
    else:
        meta["error"] = "Extraction failed"
    return {
        "seg": seg,
        "clip_path": clip_path,
        "meta": meta,
        "success": success,
        "start_sec": start_sec,
    }


def extract_clips(
    filtered: list[Segment],
    ctx: ClipBatchContext,
    clip_config: ClipExtractionConfig,
) -> list[dict]:
    """Extract all clips of a batch, parallel or serial (results in segment order).

    Parallel path uses a bounded ``ThreadPoolExecutor`` over ``_extract_one_clip``; moviepy
    runs ffmpeg as a subprocess so the GIL is released during the encode and the N clips
    overlap. ``ex.map`` returns results in submission order, so the caller's sidecar/log
    writes stay ordered exactly as serial. Single-clip batches use the serial path (no
    thread overhead).
    """
    if not (clip_config.parallel_clip_extraction and len(filtered) > 1):
        return [_extract_one_clip(seg, ctx) for seg in filtered]
    workers = min(len(filtered), clip_config.max_extraction_workers)
    fn = partial(_extract_one_clip, ctx=ctx)
    with ThreadPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(fn, filtered))
