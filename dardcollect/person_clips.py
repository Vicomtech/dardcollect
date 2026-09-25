"""person_clips.py — entry point for extracting person clips from a video.

The per-video run state + frame loop live in
:mod:`dardcollect.person_clips_run` (split out so this surface stays a thin
wiring dispatcher). ``process_video`` takes a single
:class:`VideoProcessRequest`.
"""

import logging

from dardcollect.frame_reader import frame_iter
from dardcollect.person_clips_run import (
    VideoProcessRequest,
    _drain_run,
    _open_run,
    _step_frame,
)

logger = logging.getLogger(__name__)


def process_video(request: VideoProcessRequest) -> list[dict]:
    """Run detection + tracking on a video and extract all qualifying person clips.

    Resumable via a progress JSON file ({video_stem}_progress.json in output_dir).
    Clips are flushed progressively every ~30s of video to bound memory use.
    Returns an empty list (clip metadata is written to disk and logged by clip_logger).
    """
    logger.info("Processing: %s", request.video_path.name)
    run = _open_run(request)
    if run is None:
        return []

    # Read-ahead decode (producer thread) or inline cap.read — yields (frame_id, frame).
    # The reader thread (if any) is joined on generator close (finally), so it never leaks.
    for frame_id, frame in frame_iter(run.cap, run.start_frame, request.clip_config):
        _step_frame(run, frame_id, frame)

    run.pbar.close()
    _drain_run(run)
    return []
