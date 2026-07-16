"""Read-ahead frame decoder for the clips stage (GPU/CPU overlap).

``cv2.VideoCapture.read()`` on webm (VP8/VP9) sources is CPU-bound and serial; running it in
the main thread starves the GPU between inference calls. ``_FrameReader`` decodes frames ahead
in a daemon producer thread into a bounded queue so the main loop pops decoded frames while
the producer keeps decoding — overlapping CPU decode with GPU inference.
"""

from __future__ import annotations

import logging
import queue
import threading

import numpy as np

from dardcollect.config import ClipExtractionConfig

logger = logging.getLogger(__name__)


class _FrameReader:
    """Producer thread that decodes frames ahead into a bounded queue (read-ahead).

    Overlaps CPU decode with GPU inference in the main thread: the consumer pops decoded
    frames while the producer reads ahead. ``cap`` must already be seeked to ``start_frame``
    (done by ``load_resume_start``) before ``start()``. The queue is bounded so the producer
    throttles when the consumer falls behind (bounds CPU RAM). On a producer exception the
    error is stored and re-raised by the consumer on end (fail-loud). Sentinel-terminated.
    """

    _END = object()

    def __init__(self, cap, start_frame: int, maxsize: int) -> None:
        self._cap = cap
        self._next_id = start_frame
        self._q: queue.Queue = queue.Queue(maxsize=maxsize)
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._exc: BaseException | None = None

    def start(self) -> None:
        self._thread.start()

    def _run(self) -> None:
        try:
            while True:
                ret, frame = self._cap.read()
                if not ret:
                    break
                self._q.put((self._next_id, frame))
                self._next_id += 1
        except Exception as exc:  # fail-loud: surface to consumer
            self._exc = exc
        self._q.put(self._END)

    def __iter__(self):
        return self

    def __next__(self) -> tuple[int, np.ndarray]:
        item = self._q.get()
        if item is self._END:
            if self._exc is not None:
                raise self._exc
            raise StopIteration
        return item  # (frame_id, frame)

    def join(self) -> None:
        self._thread.join()


def frame_iter(cap, start_frame: int, clip_config: ClipExtractionConfig):
    """Yield ``(frame_id, frame)`` for the clip loop, read-ahead or inline.

    Read-ahead spawns a ``_FrameReader`` producer thread (joined on generator close so the
    thread never leaks across sources or on early return/exception). Inline mode is a plain
    generator over ``cap.read()`` (unchanged behavior when ``readahead_decode`` is off).
    """
    if not clip_config.readahead_decode:
        frame_id = start_frame
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            yield frame_id, frame
            frame_id += 1
        return
    reader = _FrameReader(cap, start_frame, clip_config.readahead_queue_frames)
    reader.start()
    try:
        while True:
            try:
                item = next(reader)
            except StopIteration:
                break  # reader exhausted — do NOT let StopIteration leak (PEP 479)
            yield item
    finally:
        reader.join()
