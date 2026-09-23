"""Tests for film-level parallelism in the clips stage.

Covers the two concurrency-critical pieces of person_extraction.workers > 1:
1. ExtractionLogger.log_extraction is thread-safe — concurrent film workers logging
   into the single clips_extraction.csv must not double-write the header or lose/
   corrupt rows (mirrors the FramesExtractionLogger write-lock contract).
2. ClipExtractionConfig.workers parses (default 1 = serial, unchanged behaviour).

The header-race test forces the interleaving deterministically (a real thread race
is GIL-masked and would pass even without the lock — the bug-5 lesson from the frame
masks handoff): it widens the check-then-write window by making writeheader block on
a barrier, so two workers both enter the "header not written" branch unless the lock
serializes them. It is verified to fail when ``_write_lock`` is removed.
"""

import csv
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from dardcollect.config import ClipExtractionConfig
from dardcollect.extraction_logger import ClipRecord, ExtractionLogger


def _log_one(logger: ExtractionLogger, i: int) -> None:
    logger.log_extraction(
        ClipRecord(
            source_video=f"film_{i}.mp4",
            fps=25.0,
            start_frame=i,
            end_frame=i + 10,
            start_seconds=float(i),
            duration_seconds=0.4,
            max_persons_per_frame=1,
            detector_model="yolox",
            detector_confidence=0.9,
            output_path=f"/out/clip_{i}.mp4",
        )
    )


def test_concurrent_logging_no_duplicate_header(tmp_path: Path, monkeypatch) -> None:
    """With the write-lock, only ONE worker can be inside the header-write window at a
    time, so exactly one header is written even when two workers start on an empty file."""
    logger = ExtractionLogger(output_dir=str(tmp_path), downloads_csv_path=None)

    # Force both threads to overlap inside the check-then-write header window. A barrier
    # of 2 makes writeheader block until two workers reach it simultaneously; the lock
    # (if present) prevents the second from ever getting there while the first holds it,
    # so the barrier times out for the second and only one header is written.
    real_writeheader = csv.DictWriter.writeheader
    barrier = threading.Barrier(2, timeout=1.0)

    def slow_writeheader(self):
        try:
            barrier.wait()
        except threading.BrokenBarrierError:
            pass
        return real_writeheader(self)

    monkeypatch.setattr(csv.DictWriter, "writeheader", slow_writeheader)

    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(lambda i: _log_one(logger, i), range(2)))

    csv_path = tmp_path / "clips_extraction.csv"
    with open(csv_path, encoding="utf-8") as f:
        header_lines = [ln for ln in f if ln.startswith("uuid,")]
    assert len(header_lines) == 1, f"expected 1 header, got {len(header_lines)}"


def test_concurrent_logging_all_rows_present(tmp_path: Path) -> None:
    logger = ExtractionLogger(output_dir=str(tmp_path), downloads_csv_path=None)
    n = 200
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(lambda i: _log_one(logger, i), range(n)))

    with open(tmp_path / "clips_extraction.csv", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == n
    assert all(r.get("uuid") for r in rows)
    assert {r["source_video"] for r in rows} == {f"film_{i}.mp4" for i in range(n)}


def test_logger_has_write_lock(tmp_path: Path) -> None:
    logger = ExtractionLogger(output_dir=str(tmp_path), downloads_csv_path=None)
    assert hasattr(logger, "_write_lock")


def test_clip_config_workers_default_is_one() -> None:
    fields = {f.name: f for f in ClipExtractionConfig.__dataclass_fields__.values()}
    assert "workers" in fields
    assert fields["workers"].default == 1


def test_clip_config_workers_parses(tmp_path: Path) -> None:
    cfg = tmp_path / "c.yaml"
    cfg.write_text(
        "person_extraction:\n"
        "  input_dir: in\n"
        "  output_clips_dir: out\n"
        "  min_clip_duration_seconds: 2.0\n"
        "  max_clip_duration_seconds: 60.0\n"
        "  min_consecutive_frames: 10\n"
        "  merge_gap_frames: 10\n"
        "  require_face_visibility: true\n"
        "  min_face_size_percent: 10.0\n"
        "  min_face_visible_frames: 15\n"
        "  workers: 4\n",
        encoding="utf-8",
    )
    c = ClipExtractionConfig.from_yaml(str(cfg))
    assert c.workers == 4


def test_clip_config_workers_floored_to_one(tmp_path: Path) -> None:
    cfg = tmp_path / "c.yaml"
    cfg.write_text(
        "person_extraction:\n"
        "  input_dir: in\n"
        "  output_clips_dir: out\n"
        "  min_clip_duration_seconds: 2.0\n"
        "  max_clip_duration_seconds: 60.0\n"
        "  min_consecutive_frames: 10\n"
        "  merge_gap_frames: 10\n"
        "  require_face_visibility: true\n"
        "  min_face_size_percent: 10.0\n"
        "  min_face_visible_frames: 15\n"
        "  workers: 0\n",
        encoding="utf-8",
    )
    c = ClipExtractionConfig.from_yaml(str(cfg))
    assert c.workers == 1
