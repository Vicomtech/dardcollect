"""Regression tests for the transcription scan's memory contract.

`scan_for_untranscribed_clips` used to append each clip's fully parsed sidecar to
its result list. Person-clip sidecars carry `frame_data` (133 keypoints + scores
per detection per frame), so they run from ~1 MB to 15 MB on disk and several
times that in memory. Across a full run that grew the transcription stage past
50 GB during the scan alone — before a single clip was transcribed — and the OOM
killer took the whole pipeline down twice.
"""

import json
import pickle
from pathlib import Path

from dardcollect.audio import scan_for_untranscribed_clips

_UUID = "11111111-2222-3333-4444-555555555555"


def _write_clip(tmp_path: Path, name: str, frames: int = 200) -> Path:
    """A clip whose sidecar carries a realistically bulky `frame_data` block."""
    detection = {
        "track_id": 0,
        "bbox": [10.0, 10.0, 50.0, 50.0],
        "score": 0.9,
        "keypoints": [[float(i), float(i)] for i in range(133)],
        "keypoint_scores": [0.9] * 133,
    }
    (tmp_path / f"{name}.mp4").write_bytes(b"\x00\x00\x00\x18ftyp")
    (tmp_path / f"{name}.json").write_text(
        json.dumps(
            {
                "uuid": _UUID,
                "start_frame": 0,
                "frame_data": {str(i): [detection] for i in range(frames)},
            }
        ),
        encoding="utf-8",
    )
    return tmp_path / f"{name}.json"


def test_scan_returns_parent_uuid_not_the_whole_sidecar(tmp_path):
    """The 4th tuple element must be the UUID, never the parsed sidecar dict."""
    _write_clip(tmp_path, "clip")

    (found,) = scan_for_untranscribed_clips(tmp_path)
    mp4_path, json_path, trans_path, parent_uuid = found

    assert mp4_path.name == "clip.mp4"
    assert json_path.name == "clip.json"
    assert trans_path.name == "clip.transcription.json"
    assert parent_uuid == _UUID
    assert not isinstance(parent_uuid, dict), "scan is retaining the parsed sidecar again"


def test_scan_result_does_not_grow_with_sidecar_size(tmp_path):
    """Result size must be flat in sidecar size — that is the whole point.

    A 20x bulkier `frame_data` must not make the retained result meaningfully
    bigger; if it does, the scan is holding frame data alive again.
    """
    # Equal-length names: the retained tuples hold paths, so a longer directory
    # name would shift the measurement on its own.
    small_dir = tmp_path / "aa"
    big_dir = tmp_path / "bb"
    small_dir.mkdir()
    big_dir.mkdir()
    _write_clip(small_dir, "clip", frames=10)
    _write_clip(big_dir, "clip", frames=200)

    assert (big_dir / "clip.json").stat().st_size > 15 * (small_dir / "clip.json").stat().st_size

    small = scan_for_untranscribed_clips(small_dir)
    big = scan_for_untranscribed_clips(big_dir)

    # pickle walks the object graph, so this measures what is actually retained.
    # sys.getsizeof would not: it reports a dict's shallow size, which is
    # identical whether or not it holds megabytes of nested frame_data.
    assert len(pickle.dumps(big)) == len(pickle.dumps(small))


def test_scan_skips_clips_that_already_have_a_transcription(tmp_path):
    """Resumability: a clip with its transcription sidecar is not rescanned."""
    _write_clip(tmp_path, "done", frames=5)
    (tmp_path / "done.transcription.json").write_text("{}", encoding="utf-8")
    _write_clip(tmp_path, "pending", frames=5)

    names = {row[0].name for row in scan_for_untranscribed_clips(tmp_path)}
    assert names == {"pending.mp4"}

    names_overwrite = {
        row[0].name for row in scan_for_untranscribed_clips(tmp_path, overwrite=True)
    }
    assert names_overwrite == {"done.mp4", "pending.mp4"}


def test_scan_tolerates_a_sidecar_without_uuid(tmp_path):
    """A malformed sidecar yields None, which the caller reports and skips."""
    (tmp_path / "clip.mp4").write_bytes(b"\x00\x00\x00\x18ftyp")
    (tmp_path / "clip.json").write_text(json.dumps({"start_frame": 0}), encoding="utf-8")

    (found,) = scan_for_untranscribed_clips(tmp_path)
    assert found[3] is None
