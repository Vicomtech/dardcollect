"""Tests for the video pre-processing masks feature.

Covers the contract in docs/DESIGN_video_frame_masks.md: N consecutive frames pulled from
the ORIGINAL video at timestamps where a person was already detected, plus one filled
mask per detected identity, derived from that identity's OFIQ face crop.
"""

import json
from pathlib import Path

import cv2
import numpy as np

from dardcollect.source_frames import (
    extract_source_frames_for_clip,
    select_consecutive_detected_frames,
)
from pipeline.generate_face_masks import (
    _detections_with_faces,
    _generate_crop_quad_masks,
    _ofiq_quad,
    _quad_mask,
)

_START = 10  # non-zero on purpose: keys are ABSOLUTE source frames
_W, _H = 96, 64


def _detection(track_id: int = 0, quad=None, with_face: bool = True) -> dict:
    """A detection. `with_face=False` mimics a subject whose face crop was never cut."""
    scores = [0.9] * 133 if with_face else [0.0] * 133
    det = {
        "track_id": track_id,
        "bbox": [10.0, 8.0, 40.0, 36.0],
        "score": 0.9,
        "keypoints": [[float(i), float(i)] for i in range(133)],
        "keypoint_scores": scores,
    }
    if with_face:
        # Rotated on purpose: OFIQ levels the eyes, so real crops are never axis-aligned.
        det["face_crop_corners_ofiq"] = quad or [
            [12.0, 4.0],
            [38.0, 10.0],
            [32.0, 34.0],
            [6.0, 28.0],
        ]
    return det


def _write_source_and_clip(tmp_path, *, n_source_frames=20, detected_frames=None):
    """A source video plus a clip sidecar keyed by ABSOLUTE source frame numbers."""
    videos = tmp_path / "videos" / "eng"
    videos.mkdir(parents=True)
    source = videos / "Some Film (1960).mp4"
    writer = cv2.VideoWriter(str(source), cv2.VideoWriter.fourcc(*"mp4v"), 25.0, (_W, _H))
    for i in range(n_source_frames):
        writer.write(np.full((_H, _W, 3), i * 5 % 256, dtype=np.uint8))
    writer.release()

    clips = tmp_path / "clips" / "eng"
    clips.mkdir(parents=True)
    (clips / "clip.mp4").write_bytes(b"\x00\x00\x00\x18ftyp")
    sidecar = clips / "clip.json"

    if detected_frames is None:
        detected_frames = {str(_START + i): [_detection()] for i in range(6)}
    sidecar.write_text(
        json.dumps(
            {
                "uuid": "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee",
                "source_video": str(source),
                "start_frame": _START,
                "end_frame": _START + 10,
                "frame_data": detected_frames,
            }
        ),
        encoding="utf-8",
    )
    return sidecar


def test_selection_picks_a_contiguous_run_of_n():
    frame_data = {str(_START + i): [_detection()] for i in range(6)}
    assert select_consecutive_detected_frames(frame_data, 5) == [_START + i for i in range(5)]


def test_selection_skips_gaps_and_faceless_frames():
    """A run must be unbroken in the source video AND face-detected throughout."""
    frame_data = {
        "100": [_detection()],
        "101": [_detection(with_face=False)],  # breaks the run
        "102": [_detection()],
        "103": [_detection()],
        "107": [_detection()],  # gap: 104-106 missing
        "108": [_detection()],
        "109": [_detection()],
    }
    assert select_consecutive_detected_frames(frame_data, 3) == [107, 108, 109]


def test_selection_returns_empty_when_no_run_qualifies():
    frame_data = {"100": [_detection()], "105": [_detection()]}
    assert select_consecutive_detected_frames(frame_data, 3) == []


def test_frames_come_from_the_source_video_at_absolute_numbers(tmp_path):
    """Regression: frames must be the ORIGINAL video's, keyed by absolute frame."""
    sidecar = _write_source_and_clip(tmp_path)
    out = tmp_path / "frames"

    written, had_run = extract_source_frames_for_clip(sidecar, out, frames_per_clip=5)
    assert (written, had_run) == (5, True)

    # Grouped by source video, not by clip, so overlapping clips converge.
    frame_dir = out / "eng" / "Some Film (1960)"
    names = sorted(p.name for p in frame_dir.glob("frame_*.png"))
    assert names == [f"frame_{_START + i:06d}.png" for i in range(5)]

    meta = json.loads((frame_dir / f"frame_{_START:06d}.json").read_text(encoding="utf-8"))
    assert meta["frame_number"] == _START
    assert meta["detections"], "detections were dropped"
    assert meta["source_video"].endswith("Some Film (1960).mp4")


def test_extraction_is_idempotent_across_overlapping_clips(tmp_path):
    """A second pass writes nothing: output is keyed by source video + absolute frame."""
    sidecar = _write_source_and_clip(tmp_path)
    out = tmp_path / "frames"

    assert extract_source_frames_for_clip(sidecar, out, frames_per_clip=5) == (5, True)
    # Second pass: nothing written, but the run still qualified. Reporting these as
    # "no qualifying run" made a fully resumed pass look like a total failure.
    assert extract_source_frames_for_clip(sidecar, out, frames_per_clip=5) == (0, True)


def test_clip_without_a_qualifying_run_writes_nothing(tmp_path):
    sidecar = _write_source_and_clip(
        tmp_path, detected_frames={str(_START): [_detection(with_face=False)]}
    )
    out = tmp_path / "frames"
    assert extract_source_frames_for_clip(sidecar, out, frames_per_clip=5) == (0, False)
    assert not out.exists() or not list(out.rglob("*.png"))


_QUAD = np.array([[12, 4], [38, 10], [32, 34], [6, 28]], np.int32)


def test_quad_mask_rotated_is_not_axis_aligned():
    """`ofiq_crop_quad`: the crop quad itself, rotated, matching the crop video."""
    mask = _quad_mask(_QUAD, _H, _W, axis_aligned=False)
    assert set(np.unique(mask).tolist()) == {0, 255}

    ys, _ = np.where(mask == 255)
    widths = {int((row == 255).sum()) for row in mask[ys.min() : ys.max() + 1]}
    assert len(widths) > 1, "a rotated quad must not produce uniform row widths"

    # Inside the quad, but outside it at the corner of its upright box.
    assert mask[19, 22] == 255
    assert mask[4, 6] == 0


def test_quad_mask_axis_aligned_is_the_upright_box():
    """`ofiq_crop_bbox`: the upright bounding box of that same quad."""
    mask = _quad_mask(_QUAD, _H, _W, axis_aligned=True)
    assert set(np.unique(mask).tolist()) == {0, 255}

    ys, xs = np.where(mask == 255)
    assert (xs.min(), xs.max()) == (6, 38), "must span the quad's extreme x pixels"
    assert (ys.min(), ys.max()) == (4, 34), "must span the quad's extreme y pixels"
    widths = {int((row == 255).sum()) for row in mask[ys.min() : ys.max() + 1]}
    assert widths == {33}, "the upright box must have identical row widths"

    # The upright box always contains the rotated quad.
    rotated = _quad_mask(_QUAD, _H, _W, axis_aligned=False)
    assert np.all(mask[rotated > 0] == 255)
    assert (mask > 0).sum() > (rotated > 0).sum(), "upright box takes in extra background"


def test_quad_mask_clips_to_the_frame():
    huge = np.array([[-99, -99], [_W + 99, -99], [_W + 99, _H + 99], [-99, _H + 99]], np.int32)
    assert _quad_mask(huge, _H, _W).min() == 255
    assert _quad_mask(huge, _H, _W, axis_aligned=True).min() == 255


def test_ofiq_quad_rejects_malformed_corners():
    assert _ofiq_quad({"face_crop_corners_ofiq": [[0, 0], [1, 1]]}) is None
    assert _ofiq_quad({"face_crop_corners_ofiq": "nope"}) is None
    assert _ofiq_quad({}) is None, "no crop corners means no mask"
    assert _ofiq_quad(_detection()) is not None


def test_one_mask_per_detected_identity(tmp_path):
    """Each tracked identity gets its own file, named with its track id."""
    frame = tmp_path / "frame_003486.png"
    cv2.imwrite(str(frame), np.zeros((_H, _W, 3), dtype=np.uint8))
    frame.with_suffix(".json").write_text(
        json.dumps(
            {
                "detections": [
                    _detection(track_id=7, quad=[[2, 2], [20, 5], [18, 20], [1, 17]]),
                    _detection(track_id=12, quad=[[40, 10], [80, 16], [76, 50], [37, 44]]),
                ]
            }
        ),
        encoding="utf-8",
    )

    assert _generate_crop_quad_masks(frame) == "mask"
    masks = sorted(p.name for p in tmp_path.glob("*_mask.png"))
    assert masks == ["frame_003486_track007_mask.png", "frame_003486_track012_mask.png"]

    written = cv2.imread(str(tmp_path / masks[0]), cv2.IMREAD_GRAYSCALE)
    assert written is not None
    assert set(np.unique(written).tolist()) == {0, 255}

    # Idempotent: both masks already exist.
    assert _generate_crop_quad_masks(frame) == "noop"


def test_no_mask_without_a_face_crop(tmp_path):
    """No OFIQ crop corners means no mask — the request's "if a face is detected" branch."""
    frame = tmp_path / "frame_000001.png"
    cv2.imwrite(str(frame), np.zeros((_H, _W, 3), dtype=np.uint8))
    frame.with_suffix(".json").write_text(
        json.dumps({"detections": [_detection(track_id=1, with_face=False)]}), encoding="utf-8"
    )

    assert _generate_crop_quad_masks(frame) == "no_face"
    assert not list(tmp_path.glob("*_mask.png"))


def test_detection_without_crop_corners_is_rejected(tmp_path):
    sidecar = tmp_path / "frame.json"
    det = _detection()
    del det["face_crop_corners_ofiq"]
    sidecar.write_text(json.dumps({"detections": [det]}), encoding="utf-8")
    assert _detections_with_faces(sidecar) == []


def test_sidecar_discovery_keeps_clips_whose_film_title_has_a_dot(tmp_path):
    """Regression: `Path.suffixes` splits on every dot, not just the extension.

    Filtering clip sidecars with `len(p.suffixes) == 1` silently dropped every clip of
    any film with a dot in its title — 2,694 of 13,406 clips (20%) on the 2026-08-04 run.
    The `.mp4` sibling test is what separates a clip sidecar from a derived one.
    """
    clips = tmp_path / "eng"
    clips.mkdir()
    for stem in ("Esther and the King (1960).ia_04m33s-04m41s", "Space Men (1960)_10m21s-10m26s"):
        (clips / f"{stem}.mp4").write_bytes(b"\x00")
        (clips / f"{stem}.json").write_text("{}", encoding="utf-8")
    # Derived sidecars: no .mp4 sibling, so they must not be picked up.
    (clips / "Space Men (1960)_10m21s-10m26s.transcription.json").write_text("{}", encoding="utf-8")
    (clips / "Space Men (1960)_10m21s-10m26s.quality.json").write_text("{}", encoding="utf-8")

    found = sorted(p.name for p in tmp_path.rglob("*.json") if p.with_suffix(".mp4").exists())
    assert found == [
        "Esther and the King (1960).ia_04m33s-04m41s.json",
        "Space Men (1960)_10m21s-10m26s.json",
    ]

    dotted = Path("Esther and the King (1960).ia_04m33s-04m41s.json")
    assert len(dotted.suffixes) == 2, "the trap this test guards against"
