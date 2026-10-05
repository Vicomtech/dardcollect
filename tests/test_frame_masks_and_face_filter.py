import json
from typing import Any

import cv2
import numpy as np
import pytest

from dardcollect.frames import FrameRequest, _frame_has_face, extract_frames
from dardcollect.pipeline_utils import FACE_LANDMARK_INDICES
from pipeline.generate_face_masks import (
    _generate_one_mask,
    _run_mask_jobs,
)

_CLIP_START_FRAME = 1000  # deliberately non-zero: see test_... below
_FRAMES = 4
_SIZE = 64


def _face_keypoints() -> tuple[list[list[float]], list[float]]:
    """133 keypoints whose face landmarks (23-90) outline a square in the frame."""
    box = [[10.0, 10.0], [50.0, 10.0], [50.0, 50.0], [10.0, 50.0]]
    kpts = [[2.0, 2.0] for _ in range(133)]
    for n, i in enumerate(FACE_LANDMARK_INDICES):
        kpts[i] = box[n % len(box)]
    return kpts, [0.9] * 133


def _write_person_clip(tmp_path):
    """Write a tiny person clip whose frame_data is keyed by ABSOLUTE frame number."""
    video = tmp_path / "clip.mp4"
    fourcc = cv2.VideoWriter.fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(video), fourcc, 10.0, (_SIZE, _SIZE))
    for _ in range(_FRAMES):
        writer.write(np.full((_SIZE, _SIZE, 3), 128, dtype=np.uint8))
    writer.release()

    kpts, scores = _face_keypoints()
    detection = {
        "track_id": 0,
        "bbox": [10.0, 10.0, 50.0, 50.0],
        "score": 0.9,
        "keypoints": kpts,
        "keypoint_scores": scores,
        "face_crop_corners_ofiq": [[10.0, 10.0], [50.0, 10.0], [50.0, 50.0], [10.0, 50.0]],
    }
    sidecar = video.with_suffix(".json")
    sidecar.write_text(
        json.dumps(
            {
                "uuid": "11111111-2222-3333-4444-555555555555",
                "schema_version": "1.0",
                "start_frame": _CLIP_START_FRAME,
                "end_frame": _CLIP_START_FRAME + _FRAMES - 1,
                "frame_data": {str(_CLIP_START_FRAME + i): [detection] for i in range(_FRAMES)},
            }
        ),
        encoding="utf-8",
    )
    return video, sidecar


def test_extract_frames_carries_detections_for_absolute_keyed_clips(tmp_path):
    """Regression: person-clip frame_data is keyed by absolute source-video frame.

    Looking it up with a clip-relative counter misses every entry and silently
    writes `"detections": []` into every frame sidecar, which leaves
    generate_face_masks.py with no landmarks and produces zero masks.
    """
    video, sidecar = _write_person_clip(tmp_path)

    extract_frames(FrameRequest(video, sidecar, tmp_path / "frames", clip_type="person_clip"))

    frame_jsons = sorted((tmp_path / "frames").glob("frame_*.json"))
    assert len(frame_jsons) == _FRAMES
    for path in frame_jsons:
        detections = json.loads(path.read_text(encoding="utf-8"))["detections"]
        assert detections, f"{path.name} lost its detections"
        assert len(detections[0]["keypoints"]) == 133


def test_extracted_frames_yield_non_empty_masks(tmp_path):
    """Source frames get the OFIQ quad mask of what extract_frames wrote."""
    from pipeline.generate_face_masks import _generate_crop_quad_masks

    video, sidecar = _write_person_clip(tmp_path)
    extract_frames(FrameRequest(video, sidecar, tmp_path / "frames", clip_type="person_clip"))

    frames = sorted((tmp_path / "frames").glob("frame_*.png"))
    assert frames, "no frames extracted"
    for frame in frames:
        counts = _generate_crop_quad_masks(frame)
        assert counts["mask"] == 1, f"no quad mask for {frame.name}"
        mask_path = frame.parent / f"{frame.stem}_track000_mask.png"
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        assert mask is not None
        assert mask.max() == 255, "quad mask collapsed to empty"
        assert set(np.unique(mask)).issubset({0, 255}), "mask is not binary"


def test_run_mask_jobs_matches_between_serial_and_threaded(tmp_path):
    """Threading the mask stage must not change what it writes or reports."""
    video, sidecar = _write_person_clip(tmp_path)
    extract_frames(FrameRequest(video, sidecar, tmp_path / "frames", clip_type="person_clip"))
    crops = sorted((tmp_path / "frames").glob("frame_*.png"))
    assert crops, "no frames to mask"

    threaded = _run_mask_jobs(crops, "frames", workers=4)
    assert threaded["mask"] == len(crops)
    assert threaded["no_face"] == 0

    masks = sorted((tmp_path / "frames").glob("*_mask.png"))
    assert len(masks) == len(crops)
    for mask_path in masks:
        written = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        assert written is not None, f"unreadable mask {mask_path.name}"
        assert set(np.unique(written)).issubset({0, 255})

    # Idempotent: a second pass finds every mask already present.
    again = _run_mask_jobs(crops, "frames", workers=4)
    assert again["mask"] == 0
    assert again["noop"] == len(crops)


_ARC_QUAD = [[20.0, 20.0], [44.0, 20.0], [44.0, 44.0], [20.0, 44.0]]


def _write_video_crop(tmp_path, name="c_face_0", size=64, frames=("0", "1", "2")):
    """Fake video-crop mp4 + sidecar; frame '1' has no ArcFace quad."""
    mp4 = tmp_path / f"{name}.mp4"
    mp4.touch()
    entries: dict[str, list[dict[str, Any]]] = {}
    for key in frames:
        entry: dict[str, Any] = {"track_id": 0, "score": 0.9}
        if key != "1":
            entry["face_crop_corners_arcface"] = _ARC_QUAD
        entries[key] = [entry]
    sidecar = tmp_path / f"{name}.json"
    sidecar.write_text(
        json.dumps({"output_size": size, "frame_data": entries}),
        encoding="utf-8",
    )
    return mp4


def test_video_crop_masks_fill_arcface_quad_per_frame(tmp_path):
    """Crop masks are binary images white inside the ArcFace quad, one per frame."""
    mp4 = _write_video_crop(tmp_path)

    counts = _generate_one_mask(mp4)
    assert counts["mask"] == 2
    assert counts["no_face"] == 1

    masks = sorted(tmp_path.glob("*_mask.png"))
    assert [p.name for p in masks] == ["c_face_0_f000000_mask.png", "c_face_0_f000002_mask.png"]
    for mask_path in masks:
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        assert mask is not None, f"unreadable mask {mask_path.name}"
        assert mask.shape == (64, 64)
        assert set(np.unique(mask)).issubset({0, 255})
        assert mask[32, 32] == 255  # inside the quad
        assert mask[0, 0] == 0  # outside the quad

    # Idempotent rerun writes nothing new.
    again = _generate_one_mask(mp4)
    assert again["mask"] == 0
    assert again["noop"] == 2


def test_image_crop_mask_fills_arcface_quad(tmp_path):
    """Image crops get a single ArcFace mask next to the crop."""
    jpg = tmp_path / "img_face_0.jpg"
    jpg.touch()
    (tmp_path / "img_face_0.json").write_text(
        json.dumps(
            {
                "image_path": "x.jpg",
                "output_size": 64,
                "face_crop_corners_arcface": _ARC_QUAD,
            }
        ),
        encoding="utf-8",
    )

    counts = _generate_one_mask(jpg)
    assert counts["mask"] == 1
    mask = cv2.imread(str(tmp_path / "img_face_0_mask.png"), cv2.IMREAD_GRAYSCALE)
    assert mask is not None and set(np.unique(mask)).issubset({0, 255})
    assert mask[32, 32] == 255
    assert mask[63, 63] == 0


def test_frame_extraction_config_reads_workers(tmp_path):
    """`frame_extraction.workers` is parsed and floored at 1 (serial)."""
    from dardcollect.config import FrameExtractionConfig

    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(
        "frame_extraction:\n  input_dir: a\n  output_dir: b\n  workers: 8\n", encoding="utf-8"
    )
    assert FrameExtractionConfig.from_yaml(str(cfg_path)).workers == 8

    cfg_path.write_text(
        "frame_extraction:\n  input_dir: a\n  output_dir: b\n  workers: 0\n", encoding="utf-8"
    )
    assert FrameExtractionConfig.from_yaml(str(cfg_path)).workers == 1

    cfg_path.write_text("frame_extraction:\n  input_dir: a\n  output_dir: b\n", encoding="utf-8")
    assert FrameExtractionConfig.from_yaml(str(cfg_path)).workers == 1


def test_resumed_extraction_keeps_full_manifest(tmp_path):
    """Regression: resuming must not truncate frames_manifest.json.

    The manifest is rebuilt on every call, so frames skipped by the resume check
    have to be re-listed. Otherwise re-running over a complete directory leaves
    `"frames": []` and the frame UUIDs become undiscoverable.
    """
    video, sidecar = _write_person_clip(tmp_path)
    out = tmp_path / "frames"

    extract_frames(FrameRequest(video, sidecar, out, clip_type="person_clip"))
    first = json.loads((out / "frames_manifest.json").read_text(encoding="utf-8"))["frames"]
    assert len(first) == _FRAMES

    extract_frames(
        FrameRequest(video, sidecar, out, clip_type="person_clip")
    )  # resume: all present
    second = json.loads((out / "frames_manifest.json").read_text(encoding="utf-8"))["frames"]

    assert len(second) == _FRAMES, "resuming truncated the manifest"
    assert [f["uuid"] for f in second] == [f["uuid"] for f in first], "frame UUIDs changed"


def test_frame_has_face_requires_keypoints_list():
    assert _frame_has_face([]) is False
    assert _frame_has_face([{"score": 0.9}]) is False
    assert _frame_has_face([{"keypoints": []}]) is False
    assert _frame_has_face([{"keypoints": [[1.0, 2.0]]}]) is True


def test_frame_extraction_config_reads_min_free_disk_gb(tmp_path):
    """`frame_extraction.min_free_disk_gb` is parsed; default matches the other stages."""
    from dardcollect.config import FrameExtractionConfig

    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(
        "frame_extraction:\n  input_dir: a\n  output_dir: b\n  min_free_disk_gb: 20.0\n",
        encoding="utf-8",
    )
    assert FrameExtractionConfig.from_yaml(str(cfg_path)).min_free_disk_gb == 20.0

    cfg_path.write_text("frame_extraction:\n  input_dir: a\n  output_dir: b\n", encoding="utf-8")
    assert FrameExtractionConfig.from_yaml(str(cfg_path)).min_free_disk_gb == 2.0


def test_frame_stage_aborts_when_disk_is_low(tmp_path, monkeypatch):
    """Regression: the frames stage must refuse to write into a full filesystem.

    It is the pipeline's heaviest writer (~190 MB of lossless PNG per filtered
    face crop), and it used to have no disk guard at all — a full pass would
    simply fill the volume, which on a shared quota takes other users down too.
    """
    import pipeline.extract_frames_from_videos as stage

    video, sidecar = _write_person_clip(tmp_path)
    inp = tmp_path / "in"
    inp.mkdir()
    video.rename(inp / video.name)
    sidecar.rename(inp / sidecar.name)

    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(
        f"frame_extraction:\n"
        f"  input_dir: {inp}\n"
        f"  output_dir: {tmp_path / 'out'}\n"
        f"  workers: 1\n"
        f"  min_free_disk_gb: 1000000.0\n",
        encoding="utf-8",
    )

    extracted = []
    monkeypatch.setattr(stage, "extract_frames", lambda *a, **k: extracted.append(a))

    with pytest.raises(SystemExit) as excinfo:
        stage.main(str(cfg_path))

    assert excinfo.value.code == 1
    assert extracted == [], "frames were written despite the disk guard firing"
