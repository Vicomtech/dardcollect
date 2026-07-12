import json

from dardcollect.frames import _frame_has_face
from pipeline.generate_face_masks import _load_keypoints


def test_load_keypoints_reads_top_level_format(tmp_path):
    sidecar = tmp_path / "crop.json"
    sidecar.write_text(
        json.dumps(
            {
                "keypoints": [[float(i), float(i)] for i in range(133)],
                "keypoint_scores": [0.9] * 133,
            }
        ),
        encoding="utf-8",
    )

    out = _load_keypoints(sidecar)
    assert out is not None
    kpts, scores = out
    assert kpts.shape == (133, 2)
    assert scores.shape == (133,)


def test_load_keypoints_reads_detection_format(tmp_path):
    sidecar = tmp_path / "frame.json"
    sidecar.write_text(
        json.dumps(
            {
                "detections": [
                    {
                        "score": 0.7,
                        "keypoints": [[float(i), float(i)] for i in range(133)],
                        "keypoint_scores": [0.8] * 133,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    out = _load_keypoints(sidecar)
    assert out is not None
    kpts, scores = out
    assert kpts.shape == (133, 2)
    assert scores.shape == (133,)


def test_frame_has_face_requires_keypoints_list():
    assert _frame_has_face([]) is False
    assert _frame_has_face([{"score": 0.9}]) is False
    assert _frame_has_face([{"keypoints": []}]) is False
    assert _frame_has_face([{"keypoints": [[1.0, 2.0]]}]) is True
