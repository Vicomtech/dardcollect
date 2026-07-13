"""CPU-only smoke tests for viewer indexing/server helpers.

These tests validate core viewer discovery behavior without launching a browser:
- index builders include valid artifacts and ignore auxiliary sidecars,
- video/image entries require the expected media pairings,
- data_index loading returns data_root + proxy mode.
"""

from pathlib import Path

import viewer.index_data as viewer_index
import viewer.serve as viewer_serve


def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{}", encoding="utf-8")


def test_scan_image_detections_dir_filters_aux_sidecars(tmp_path):
    _touch(tmp_path / "img_a.json")
    _touch(tmp_path / "img_a.magface.json")
    _touch(tmp_path / "img_a.ofiq_attr.json")
    _touch(tmp_path / "img_a.quality.json")
    _touch(tmp_path / "img_a.transcription.json")

    items = viewer_index._scan_image_detections_dir(tmp_path, "extracted_image_detections")

    assert items == [
        {
            "type": "image_detection",
            "json_path": "data_link/extracted_image_detections/img_a.json",
        }
    ]


def test_scan_image_face_crops_requires_jpg_and_collects_optional_sidecars(tmp_path):
    _touch(tmp_path / "crop_ok.json")
    _touch(tmp_path / "crop_ok.magface.json")
    _touch(tmp_path / "crop_ok.ofiq_attr.json")
    _touch(tmp_path / "crop_skip.json")

    (tmp_path / "crop_ok.jpg").write_bytes(b"\xff\xd8\xff")

    items = viewer_index._scan_image_face_crops_dir(tmp_path, "image_face_crops")

    assert len(items) == 1
    item = items[0]
    assert item["type"] == "image_face_crop"
    assert item["json_path"] == "data_link/image_face_crops/crop_ok.json"
    assert item["image_path"] == "data_link/image_face_crops/crop_ok.jpg"
    assert item["magface_path"] == "data_link/image_face_crops/crop_ok.magface.json"
    assert item["ofiq_attr_path"] == "data_link/image_face_crops/crop_ok.ofiq_attr.json"


def test_scan_video_dir_requires_mp4_and_collects_aux_sidecars(tmp_path):
    _touch(tmp_path / "clip_ok.json")
    _touch(tmp_path / "clip_ok.magface.json")
    _touch(tmp_path / "clip_ok.ofiq_attr.json")
    _touch(tmp_path / "clip_ok.transcription.json")
    _touch(tmp_path / "clip_skip.json")

    (tmp_path / "clip_ok.mp4").write_bytes(b"\x00\x00\x00\x18ftyp")

    items = viewer_index._scan_video_dir(tmp_path, "video_face_crops")

    assert len(items) == 1
    item = items[0]
    assert item["json_path"] == "data_link/video_face_crops/clip_ok.json"
    assert item["video_path"] == "data_link/video_face_crops/clip_ok.mp4"
    assert item["magface_path"] == "data_link/video_face_crops/clip_ok.magface.json"
    assert item["ofiq_attr_path"] == "data_link/video_face_crops/clip_ok.ofiq_attr.json"
    assert item["transcription_path"] == "data_link/video_face_crops/clip_ok.transcription.json"


def test_load_data_root_reads_proxy_flag(monkeypatch, tmp_path):
    index_file = tmp_path / "data_index.json"
    index_file.write_text(
        '{"data_root": "C:/Data/Root", "use_server_proxy": true}',
        encoding="utf-8",
    )

    monkeypatch.setattr(viewer_serve, "INDEX_FILE", index_file)

    root, use_proxy = viewer_serve._load_data_root()

    assert root is not None
    assert root.as_posix() == "C:/Data/Root"
    assert use_proxy is True


def test_resolve_uses_repo_root_not_config_dir():
    """Viewer config paths resolve relative to the repo root (where stage scripts
    write outputs), NOT relative to the config file's directory.

    Regression guard for the configs/ move: the viewer's _resolve used
    CONFIG_PATH.parent, so paths resolved to configs/DARD_test/... (broken)
    instead of repo-root/DARD_test/... (where outputs live)."""
    from viewer.index_data import REPO_ROOT, _resolve

    assert (
        _resolve("DARD_test/video_face_crops")
        == (REPO_ROOT / "DARD_test" / "video_face_crops").resolve()
    )


def test_load_cfg_resolves_root_templates(tmp_path, monkeypatch):
    """_load_cfg applies {root}/{output_root} template resolution.

    Regression guard: the viewer used to yaml.safe_load only, leaving
    "{root}/extracted_person_clips" literal, so no configured dir ever resolved
    and index_data raised 'None of the configured output directories exist
    yet.' on any templated config (the default config.archive_all.yaml)."""
    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        "root: 'DARD'\nperson_extraction:\n  output_clips_dir: '{root}/extracted_person_clips'\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(viewer_index, "CONFIG_PATH", cfg)
    loaded = viewer_index._load_cfg()
    assert loaded["person_extraction"]["output_clips_dir"] == "DARD/extracted_person_clips"
