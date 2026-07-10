"""Unit tests for `dardcollect.fair` — FAIR metadata + JSON Schema validation.

Pure CPU tests: no GPU, no models, no network. They exercise the in-process
provenance helpers (`generate_uuid`, `add_fair_metadata`, `reorganize_for_fair`)
and the JSON Schema loading/validation against the ratified sidecar schemas in
`schemas/`.

Note: importing `dardcollect.fair` triggers `dardcollect/__init__.py`, which
preloads NVIDIA libs (a no-op on CPU-only installs) and imports the heavy
detection/audio submodules. Collection is therefore ~a few seconds, but the
tests themselves are CPU-only and deterministic.
"""

import copy

import jsonschema
import pytest

from dardcollect.fair import (
    SCHEMA_VERSIONS,
    add_fair_metadata,
    generate_uuid,
    load_schema,
    reorganize_for_fair,
    validate_against_schema,
)

UUID_V4_PATTERN = "^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$"

ALL_SCHEMA_TYPES = [
    "person_clip",
    "face_crop",
    "quality_annotation",
    "transcription",
    "document",
]


# ── generate_uuid ─────────────────────────────────────────────────────────────


def test_generate_uuid_matches_v4_pattern():
    import re

    assert re.match(UUID_V4_PATTERN, generate_uuid()) is not None


def test_generate_uuid_is_unique():
    assert generate_uuid() != generate_uuid()


def test_generate_uuid_is_version_4():
    # 14th char is the version nibble; v4 → '4'.
    assert generate_uuid()[14] == "4"


# ── add_fair_metadata ──────────────────────────────────────────────────────────


def test_add_fair_metadata_injects_uuid_and_schema_version():
    data = add_fair_metadata({}, schema_type="person_clip")
    assert "uuid" in data
    assert data["schema_version"] == SCHEMA_VERSIONS["person_clip"] == "1.0"


def test_add_fair_metadata_preserves_existing_uuid():
    data = add_fair_metadata({"uuid": "preset-uuid"}, schema_type="person_clip")
    assert data["uuid"] == "preset-uuid"


def test_add_fair_metadata_unknown_schema_type_defaults_to_1_0():
    data = add_fair_metadata({}, schema_type="not_a_real_type")
    assert data["schema_version"] == "1.0"


def test_add_fair_metadata_face_crop_links_parent_clip():
    data = add_fair_metadata(
        {},
        schema_type="face_crop",
        parent_uuid="clip-uuid",
        parent_file="clip.mp4",
    )
    assert data["parent_clip"] == {"uuid": "clip-uuid", "file": "clip.mp4"}


def test_add_fair_metadata_quality_links_parent_crop():
    data = add_fair_metadata(
        {},
        schema_type="quality_annotation",
        parent_uuid="crop-uuid",
        parent_file="crop.mp4",
    )
    assert data["parent_crop"] == {"uuid": "crop-uuid", "file": "crop.mp4"}


def test_add_fair_metadata_transcription_links_parent_clip():
    data = add_fair_metadata(
        {},
        schema_type="transcription",
        parent_uuid="clip-uuid",
        parent_file="clip.mp4",
    )
    assert data["parent_clip"] == {"uuid": "clip-uuid", "file": "clip.mp4"}


def test_add_fair_metadata_no_parent_when_none_given():
    data = add_fair_metadata({}, schema_type="face_crop")
    assert "parent_clip" not in data
    assert "parent_crop" not in data


def test_add_fair_metadata_source_and_license_from_archive_org():
    data = add_fair_metadata(
        {},
        schema_type="person_clip",
        archive_org_id="titanic_1912",
        archive_org_url="https://archive.org/details/titanic_1912",
    )
    assert data["source"]["archive_org_id"] == "titanic_1912"
    assert data["source"]["archive_org_url"] == "https://archive.org/details/titanic_1912"
    assert data["source"]["license"] == "public-domain"


def test_add_fair_metadata_returns_same_dict_object():
    data = {}
    assert add_fair_metadata(data, schema_type="person_clip") is data


# ── reorganize_for_fair ───────────────────────────────────────────────────────
# NOTE: reorganize_for_fair pops keys from its input (its docstring claims
# non-mutation, but the implementation mutates). Tests pass a copy so the
# ordering/completeness contract is checked independently of that divergence.


def test_reorganize_for_fair_puts_fair_fields_first():
    data = {
        "uuid": "u",
        "schema_version": "1.0",
        "source": {"archive_org_id": "x"},
        "parent_clip": {"uuid": "p"},
        "payload_field": 1,
    }
    result = reorganize_for_fair(dict(data), schema_type="person_clip")
    keys = list(result.keys())
    assert keys[:4] == ["uuid", "schema_version", "source", "parent_clip"]
    assert keys[-1] == "payload_field"


def test_reorganize_for_fair_preserves_all_keys():
    data = {"uuid": "u", "schema_version": "1.0", "foo": 1, "bar": 2}
    result = reorganize_for_fair(dict(data), schema_type="person_clip")
    assert set(result.keys()) == {"uuid", "schema_version", "foo", "bar"}


def test_reorganize_for_fair_no_fair_fields_passes_through():
    data = {"foo": 1, "bar": 2}
    result = reorganize_for_fair(copy.deepcopy(data), schema_type="person_clip")
    assert result == {"foo": 1, "bar": 2}


# ── load_schema / validate_against_schema ─────────────────────────────────────


@pytest.mark.parametrize("schema_type", ALL_SCHEMA_TYPES)
def test_load_schema_returns_dict_for_each_type(schema_type):
    schema = load_schema(schema_type)
    assert isinstance(schema, dict)
    assert schema.get("type") == "object"


def test_load_schema_missing_type_raises_file_not_found():
    with pytest.raises(FileNotFoundError):
        load_schema("does_not_exist")


def test_validate_against_schema_accepts_valid_person_clip():
    data = {
        "uuid": generate_uuid(),
        "schema_version": "1.0",
        "source_video": "videos/fingerDance1956.mp4",
        "start_frame": 0,
        "end_frame": 90,
        "duration_seconds": 3.0,
    }
    assert validate_against_schema(data, "person_clip") is True


def test_validate_against_schema_accepts_valid_face_crop():
    data = {
        "uuid": generate_uuid(),
        "schema_version": "1.0",
        "parent_clip": {"uuid": generate_uuid(), "file": "clip.mp4"},
        "source_video": "clip.mp4",
        "track_id": 0,
        "duration_seconds": 2.5,
    }
    assert validate_against_schema(data, "face_crop") is True


def test_validate_against_schema_rejects_missing_required_field():
    # start_frame / end_frame / duration_seconds are required for person_clip.
    data = {"uuid": generate_uuid(), "schema_version": "1.0", "source_video": "v.mp4"}
    with pytest.raises(jsonschema.ValidationError):
        validate_against_schema(data, "person_clip")


def test_validate_against_schema_rejects_bad_uuid_pattern():
    data = {
        "uuid": "not-a-uuid",
        "schema_version": "1.0",
        "source_video": "v.mp4",
        "start_frame": 0,
        "end_frame": 1,
        "duration_seconds": 0.0,
    }
    with pytest.raises(jsonschema.ValidationError):
        validate_against_schema(data, "person_clip")


def test_load_schema_returns_image_detection_schema():
    schema = load_schema("image_detection")
    assert isinstance(schema, dict)
    assert schema.get("type") == "object"
    assert "detections" in schema["required"]


def test_validate_against_schema_accepts_valid_image_detection():
    data = {
        "uuid": generate_uuid(),
        "schema_version": "1.0",
        "image_path": "photo.JPG",
        "image_size": {"width": 1024, "height": 768},
        "num_persons": 1,
        "detections": [
            {
                "person_idx": 0,
                "bbox_tlbr": [120, 80, 540, 720],
                "bbox_confidence": 0.83,
                "keypoints": [[333.4, 327.9], [340.1, 330.0]],
                "keypoint_scores": [2.57, 2.41],
                "face_visible": True,
                "frontal_face": True,
            }
        ],
        "detector": {"name": "yolox-tiny", "confidence_threshold": 0.5},
    }
    assert validate_against_schema(data, "image_detection") is True


def test_validate_against_schema_rejects_image_detection_missing_detections():
    data = {"uuid": generate_uuid(), "schema_version": "1.0", "image_path": "p.JPG"}
    with pytest.raises(jsonschema.ValidationError):
        validate_against_schema(data, "image_detection")
