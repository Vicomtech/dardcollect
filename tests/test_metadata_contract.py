"""CPU-only tests pinning the code to `scripts/metadata_contract.json`.

The contract is the single source of truth for the metadata surface. Docs are
checked against it in `scripts/validate_harness.py`; here the code is: every CSV
writer's `fieldnames` must equal the contract's columns, and the schema
inventory must match the `schemas/` directory exactly. Together they make a
column/field rename fail a gate instead of silently drifting the docs.
"""

import json
from pathlib import Path

from dardcollect.extraction_logger import ExtractionLogger
from dardcollect.fair import SCHEMA_VERSIONS
from dardcollect.modality_loggers import (
    AudioTranscriptionsExtractionLogger,
    DocumentTextExtractionLogger,
    ImageFaceCropsExtractionLogger,
    ImagePersonDetectionLogger,
)
from dardcollect.pipeline_loggers import (
    FaceCropsExtractionLogger,
    FilteredFaceCropsLogger,
    FramesExtractionLogger,
    TranscriptionsExtractionLogger,
)

REPO_ROOT = Path(__file__).resolve().parent.parent

# Which logger owns each contract CSV (filename -> class exposing `fieldnames`).
WRITERS = {
    "clips_extraction.csv": ExtractionLogger,
    "frames_extraction.csv": FramesExtractionLogger,
    "video_face_crops_extraction.csv": FaceCropsExtractionLogger,
    "image_face_crops_extraction.csv": ImageFaceCropsExtractionLogger,
    "transcriptions_extraction.csv": TranscriptionsExtractionLogger,
    "audio_transcriptions_extraction.csv": AudioTranscriptionsExtractionLogger,
    "image_person_detection.csv": ImagePersonDetectionLogger,
    "document_text_extraction.csv": DocumentTextExtractionLogger,
    "video_filtered_face_crops.csv": FilteredFaceCropsLogger,
    "image_filtered_face_crops.csv": FilteredFaceCropsLogger,
}


def _contract() -> dict:
    path = REPO_ROOT / "scripts" / "metadata_contract.json"
    return json.loads(path.read_text(encoding="utf-8"))


def test_every_contract_csv_matches_its_logger_fieldnames():
    contract = _contract()
    assert set(contract["csv"]) == set(WRITERS), "contract CSVs and writer map disagree"
    for name, columns in contract["csv"].items():
        cls = WRITERS[name]
        assert cls.fieldnames == columns, (
            f"{name}: {cls.__name__}.fieldnames {cls.fieldnames} != contract {columns}"
        )


def test_schema_inventory_matches_the_schemas_directory():
    contract = _contract()
    declared = set(contract["schema"].values())
    on_disk = {p.name for p in (REPO_ROOT / "schemas").glob("*.json")}
    assert declared == on_disk, f"schemas on disk {sorted(on_disk)} != contract {sorted(declared)}"


def test_schema_version_keys_are_declared_in_the_contract():
    contract = _contract()
    assert set(SCHEMA_VERSIONS) <= set(contract["schema"])


def test_harness_check_fires_on_doc_drift(tmp_path, monkeypatch):
    """The validator check must fail when a doc drops a contract column/schema."""
    from harness_fixture import build_repo, vh

    build_repo(tmp_path)
    (tmp_path / "scripts" / "metadata_contract.json").write_text(
        json.dumps(
            {
                "csv": {"clips_extraction.csv": ["uuid", "output_path"]},
                "schema": {"frame": "frame_schema.json"},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(vh, "REPO_ROOT", tmp_path)

    # Stale docs: the column list differs and the schema is not named.
    (tmp_path / "docs" / "2-LINEAGE.md").write_text("uuid,download_uuid\n", encoding="utf-8")
    (tmp_path / "docs" / "3-ANNOTATIONS.md").write_text("no schema here\n", encoding="utf-8")
    errors = vh._check_metadata_contract()
    assert any("clips_extraction.csv" in e for e in errors)
    assert any("frame_schema.json" in e for e in errors)

    # Aligned docs: no errors.
    (tmp_path / "docs" / "2-LINEAGE.md").write_text("uuid,output_path\n", encoding="utf-8")
    (tmp_path / "docs" / "3-ANNOTATIONS.md").write_text("frame_schema.json\n", encoding="utf-8")
    assert vh._check_metadata_contract() == []
