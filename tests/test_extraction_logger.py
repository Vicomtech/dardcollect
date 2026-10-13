import csv

import pytest

from dardcollect.extraction_logger import ClipRecord, ExtractionLogger


def _record(output_dir) -> ClipRecord:
    return ClipRecord(
        source_video="sample.mp4",
        detector_model="yolox_tiny",
        detector_confidence=0.8,
        output_path=str(output_dir / "sample_00m00s-00m02s.mp4"),
    )


def _rows(csv_path):
    with open(csv_path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def test_print_summary_reports_counts_without_error(tmp_path, caplog):
    output_dir = tmp_path / "clips"
    output_dir.mkdir(parents=True, exist_ok=True)

    logger = ExtractionLogger(output_dir=output_dir)
    logger.log_extraction(_record(output_dir))

    with caplog.at_level("INFO"):
        logger.print_summary()

    assert "Extraction Summary" in caplog.text
    assert "Total clips extracted: 1" in caplog.text


def test_row_uuid_is_the_artifact_uuid(tmp_path):
    """The row uuid is the clip sidecar's uuid (one identity space)."""
    output_dir = tmp_path / "clips"
    logger = ExtractionLogger(output_dir=output_dir)
    logger.log_extraction(_record(output_dir), artifact_uuid="clip-sidecar-uuid")

    assert _rows(output_dir / "clips_extraction.csv")[0]["uuid"] == "clip-sidecar-uuid"


def test_download_uuid_resolved_from_manifest(tmp_path):
    """The clip links upstream via download_uuid, matched by source filename."""
    manifest = tmp_path / "downloads.csv"
    with open(manifest, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["uuid", "filename_downloaded"])
        writer.writeheader()
        writer.writerow({"uuid": "dl-1", "filename_downloaded": "sample.mp4"})

    output_dir = tmp_path / "clips"
    logger = ExtractionLogger(output_dir=output_dir, downloads_csv_path=manifest)
    logger.log_extraction(_record(output_dir))

    row = _rows(output_dir / "clips_extraction.csv")[0]
    assert row["download_uuid"] == "dl-1"
    assert "archive_org_identifier" not in row


def test_stale_header_is_rejected(tmp_path):
    """A clips CSV with the pre-2026-10-11 header must not be silently appended to."""
    output_dir = tmp_path / "clips"
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "clips_extraction.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["uuid", "archive_org_identifier"])
        writer.writeheader()
        writer.writerow({"uuid": "old", "archive_org_identifier": ""})

    with pytest.raises(ValueError, match="stale header"):
        ExtractionLogger(output_dir=output_dir)
