from dardcollect.extraction_logger import ExtractionLogger


def test_print_summary_uses_max_persons_per_frame_without_error(tmp_path, caplog):
    output_dir = tmp_path / "clips"
    output_dir.mkdir(parents=True, exist_ok=True)

    logger = ExtractionLogger(output_dir=output_dir)
    logger.log_extraction(
        source_video="sample.mp4",
        fps=24.0,
        start_frame=0,
        end_frame=47,
        start_seconds=0.0,
        duration_seconds=2.0,
        max_persons_per_frame=2,
        detector_model="yolox_tiny",
        detector_confidence=0.8,
        output_path=str(output_dir / "sample_00m00s-00m02s.mp4"),
    )

    with caplog.at_level("INFO"):
        logger.print_summary()

    assert "Extraction Summary" in caplog.text
    assert "Total persons: 2" in caplog.text


def test_print_summary_supports_legacy_num_persons_field(tmp_path, caplog):
    output_dir = tmp_path / "clips"
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "clips_extraction.csv"
    csv_path.write_text(
        "uuid,source_video,duration_seconds,num_persons,detector_confidence\n"
        "1,sample.mp4,2.0,3,0.9\n",
        encoding="utf-8",
    )

    logger = ExtractionLogger(output_dir=output_dir)

    with caplog.at_level("INFO"):
        logger.print_summary()

    assert "Extraction Summary" in caplog.text
    assert "Total persons: 3" in caplog.text
