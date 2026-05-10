#!/usr/bin/env python3
"""Extract text and write annotation sidecars for downloaded text documents."""

import json
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from tqdm import tqdm

from dardcollect.config import DocumentPreprocessConfig, get_log_level
from dardcollect.fair import (
    add_fair_metadata,
    generate_uuid,
    reorganize_for_fair,
    validate_against_schema,
)
from dardcollect.ocr import DocumentExtractor
from dardcollect.pipeline_loggers import DocumentTextExtractionLogger
from dardcollect.pipeline_utils import _TqdmHandler

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.yaml"

SUPPORTED_EXTENSIONS = {".pdf", ".txt"}


_handler = _TqdmHandler()
_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
logging.basicConfig(handlers=[_handler], level=logging.INFO, force=True)
logger = logging.getLogger(__name__)


def main() -> None:
    logging.getLogger().setLevel(get_log_level(str(CONFIG_PATH)))
    cfg = DocumentPreprocessConfig.from_yaml(str(CONFIG_PATH))
    input_dir = Path(cfg.input_dir)
    output_dir = Path(cfg.output_dir)

    if not input_dir.exists():
        logger.warning("Input directory does not exist: %s", input_dir)
        sys.exit(0)

    output_dir.mkdir(parents=True, exist_ok=True)

    extractor = DocumentExtractor(
        gpu_id=cfg.gpu_id,
        enable_ocr=cfg.enable_ocr,
        languages=cfg.ocr_languages,
    )

    # Initialize traceability logger
    downloads_csv = input_dir.parent / "downloads.csv"
    text_extraction_logger = DocumentTextExtractionLogger(
        output_dir=str(output_dir), downloads_csv_path=downloads_csv
    )

    files = [
        f
        for f in sorted(input_dir.rglob("*"))
        if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS
    ]

    if not files:
        logger.info("No supported documents found in %s", input_dir)
        sys.exit(0)

    logger.info("Found %d documents to process", len(files))
    processed = skipped = 0

    for doc_path in tqdm(files, desc="Preprocessing", unit="doc"):
        annotation_path = output_dir / (doc_path.stem + ".annotation.json")
        text_path = output_dir / (doc_path.stem + ".text.txt")

        if annotation_path.exists() and text_path.exists() and not cfg.overwrite:
            skipped += 1
            continue

        try:
            result = extractor.extract(doc_path)
            text = result["text"]

            if len(text.strip()) < cfg.min_text_length:
                logger.debug("%s: text too short (%d chars), skipping", doc_path.name, len(text))
                skipped += 1
                continue

            text_path.write_text(text, encoding="utf-8")

            annotation: dict[str, Any] = {
                "uuid": generate_uuid(),
                "schema_version": "1.0",
                "source_file": doc_path.name,
                "extraction_method": result["method"],
                "page_count": result["page_count"],
                "word_count": result["word_count"],
                "char_count": result["char_count"],
                "text_file": text_path.name,
                "processed_at": datetime.now(UTC).isoformat(),
            }
            annotation = add_fair_metadata(annotation, schema_type="document")
            annotation = reorganize_for_fair(annotation, "document")
            validate_against_schema(annotation, "document")

            annotation_path.write_text(
                json.dumps(annotation, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            processed += 1

            # Log extraction to traceability CSV
            text_extraction_logger.log_text_extraction(
                source_document_path=str(doc_path.absolute()),
                text_length=result["char_count"],
                word_count=result["word_count"],
                model_version=result["method"],
                output_annotation_path=str(annotation_path.absolute()),
                output_text_path=str(text_path.absolute()),
            )

            logger.debug("%s: %s, %d words", doc_path.name, result["method"], result["word_count"])

        except Exception as e:
            logger.warning("Failed to process %s: %s", doc_path.name, e)

    logger.info(
        "Done — %d processed, %d skipped → %s",
        processed,
        skipped,
        output_dir.resolve(),
    )
    text_extraction_logger.print_summary()


if __name__ == "__main__":
    main()
