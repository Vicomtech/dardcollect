#!/usr/bin/env python3
"""Extract text and write annotation sidecars for downloaded text documents."""

import json
import logging
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml
from tqdm import tqdm

from persondet.config import get_log_level
from persondet.fair import add_fair_metadata, generate_uuid, reorganize_for_fair
from persondet.ocr import DocumentExtractor
from persondet.pipeline_loggers import DocumentTextExtractionLogger
from persondet.provenance import PROVENANCE_FILENAME, now_iso, record_stage
from persondet.script_utilities import _TqdmHandler

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.yaml"

SUPPORTED_EXTENSIONS = {".pdf", ".txt"}


_handler = _TqdmHandler()
_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
logging.basicConfig(handlers=[_handler], level=logging.INFO, force=True)
logger = logging.getLogger(__name__)


@dataclass
class DocumentPreprocessConfig:
    input_dir: str
    output_dir: str
    overwrite: bool = False
    min_text_length: int = 50
    enable_ocr: bool = True
    gpu_id: int = 0

    @classmethod
    def from_yaml(cls, config_path: str) -> "DocumentPreprocessConfig":
        with open(config_path, encoding="utf-8") as f:
            config = yaml.safe_load(f)
        cfg = config.get("document_preprocessing", {})
        gpu_id = config.get("gpu_id", 0)  # Global GPU setting
        return cls(
            input_dir=cfg.get("input_dir", "DARD/archive_org_public_domain/texts"),
            output_dir=cfg.get("output_dir", "DARD/preprocessed_documents"),
            overwrite=cfg.get("overwrite", False),
            min_text_length=cfg.get("min_text_length", 50),
            enable_ocr=cfg.get("enable_ocr", True),
            gpu_id=gpu_id,
        )


def main() -> None:
    logging.getLogger().setLevel(get_log_level(str(CONFIG_PATH)))
    cfg = DocumentPreprocessConfig.from_yaml(str(CONFIG_PATH))
    input_dir = Path(cfg.input_dir)
    output_dir = Path(cfg.output_dir)

    if not input_dir.exists():
        logger.warning("Input directory does not exist: %s", input_dir)
        sys.exit(0)

    output_dir.mkdir(parents=True, exist_ok=True)

    extractor = DocumentExtractor(gpu_id=cfg.gpu_id, enable_ocr=cfg.enable_ocr)

    # Initialize traceability logger
    text_extraction_logger = DocumentTextExtractionLogger(dard_root="DARD")

    files = [
        f
        for f in sorted(input_dir.rglob("*"))
        if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS
    ]

    if not files:
        logger.info("No supported documents found in %s", input_dir)
        sys.exit(0)

    logger.info("Found %d documents to process", len(files))
    started_at = now_iso()
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

            annotation_path.write_text(
                json.dumps(annotation, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            processed += 1

            # Log extraction to traceability CSV
            text_extraction_logger.log_text_extraction(
                extraction_id=annotation.get("uuid", doc_path.stem),
                source_document=doc_path.name,
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

    record_stage(
        output_dir.parent / PROVENANCE_FILENAME,
        {
            "stage": "document_preprocessing",
            "started_at": started_at,
            "completed_at": now_iso(),
            "software": {"script": "scripts/extract_text_from_doc.py"},
            "stats": {"processed": processed, "skipped": skipped},
        },
    )


if __name__ == "__main__":
    main()
