# GitHub Copilot — DARDcollect project instructions

Project context loaded every session. Detailed skill instructions are in `.github/instructions/` and apply automatically.

## What this repo is

A GPU-accelerated multi-modal toolkit for downloading, processing, and annotating historical public-domain media from the [Internet Archive](https://archive.org), originally built for the [DETECTOR project](https://detector-project.eu/). It downloads videos/images/audio/documents organised by language, extracts person detections + 133-keypoint poses, transcribes speech, extracts document text, and produces 616×616 OFIQ-aligned face crops with rich `.json` sidecars — all with [FAIR](https://www.go-fair.org/fair-principles/) provenance and EU AI Act Annex IV documentation. Usable as a complete pipeline (bulk processing) or as a modular library (import individual components).

Eleven decoupled, resumable, independently re-runnable stages across four modality tracks (video / image / audio / document) that converge at quality filtering. See [README.md](../README.md) (hub), [docs/0-GETTING-STARTED.md](../docs/0-GETTING-STARTED.md), [docs/1-ARCHITECTURE.md](../docs/1-ARCHITECTURE.md), [docs/2-LINEAGE.md](../docs/2-LINEAGE.md), [docs/3-ANNOTATIONS.md](../docs/3-ANNOTATIONS.md), [docs/4-DEVELOPMENT.md](../docs/4-DEVELOPMENT.md), [docs/5-LIBRARY-API.md](../docs/5-LIBRARY-API.md).

## Toolchain

- **Package manager / runner:** `uv` (Python 3.12, TensorRT + CUDA 12.1 on Linux/Windows, MPS on macOS, CPU fallback). Run things via `uv run python …`, `uv run python -m ruff …`, `uv run python -m ty …`.
- **Lint + type-check:** `uv run python -m ruff check .` / `ruff format --check .` / `uv run python -m ty check` (configured in `pyproject.toml`).
- **Tests:** `uv run python -m pytest tests/ -q` (34 CPU-only tests, ~3s, no GPU needed). `pytest` is in the `dev` extra (`uv sync --extra dev`).
- **Config:** `config.yaml` is the user-owned source of truth. Validate with `uv run python -m dardcollect.config`.
- **GPU:** auto-detected at import. Use the GPU when available — detection/pose/OCR are GPU-accelerated.

## Working rules — non-negotiable

- **The user commits, never the assistant.** Do NOT run `git add`, `git commit`, or `git push`. Do not propose committing or ask "want me to commit?". Do the work, run verifications, stop at the working tree.
- **No silent runtime fallbacks.** Do not add backend cascades, `try/except` that swallows errors and defaults to a degraded path, or placeholder models without asking the user first. Surface the failing path and let the user choose.
- **Behavior-preserving changes must stay behavior-preserving.** CSV set, sidecar volume, provenance links, and schema-validity must be preserved — NOT byte-identical inference output (GPU is non-deterministic).
- **Authoring language is English.** Match the surrounding repo content; translate any stray non-English authored text in the same chunk.
- **Triple-platform:** a chunk is not done until it works on Linux, Windows, AND macOS (or the platform gap is surfaced honestly).

## Objective (acceptance criterion)

The toolkit must produce, end-to-end:

- **Download** — videos/images/audio/documents from Archive.org by language (ISO 639-2), writing `downloads.csv`.
- **Detection + pose** — YOLOX-Tiny person bboxes + CIGPose 133-keypoint wholebody poses.
- **Video pipeline** — person clips (OC-SORT + scene-change) → 616×616 OFIQ face crops → Whisper-Small transcriptions.
- **Image pipeline** — person detections (JSON sidecar) → 616×616 OFIQ face crops.
- **Audio pipeline** — Whisper-Small transcriptions with language detection.
- **Document pipeline** — text from PDFs (text layer / PaddleOCR PP-OCRv5 fallback) + plain-text files. Output `.text.txt` + `.annotation.json`.
- **Quality** — OFIQ 7-dimensional + MagFace unified scoring on every face crop.
- **FAIR + EU AI Act** — UUID v4, full provenance chain, 10 incremental CSVs + JSON sidecars validated by `jsonschema` at write time.

### Objective verification (fast fixture, ~1–2 min)

```
uv run python scripts/run_pipeline.py --config config.test.yaml
uv run python scripts/golden_snapshot.py --dard-root DARD_test compare tests/fixtures/golden_manifest.json --validate
```

Both must exit 0. The golden compare is **non-determinism-tolerant**: hard-fails on missing CSV, sidecar volume out of bounds, broken provenance, or schema-invalid sidecar; hash drift from GPU non-determinism is informational only. A green `pytest tests/` does NOT substitute for this gate.

## Code quality ratchet (measured, not aspirational)

- **Complexity:** `uv run python -m ruff check . --select C901 --config 'lint.mccabe.max-complexity=20' --no-cache`. Current violations (2026-07-08): 5. A chunk must NOT increase this count.
- **Size:** functions ≤ ~80 lines, files ≤ ~600 lines. God-files (2026-07-08): `dardcollect/tracker.py` (~858), `dardcollect/quality.py` (~784), `dardcollect/pipeline_loggers.py` (~748). A chunk must not grow a god-file.
- **Circular deps:** `codebase_graph_circular` (SocratiCode) must stay at 0.

## Scope honesty

Each session does one concrete chunk. Be honest about what is **done** vs **blocked by env** (GPU, dataset, missing tooling) vs **pending user ratification of a new golden baseline**. Do not mark the loop complete until the code demonstrably satisfies the objective.
