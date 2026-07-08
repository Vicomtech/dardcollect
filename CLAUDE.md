# CLAUDE.md — DARDcollect

Project context for Claude Code, loaded every session. Portable (committed) layer; the local auto-memory at `~/.claude/projects/.../memory/` is the personal/live scratch layer on top of this.

## What this repo is
A GPU-accelerated multi-modal toolkit for downloading, processing, and annotating historical public-domain media from the [Internet Archive](https://archive.org), originally built for the [DETECTOR project](https://detector-project.eu/). It downloads videos/images/audio/documents organised by language, extracts person detections + 133-keypoint poses, transcribes speech, extracts document text, and produces 616×616 OFIQ-aligned face crops with rich `.json` sidecars — all with [FAIR](https://www.go-fair.org/fair-principles/) provenance and EU AI Act Annex IV documentation. Usable as a complete pipeline (bulk processing) or as a modular library (import individual components).

Eleven decoupled, resumable, independently re-runnable stages across four modality tracks (video / image / audio / document) that converge at quality filtering. See [README.md](README.md) (hub), [docs/0-GETTING-STARTED.md](docs/0-GETTING-STARTED.md) (walkthrough), [docs/1-ARCHITECTURE.md](docs/1-ARCHITECTURE.md) (full architecture + FAIR strategy), [docs/2-LINEAGE.md](docs/2-LINEAGE.md) (CSV provenance/traceability), [docs/3-ANNOTATIONS.md](docs/3-ANNOTATIONS.md) (sidecar JSON formats), [docs/4-DEVELOPMENT.md](docs/4-DEVELOPMENT.md) (GPU setup + dev workflow), [docs/5-LIBRARY-API.md](docs/5-LIBRARY-API.md) (library API).

## Toolchain
- **Package manager / runner:** `uv` (creates the venv, pins Python 3.12, resolves all deps incl. TensorRT + CUDA 12.1 wheels on Linux/Windows, MPS on macOS). Run things via `uv run python …`, `uv run python -m ruff …`, `uv run python -m ty …`. The venv also lives at `.venv/` if you prefer the interpreter directly (`.venv/Scripts/python.exe` on Windows, `.venv/bin/python` on Linux/macOS).
- **Lint + type-check** (configured in `pyproject.toml`): `uv run python -m ruff check .` / `ruff format --check .` / `python -m ty check`. Ruff selects E/W/.../RUF; isort with `known-first-party = ["dardcollect"]`.
- **Tests:** `pytest tests/` is documented in [docs/4-DEVELOPMENT.md](docs/4-DEVELOPMENT.md) but **`tests/` does not exist yet in this repo** — treat the unit-suite gate as aspirational, not green-by-default. If you create tests, do it explicitly; otherwise rely on the snapshot/golden + CSV/JSON-sidecar verification below.
- **GPU:** auto-detected at import (NVIDIA libs auto-preloaded). TensorRT/CUDA 12.1 on Linux/Windows, MPS on macOS, automatic CPU-only fallback. **Use the GPU when available** — detection/pose/OCR are GPU-accelerated.
- **Config:** `config.yaml` is the user-owned source of truth (search query, `media_types`, model paths, detection/quality thresholds, output dirs, device). Don't hardcode its values in this doc; read them from `config.yaml` at run time. Validate with `uv run python -m dardcollect.config`.
- **Triple-platform verification:** a chunk is not done until it works on Linux, Windows, AND macOS — the program claims all three. Run on at least the platforms you have available; if a platform can't be exercised (e.g. no macOS machine), surface that honestly rather than marking done off a single-platform pass.

## Working rule — the user commits, never the assistant
The user stages and commits their own work. Do NOT run `git add`, `git commit`, or `git push`; do not propose committing or ask "want me to commit?". Do the work, run verifications, and stop at the working tree. Treat "do you need this file?" as a real keep/delete question, not a commit prompt.

## Objective
Build a labelled audiovisual dataset from public-domain Internet Archive media (historical, pre-1960 film). The toolkit must produce, end-to-end across its four modalities (video / image / audio / document), the following artifacts — **this IS the acceptance criterion** (there is no frozen spec):

**Download** — fetch videos / images / audio / documents from Archive.org, organised by language (ISO 639-2; `und/` for items with no language metadata), writing a unified `downloads.csv` (one row per file).

**Detection + pose** — YOLOX-Tiny person bounding boxes + CIGPose 133-keypoint wholebody poses (face/body/hands), robust to the grain and low resolution of pre-1960 film.

**Video pipeline** — extract person clips (OC-SORT tracking + scene-change detection + face/duration/frontal clip-segmentation rules) → 616×616 OFIQ-aligned face crops → Whisper-Small transcriptions (`.transcription.json` with language detection).

**Image pipeline** — person detections (JSON sidecar) → 616×616 OFIQ-aligned face crops.

**Audio pipeline** — Whisper-Small transcriptions with language detection.

**Document pipeline** — extract text from PDFs (text layer if ≥100 chars, else PaddleOCR PP-OCRv5 fallback with per-script routing Latin/Cyrillic/Greek for all 24 EU official languages) and plain-text files with encoding detection; discard documents below `min_text_length` (50 chars). Output `.text.txt` + `.annotation.json`.

**Quality** — OFIQ 7-dimensional (ISO/IEC 29794-5) + MagFace unified scoring on every face crop; filter by MagFace threshold. Optional frame extraction.

**FAIR + EU AI Act** — every artifact carries a UUID v4 and a full provenance chain (Archive.org ID → Download UUID → Clip/Crop UUID → Quality scores), recorded in 10 incremental CSVs + JSON sidecars with `schema_version`, validated by `jsonschema` at write time. Every automated component — learned model or rule-based — is documented as an AI system per Annex IV (see the AI Systems table in [README.md](README.md)).

**No-regression rule:** the eleven stages are each resumable and independently re-runnable. Behavior-preserving changes must keep outputs identical — the CSVs and JSON sidecars byte-match the `snapshots/` baselines. Intentional changes (a fix, a threshold tuning, a model swap) must be **no worse than the previous run** on test media driven by `config.yaml` (user-owned). A green `pytest tests/` does NOT count if a produced artifact or a provenance link regressed.

**Code quality & documentation:** follow the rules already set in this file and the loaded skills (`refactor-to-objective`, `keep-docs-navigable`, `socraticode-index-first`) — not restated here.

## Refactor methodology — behavior-preserving + golden-gated
When refactoring, every step must be **behavior-preserving** and verified against a golden snapshot before being marked done — a green unit suite alone is not enough (and the unit suite is aspirational here anyway, see Toolchain).

**Complexity ratchet (runnable, non-regression):** `uv run python -m ruff check . --select C901 --config 'lint.mccabe.max-complexity=20' --no-cache`. Measure the violation count at chunk start; the chunk is NOT done if it *increased* the count. Reducing the count is progress; the goal is to drive toward `max-complexity=10` over time. (C901 is intentionally NOT in `pyproject.toml` `select` so the main `ruff check` gate stays green while complexity is tracked as ratcheted debt — don't silently move C901 into `select` without fixing the offenders.) **Current count as of 2026-07-08: 5 violations** — `process_video` in [dardcollect/person_clips.py](dardcollect/person_clips.py) (36), `process_video` in [dardcollect/face_crops.py](dardcollect/face_crops.py) (33), `main` in [pipeline/download_media_from_archive.py](pipeline/download_media_from_archive.py) (24), `main` in [pipeline/extract_persons_from_images.py](pipeline/extract_persons_from_images.py) (22), `index_data` in [viewer/index_data.py](viewer/index_data.py) (21). Re-measure each chunk.

**Size discipline (tracked debt):** target functions ≤ ~80 lines and files ≤ ~600 lines. **God-files over the target** (as of 2026-07-08, re-measure with `wc -l`): [dardcollect/tracker.py](dardcollect/tracker.py) (~858), [dardcollect/quality.py](dardcollect/quality.py) (~784), [dardcollect/pipeline_loggers.py](dardcollect/pipeline_loggers.py) (~748). A chunk must not *grow* a god-file; when you touch one, shrink it (extract coherent units out). New code must stay under the targets.

**Layer boundaries (runnable):** `codebase_graph_circular` (SocratiCode) must stay at 0 circular deps (non-regression). Before splitting/renaming, run `codebase_impact` for blast radius. The subsystem→pattern map is maintained in [docs/1-ARCHITECTURE.md](docs/1-ARCHITECTURE.md); when you introduce or change a subsystem's pattern, update that map in the same chunk (per `keep-docs-navigable`). A hard import-linter gate is a proposed future hardening — not enabled yet, so don't write checks that assume it exists.

**Golden/snapshot verification:**
- `snapshots/` exists but currently only holds `tmp/` — **there are no ratified baselines yet**. Treat the golden gate as **blocked / pending ratification**, not silently passing. For the surface you touched, build a harness that compares the produced CSVs + JSON sidecars (and/or the `DARD/` output tree) against a captured baseline under `snapshots/`, capture a baseline, and flag it to the user for ratification — do NOT invent a "pass". Until a baseline is ratified, the golden gate is pending and you surface that honestly.
- **No-regression on intentional changes:** when a change is NOT behavior-preserving (an improvement, a fix, a threshold tuning, a model swap), run the relevant stages per `config.yaml` on test media and compare against the **prior** outputs — confirm it is no worse than before (same or more valid detections/crops, provenance chain intact, sidecars still `jsonschema`-valid). If a metric regressed, the change is not done — revert or fix.
- `data/`-style test media and `DARD/` outputs are large and gitignored → test videos and golden baselines do NOT travel with the repo; provide the media and (re)capture baselines per machine.

Refactor history: `git log` (commits + diffs are the record). Current/live state lives in the local auto-memory; verify against code before asserting any state as fact.

## Scope honesty
Each session does one concrete chunk. Be honest about what's **done** vs **blocked by env** (GPU, dataset, missing tooling, missing `tests/`) vs **pending user ratification of a new golden baseline**. Don't mark the loop complete until the code demonstrably satisfies the Objective and the relevant stages run end-to-end on test media producing the expected FAIR artifacts with no regression.