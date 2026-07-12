# CLAUDE.md — DARDcollect

Project context for Claude Code, loaded every session. Portable (committed) layer; the local auto-memory at `~/.claude/projects/.../memory/` is the personal/live scratch layer on top of this.

## What this repo is
A GPU-accelerated multi-modal toolkit for downloading, processing, and annotating historical public-domain media from the [Internet Archive](https://archive.org), originally built for the [DETECTOR project](https://detector-project.eu/). It downloads videos/images/audio/documents organised by language, extracts person detections + 133-keypoint poses, transcribes speech, extracts document text, and produces 616×616 OFIQ-aligned face crops with rich `.json` sidecars — all with [FAIR](https://www.go-fair.org/fair-principles/) provenance and EU AI Act Annex IV documentation. Usable as a complete pipeline (bulk processing) or as a modular library (import individual components).

Twelve decoupled, resumable, independently re-runnable stages across four modality tracks (video / image / audio / document) that converge at quality filtering (11 stages in fixture, plus download for full runs). See [README.md](README.md) (hub), [docs/0-GETTING-STARTED.md](docs/0-GETTING-STARTED.md) (walkthrough), [docs/1-ARCHITECTURE.md](docs/1-ARCHITECTURE.md) (full architecture + FAIR strategy), [docs/2-LINEAGE.md](docs/2-LINEAGE.md) (CSV provenance/traceability), [docs/3-ANNOTATIONS.md](docs/3-ANNOTATIONS.md) (sidecar JSON formats), [docs/4-DEVELOPMENT.md](docs/4-DEVELOPMENT.md) (GPU setup + dev workflow), [docs/5-LIBRARY-API.md](docs/5-LIBRARY-API.md) (library API).

## Toolchain
- **Package manager / runner:** `uv` (creates the venv, pins Python 3.12, resolves all deps incl. TensorRT + CUDA 12.1 wheels on Linux/Windows, MPS on macOS). Run things via `uv run python …`, `uv run python -m ruff …`, `uv run python -m ty …`. The venv also lives at `.venv/` if you prefer the interpreter directly (`.venv/Scripts/python.exe` on Windows, `.venv/bin/python` on Linux/macOS).
- **Lint + type-check** (configured in `pyproject.toml`): `uv run python -m ruff check .` / `ruff format --check .` / `python -m ty check`. Ruff selects E/W/.../RUF; isort with `known-first-party = ["dardcollect"]`.
- **Tests:** a CPU-only unit suite exists under `tests/` (`test_fair.py` — FAIR metadata + JSON-Schema validation; `test_config.py` — config parsing + log-level; `test_viewer_smoke.py` — viewer indexing/server smoke checks). Run with `uv run python -m pytest tests/ -q` (~seconds, no GPU needed). `pytest` is in the `[project.optional-dependencies] dev` extra (`uv sync --extra dev`). The suite covers pure CPU helpers and viewer discovery logic; GPU-accelerated stages (detection/pose/OCR/quality) are verified via the objective gate / golden harness (see § Objective verification), not unit tests.
- **GPU:** auto-detected at import (NVIDIA libs auto-preloaded). TensorRT/CUDA 12.1 on Linux/Windows, MPS on macOS, automatic CPU-only fallback. **Use the GPU when available** — detection/pose/OCR are GPU-accelerated.
- **Config:** `config.yaml` is the user-owned source of truth (search query, `media_types`, model paths, detection/quality thresholds, output dirs, device). Don't hardcode its values in this doc; read them from `config.yaml` at run time. Validate with `uv run python -m dardcollect.config`.
- **CLI contract:** Pipeline orchestrator and stage scripts are config-driven; runtime workflow behavior must be controlled through config (`config.yaml` / `config.test.yaml`, including `run_pipeline` settings), not extra ad-hoc CLI flags.
- **Triple-platform verification:** a chunk is not done until it works on Linux, Windows, AND macOS — the program claims all three. Run on at least the platforms you have available; if a platform can't be exercised (e.g. no macOS machine), surface that honestly rather than marking done off a single-platform pass.

## Working rule — the user commits, never the assistant
The user stages and commits their own work. Do NOT run `git add`, `git commit`, or `git push`; do not propose committing or ask "want me to commit?". Do the work, run verifications, and stop at the working tree. Treat "do you need this file?" as a real keep/delete question, not a commit prompt.

## Feature Request Protocol — How New Features Are Evaluated

**When a feature request arrives:**

1. **Pre-implementation intake** — ask clarifying questions before coding
   - Is scope clear? Does it align with the objective?
   - Which pipeline stage(s) will it touch? Is it resumable?
   - Does it generate new CSVs or extend sidecars? (FAIR compliance)
   - Is a design doc provided?

2. **Design doc requirement** — a 1-page markdown answering:
   - Problem statement (what + why)
   - Architecture (which stages, new vs. existing)
   - FAIR impact (new CSVs? sidecars? provenance implications?)
   - Resumability strategy (`.done` sentinel? CSV dedup?)
   - Test plan (fixture + full dataset?)

3. **Implementation gates** (§ § Objective verification § Chunk DONE ✅ applies)
   - All CPU + complexity + objective gates must pass
   - Fixture pipeline must run end-to-end, EXIT 0
   - Golden snapshot must pass, 0 hard-fail
   - Platform testing: Windows + (WSL | Linux | macOS)

4. **PR checklist** (commit message includes all gates + platform tested)
   - If any gate is blocked/deferred, explain why in commit message
   - If blocked, feature is NOT merged until gate is passed

**See [`.claude/FEATURE_WORKFLOW.md`](.claude/FEATURE_WORKFLOW.md) for complete checklist and [docs/CONTRIBUTING.md](docs/CONTRIBUTING.md) for human-facing instructions.**

## Objective
Build a labelled audiovisual dataset from public-domain Internet Archive media (historical, pre-1960 film). The toolkit must produce, end-to-end across its four modalities (video / image / audio / document), the following artifacts — **this IS the acceptance criterion** (there is no frozen spec):

- **Download** — Archive.org → language-organized CSVs (`downloads.csv`)
- **Detection + pose** — YOLOX-Tiny bboxes + CIGPose 133-keypoint wholebody poses
- **Video pipeline** — person clips (tracking + scene-change) → WAV audio extraction → OFIQ-aligned face crops → face masks → Whisper-Small transcriptions
- **Image pipeline** — detections (JSON sidecar) → OFIQ-aligned face crops → face masks
- **Audio pipeline** — Whisper-Small transcriptions with language detection
- **Document pipeline** — PDF text extraction (text layer / PaddleOCR PP-OCRv5) + encoding detection
- **Quality** — OFIQ 7-dim (ISO/IEC 29794-5) + MagFace unified scoring, filter by threshold
- **FAIR + EU AI Act Annex IV** — UUID v4 + full provenance chains (Archive.org ID → Download → Clip/Crop → Quality), 10 CSVs + JSON sidecars, `jsonschema` validation at write time, every AI system documented in README AI Systems table

All 12 stages are resumable, independently re-runnable, and behavior-verified via golden snapshot (see § Objective verification below).

## Objective verification (runnable) — How we know we're done

The objective (§ Objective above) is met end-to-end when:
1. **Code quality gates — quantitative, non-negotiable:**
   - No god-files (`.py` > ~600 lines); C901 ≤ 20 (target 10); 0 circular deps
   - CPU gates green: `uv run python -m ruff check .`, `ruff format --check .`, `uv run python -m ty check`, `uv run python -m pytest tests/ -q`
   - Dead code pruned (unused imports/functions reviewed, justified or deleted)

2. **Documentation gate — MANDATORY, not optional** (see `keep-docs-navigable` skill § rule 4):
   - If behavior/config/CLI/CSV/sidecar/model/AI system changed → update README + sub-doc in same chunk
   - **Every chunk: review & sync config files if code changed:**
     - `.vscode/launch.json` — stage names/paths match pipeline/scripts/ reality
     - `pyproject.toml` — entry points, dependencies, metadata
     - `.github/instructions/*.md` + `.claude/skills/` — references stay current
   - README: one-liner + install + usage + AI Systems table (every automation component documented)
   - Sub-docs linked; no broken links
   - Chunk NOT done if docs or config out of sync with code

3. **Objective gate (runnable, ~1–2 min on fixture)** — the primary verification:
   - Step 1: `uv run python scripts/run_pipeline.py --config config.test.yaml` exits 0 (all 11 stages run to completion on fixture)
   - Step 2: `uv run python scripts/golden_snapshot.py --dard-root DARD_test compare tests/fixtures/golden_manifest.json --validate` exits 0 (CSVs present, volume within bounds, provenance links resolve, sidecars schema-valid)

**Note on GPU non-determinism:** TensorRT/CUDA inference is non-deterministic run-to-run, so byte-identical outputs are NOT expected. Inference-derived fields (keypoints, scores, bboxes, transcription text, OCR) drift, and even detection/crop counts vary. The golden gate is **non-determinism-tolerant**: it hard-fails only on missing CSVs (regression), volume collapse/>4x swing (collapse), broken provenance, or schema-invalid sidecars. Hash diffs from GPU drift are informational (does NOT fail unless `--strict`). When a change is intentional (fix, threshold tuning, model swap), confirm it is no worse than prior run (same or more valid detections/crops, provenance intact).

**One-time fixture setup per machine** (requires dataset at `DARD/archive_org_public_domain/`):
1. `uv run python scripts/make_fixture_media.py` — builds `tests/fixtures/media/` (30 s video + sample images/audio/PDFs)
2. `uv run python scripts/make_test_config.py` — generates `config.test.yaml` (redirects to fixture paths)
3. First pipeline run + `uv run python scripts/golden_snapshot.py --dard-root DARD_test capture tests/fixtures/golden_manifest.json` — capture baseline

**Full dataset (separate from dev loop):** production runs use `scripts/run_pipeline.py` without `--config` (hours, separate from fixture), baseline under gitignored `snapshots/` (per-machine).

## Refactor methodology — behavior-preserving + golden-gated

Every refactor step must preserve behavior and be verified against golden snapshot before marking done. See `refactor-to-objective` skill for the full protocol loop (dead-code review, chunk workflow, gates). This section tracks quantitative targets and current state.

**Complexity & Size Tracking** (runnable, measure before chunk start):
- **C901 (cyclomatic)**: `uv run python -m ruff check . --select C901 --config 'lint.mccabe.max-complexity=20' --no-cache`
  - Current: 0 violations (as of 2026-07-11); target: reduce to `max-complexity=10` over time
  - All offenders reduced below 20: `process_video` (person_clips 36→<20, face_crops 33→<20), `main` (download 24→<20, extract_persons 22→<20), `index_data` (viewer 21→<20)
  - Chunk NOT done if count increases
- **File size**: target ≤ 600 lines; functions ≤ 80 lines
  - God-files (current, measure with `wc -l`): [tracker.py](dardcollect/tracker.py) ~732, [quality.py](dardcollect/quality.py) ~467 (split in Chunk 12), [pipeline_loggers.py](dardcollect/pipeline_loggers.py) ~748
  - Chunk NOT done if god-file grows; when touched, shrink it (extract coherent units)

**Golden/Snapshot Verification**:
- Tool: [scripts/golden_snapshot.py](scripts/golden_snapshot.py) — captures normalized SHA-256 manifest of CSVs + JSON sidecars
- Per-machine baseline: `snapshots/golden_manifest.json` (gitignored); user ratifies as reference before a baseline becomes "passing"
- Run before marking done: `uv run python scripts/golden_snapshot.py compare tests/fixtures/golden_manifest.json --validate` (fixture) or `snapshots/golden_manifest.json` (production)
- Validates at write (§ Objective): every sidecar type (document, transcription, face-crop, image-detection, quality-annotation) calls `add_fair_metadata` + `validate_against_schema`

**Documentation**: see `keep-docs-navigable` skill — if chunk changed stage/config/CSV/sidecar/model/AI system, update README + relevant sub-doc in same chunk.

**Layer boundaries**: `codebase_graph_circular` (SocratiCode) must stay at 0 circular deps; run `codebase_impact` before splitting/renaming for blast radius.

## Scope honesty
Each session does one concrete chunk. Be honest about what's **done** vs **blocked by env** (GPU, dataset, missing tooling, missing `tests/`) vs **pending user ratification of a new golden baseline**. Don't mark the loop complete until the code demonstrably satisfies the Objective and the relevant stages run end-to-end on test media producing the expected FAIR artifacts with no regression.

### Chunk DONE ✅ (all of the following)
- ✅ CPU gates (ruff check + format, ty check, pytest) **all green**
- ✅ Complexity gate: C901 ≤ 20, **not increased from chunk start**
- ✅ Size gate: files ≤ 600 lines, functions ≤ 80 lines, **god-files not grown**
- ✅ Circular deps: 0 (verified via `codebase_graph_circular`)
- ✅ Dead-code review: unused imports/functions pruned (or user approved keeping)
- ✅ **Documentation gate — MANDATORY:** README + AI Systems table in sync if code changed; no broken links; check config files (`.vscode/launch.json`, `pyproject.toml`, etc.) for affected references
- ✅ **Objective gate — MANDATORY:**
  - ✅ Step 1: `uv run python scripts/run_pipeline.py --config config.test.yaml` **exits 0** (fixture runs, no stage fails)
  - ✅ Step 2: `uv run python scripts/golden_snapshot.py --dard-root DARD_test compare tests/fixtures/golden_manifest.json --validate` **exits 0** (provenance intact, no schema-invalid sidecar, volume within bounds)
- ✅ User reviewed diff + explicitly approved
- ✅ **User committed** (never auto-commit — assistant never runs `git add/commit/push`)

### Chunk NOT DONE ❌ (if any of the following)
- ❌ CPU gates failing (ruff, ty, pytest, C901)
- ❌ Size/complexity gates violated (file > 600 lines, C901 increased, god-file grown)
- ❌ **Documentation out of sync** (README, AI Systems, sub-docs — MANDATORY gate)
- ❌ **Objective gate failing** (pipeline EXIT ≠ 0 OR golden snapshot EXIT ≠ 0 — MANDATORY gate)
- ❌ Circular deps > 0
- ❌ Platform untested: if claiming Windows/Linux/macOS parity, test on at least Windows + 1 other (WSL counts as Linux)
- ❌ User has not committed

**CRITICAL:** If documentation OR objective gate fails, chunk is NOT done, no matter how green CPU gates are. Surface status honestly: "CPU ✅, docs ❌ UNSYNCED" or "objective ❌ FAILED (reason)".