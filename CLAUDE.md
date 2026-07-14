# CLAUDE.md — DARDcollect

Project context for Claude Code, loaded every session. Portable (committed) layer; the local auto-memory at `~/.claude/projects/.../memory/` is the personal/live scratch layer on top of this.

## What this repo is
A GPU-accelerated multi-modal toolkit for downloading, processing, and annotating historical public-domain media from the [Internet Archive](https://archive.org), originally built for the [DETECTOR project](https://detector-project.eu/). It downloads videos/images/audio/documents organised by language, extracts person detections + 133-keypoint poses, transcribes speech, extracts document text, and produces 616×616 OFIQ-aligned face crops with rich `.json` sidecars — all with [FAIR](https://www.go-fair.org/fair-principles/) provenance and EU AI Act Annex IV documentation. Usable as a complete pipeline (bulk processing) or as a modular library (import individual components).

Thirteen decoupled, resumable, independently re-runnable stages across four modality tracks (video / image / audio / document) that converge at quality annotation (12 processing stages in fixture, plus download for full runs). See [README.md](README.md) (hub), [docs/0-GETTING-STARTED.md](docs/0-GETTING-STARTED.md) (walkthrough), [docs/1-ARCHITECTURE.md](docs/1-ARCHITECTURE.md) (full architecture + FAIR strategy), [docs/2-LINEAGE.md](docs/2-LINEAGE.md) (CSV provenance/traceability), [docs/3-ANNOTATIONS.md](docs/3-ANNOTATIONS.md) (sidecar JSON formats), [docs/4-DEVELOPMENT.md](docs/4-DEVELOPMENT.md) (GPU setup + dev workflow), [docs/5-LIBRARY-API.md](docs/5-LIBRARY-API.md) (library API).

## Toolchain
- **Package manager / runner:** `uv` (creates the venv, pins Python 3.12, resolves all deps incl. TensorRT + CUDA 12.1 wheels on Linux/Windows, MPS on macOS). Run things via `uv run python …`, `uv run python -m ruff …`, `uv run python -m ty …`. The venv also lives at `.venv/` if you prefer the interpreter directly (`.venv/Scripts/python.exe` on Windows, `.venv/bin/python` on Linux/macOS).
- **Lint + type-check** (configured in `pyproject.toml`): `uv run python -m ruff check .` / `ruff format --check .` / `python -m ty check`. Ruff selects E/W/.../RUF; isort with `known-first-party = ["dardcollect"]`.
- **Tests:** a CPU-only unit suite exists under `tests/` (`test_fair.py` — FAIR metadata + JSON-Schema validation; `test_config.py` — config parsing + log-level; `test_viewer_smoke.py` — viewer indexing/server smoke checks). Run with `uv run python -m pytest tests/ -q` (~seconds, no GPU needed). `pytest` is in the `[project.optional-dependencies] dev` extra (`uv sync --extra dev`). The suite covers pure CPU helpers and viewer discovery logic; GPU-accelerated stages (detection/pose/OCR/quality) are verified via the objective gate / golden harness (see § Objective verification), not unit tests.
- **GPU:** auto-detected at import (NVIDIA libs auto-preloaded). TensorRT/CUDA 12.1 on Linux/Windows, MPS on macOS, automatic CPU-only fallback. **Use the GPU when available** — detection/pose/OCR are GPU-accelerated.
- **Config:** `configs/config.archive_all.yaml` (the general / full Archive.org config, formerly `config.yaml`) is the user-owned source of truth (search query, `media_types`, model paths, detection/quality thresholds, output dirs, device). Lean per-modality custom configs live alongside it in `configs/` (`config.custom_videos.yaml`, `config.custom_images.yaml`, `config.custom_audios.yaml`, `config.custom_texts.yaml`). Don't hardcode config values in this doc; read them at run time.
- **CLI contract:** Pipeline orchestrator and stage scripts are config-driven; runtime workflow behavior must be controlled through config (`configs/config.archive_all.yaml` / `configs/config.test.yaml`, including `run_pipeline` settings), not extra ad-hoc CLI flags. `run_pipeline.skip_stages: [aliases]` skips individual downstream stages (cascades to their dependents); `run_pipeline.skip_download` skips the download stage.
- **Config paths are repo-root-relative, not config-dir-relative:** stage scripts do `Path(cfg.input_dir)` relative to their cwd (the repo root, where the orchestrator launches them). The orchestrator's `_resolve_config_path` and the viewer's `_resolve` MUST resolve the same way (against `REPO_ROOT`), never against the config file's directory. When moving/renaming configs (e.g. into `configs/`), audit all three (orchestrator + viewer + stage `CONFIG_PATH` defaults) — a config-dir-relative regression makes downstream stages skip with "no usable inputs" and the viewer raise "no configured output dirs". Pinned by `tests/test_run_pipeline_progressive.py::test_resolve_config_path_*` and `tests/test_viewer_smoke.py::test_resolve_*` / `test_load_cfg_*`.
- **Triple-platform verification:** a chunk is not done until it works on Linux, Windows, AND macOS — the program claims all three. Run on at least the platforms you have available; if a platform can't be exercised (e.g. no macOS machine), surface that honestly rather than marking done off a single-platform pass.

## Working rule — the user commits, never the assistant
The user stages and commits their own work. Do NOT run `git add`, `git commit`, or `git push`; do not propose committing or ask "want me to commit?". Do the work, run verifications, and stop at the working tree. Treat "do you need this file?" as a real keep/delete question, not a commit prompt.

## Runtime fallback policy

Runtime fallbacks are allowed only when they are explicit, documented, observable in logs, and approved by the user.

1. **No new fallback without user approval first.**
   - Before adding any new fallback path (backend cascade, degraded-path default, catch-and-continue behavior), ask the user and get explicit approval.
   - If approval is not granted, fail loudly with a clear error instead of silently degrading behavior.

2. **Allowed exceptions (pre-approved in this repo):**
   - **GPU → CPU execution fallback** during runtime setup (documented toolchain behavior).
   - **OCR/ONNX provider fallback chain** for execution providers and OCR routing.
   - **Tracker optional dependency fallback** (`cython_bbox` unavailable → NumPy IoU path).
   - **Resume progress file fallback** (invalid/unreadable progress JSON → restart from frame 0 with warning).
   - **Quality annotation fallback** (`.magface.json` missing → compute unified score directly when possible).
   - **Single-modality config → other-modality skip** (in `filter_face_crops_by_quality` and `annotate_face_quality`: whichever of the video / image config sections is absent is skipped with an info log, so a lean `media_types: ["video"]` OR `["image"]` config works; a missing *key* within a present section is still a hard error). `extract_persons_from_images` reads its face-crop thresholds from `image_face_crop_extraction` (the image section), not the video `face_crop_extraction` section.

3. **Guardrails for any fallback (including exceptions):**
   - Never be silent: emit a warning/info log stating trigger and selected fallback path.
   - Never hide data-integrity/provenance/schema failures: those must fail loudly.
   - Keep behavior deterministic at the contract level (same outputs/contracts, even if slower path is used).

4. **Documentation rule:**
   - If a fallback is added/changed/removed, update this section and any impacted user docs in the same chunk.
   - Chunk is NOT done if fallback behavior changed but docs/rules are not updated.

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

All 13 stages are resumable, independently re-runnable, and behavior-verified via golden snapshot (see § Objective verification below).

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
   - **Run it FRESH:** `uv run python scripts/objective_gate.py` — wipes `DARD_test`, runs the fixture pipeline, runs golden `--validate`; exits 0 only if both pass. (The two steps below are what it runs.)
   - Step 1: `uv run python scripts/run_pipeline.py --config configs/config.test.yaml` exits 0 (all 12 fixture stages run to completion)
   - Step 2: `uv run python scripts/golden_snapshot.py --dard-root DARD_test compare tests/fixtures/golden_manifest.json --validate` exits 0 (CSVs present, volume within bounds, provenance links resolve, sidecars schema-valid)
   - ⚠️ **Fresh is non-negotiable:** stages are resumable (`.done` sentinels), so a re-run on an existing `DARD_test` skips completed stages and the golden compare can pass against **stale outputs** from a prior session — masking regressions that only surface when stages actually execute. Always sign off a chunk with `objective_gate.py` (fresh), never with a `--no-wipe` re-run.

**Note on GPU non-determinism:** TensorRT/CUDA inference is non-deterministic run-to-run, so byte-identical outputs are NOT expected. Inference-derived fields (keypoints, scores, bboxes, transcription text, OCR) drift, and even detection/crop counts vary. The golden gate is **non-determinism-tolerant**: it hard-fails only on missing CSVs (regression), volume collapse/>4x swing (collapse), broken provenance, or schema-invalid sidecars. Hash diffs from GPU drift are informational (does NOT fail unless `--strict`). When a change is intentional (fix, threshold tuning, model swap), confirm it is no worse than prior run (same or more valid detections/crops, provenance intact).

**One-time fixture setup per machine** (requires dataset at `DARD/archive_org_public_domain/`):

> The fixture is **for the objective gate only** — `make_fixture_media.py` samples the smallest files from a full Archive.org download into a tiny stable subset so the gate runs in ~1–2 min. The `DARD/archive_org_public_domain/` path is the source it derives from; it is NOT a constraint on normal usage. **Custom / existing datasets skip the fixture entirely** — set `run_pipeline.skip_download: true` and point config inputs at your dataset (see [docs/0-GETTING-STARTED.md](docs/0-GETTING-STARTED.md#use-an-existing-dataset-no-download)).
>
> ⚠️ **Keep `tests/fixtures/media/` small and baseline-matched.** It must hold only the files the golden baseline was captured against (e.g. `_test_short.mp4`, `1954-10-17AFRS-UN-Jingle.mp3`). Huge real-data files there make the gate slow (>~3 min, the 681 MB video / 670 MB audio hit this) and produce golden drift. Test real data via a custom config's `input_dir`, not in the fixture dir. If `objective_gate.py` is slow or drifts, check `find tests/fixtures/media -type f -size +5M` and move non-baseline files out; regenerate clean with `make_fixture_media.py`.

1. `uv run python scripts/make_fixture_media.py` — builds `tests/fixtures/media/` (30 s video + sample images/audio/PDFs)
2. `uv run python scripts/make_test_config.py` — generates `configs/config.test.yaml` (redirects to fixture paths)
3. First pipeline run + `uv run python scripts/golden_snapshot.py --dard-root DARD_test capture tests/fixtures/golden_manifest.json` — capture baseline

**Full dataset (separate from dev loop):** production runs use `scripts/run_pipeline.py` without `--config` (hours, separate from fixture), baseline under gitignored `snapshots/` (per-machine).

## Refactor methodology — behavior-preserving + golden-gated

Every refactor step must preserve behavior and be verified against the golden snapshot (§ Objective verification) before marking done. See the `refactor-to-objective` skill for the full protocol loop (dead-code review, chunk workflow, gates). Gate **definitions** live in § Objective verification; this section tracks only the quantitative targets' **current state**.

**Complexity & Size Tracking** (measure before chunk start):
- **C901 (cyclomatic)**: `uv run python -m ruff check . --select C901 --config 'lint.mccabe.max-complexity=20' --no-cache` → current: 0 violations (target: reduce to `max-complexity=10` over time). Chunk NOT done if count increases.
- **File size** (target ≤ 600 lines; functions ≤ 80 lines): god-files (measure with `wc -l`) — [quality.py](dardcollect/quality.py) ~556 (near boundary). Resolved: [run_pipeline.py](scripts/run_pipeline.py) was 686 → split into run_pipeline.py ~455 + [orchestrator_plan.py](dardcollect/orchestrator_plan.py) ~322. Previously listed [tracker.py](dardcollect/tracker.py) (~732→236) and [pipeline_loggers.py](dardcollect/pipeline_loggers.py) (~748→457) are well under 600 — re-list only if they regrow past 600. Chunk NOT done if a touched god-file grows; when touched, shrink it (extract coherent units).

**Golden/Snapshot Verification**: tool [scripts/golden_snapshot.py](scripts/golden_snapshot.py) (normalized SHA-256 manifest of CSVs + sidecars); per-machine baseline `snapshots/golden_manifest.json` (gitignored, user-ratified). Run before marking done: `compare tests/fixtures/golden_manifest.json --validate` (fixture) or `snapshots/golden_manifest.json` (production). Validates at write (§ Objective): every sidecar calls `add_fair_metadata` + `validate_against_schema`.

**Layer boundaries**: `codebase_graph_circular` (SocratiCode) must stay 0; run `codebase_impact` before splitting/renaming for blast radius.

## Scope honesty
Each session does one concrete chunk. Be honest about what's **done** vs **blocked by env** (GPU, dataset, missing tooling, missing `tests/`) vs **pending user ratification of a new golden baseline**. Don't mark the loop complete until the code demonstrably satisfies the Objective and the relevant stages run end-to-end on test media producing the expected FAIR artifacts with no regression.

### Chunk DONE ✅ (all of the following)
- ✅ Every gate in § Objective verification is green: CPU (ruff check + format, ty check, pytest), complexity (C901 ≤ 20, not increased from chunk start), size (files ≤ 600, functions ≤ 80, no god-file grown), circular deps (0), dead-code pruned, documentation synced (README + AI Systems + sub-docs + config files), objective gate (pipeline EXIT 0 + golden compare EXIT 0)
- ✅ Platform tested when claiming parity (Windows + 1 other; WSL counts as Linux)
- ✅ User reviewed the diff + explicitly approved
- ✅ **User committed** (never auto-commit — assistant never runs `git add/commit/push`)

### Chunk NOT DONE ❌ (if any)
- ❌ Any gate in § Objective verification is red (CPU / complexity / size / circular / documentation / objective)
- ❌ Platform untested when claiming parity
- ❌ User has not reviewed/approved or has not committed

**CRITICAL:** If documentation OR objective gate fails, chunk is NOT done, no matter how green CPU gates are. Surface status honestly: "CPU ✅, docs ❌ UNSYNCED" or "objective ❌ FAILED (reason)".