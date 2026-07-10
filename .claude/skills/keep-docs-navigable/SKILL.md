---
name: keep-docs-navigable
description: Keep the repo's main README.md clear and self-contained (what this is, install, usage), split detail into linked .md files so the README never balloons, and keep all docs in sync with the code. Standing rule for every session in this repo.
---

# keep-docs-navigable

The main `README.md` is the front door. Any user landing here must immediately understand: **what this repo is**, **how to install it**, and **how to use it**. Keep it short and scannable; push detail into linked sub-documents and link them so navigation is obvious.

## Standing rules

1. **README is the hub, not the whole doc.** It must always contain, at minimum:
   - One-paragraph project description (what DARDcollect is: a GPU-accelerated multi-modal toolkit for downloading, processing, and annotating historical public-domain media from the Internet Archive; FAIR + EU AI Act Annex IV; usable as complete pipeline or modular library).
   - **Install** section: prerequisites (Python 3.12, `uv`), `uv sync` (bundles TensorRT + CUDA 12.1 on Linux/Windows, MPS on macOS, CPU fallback), bundled ONNX models in `dardcollect/models/`, env gotchas (GPU/driver, `config.yaml`).
   - **Usage** section: the minimum commands to run the 11 stages by modality (download → video / image / audio / document → quality annotation + filter), plus config basics (`config.yaml`, `media_types`).
   - The **AI Systems (EU AI Act Annex IV)** table — every automated component (learned model or rule-based) documented with its implementation file + model card. A new pipeline component MUST get a row here and a model/system card.
   - **Docs index**: pointers to the numbered sub-docs (below).
   - Pointers to [docs/0-GETTING-STARTED.md](docs/0-GETTING-STARTED.md), [docs/1-ARCHITECTURE.md](docs/1-ARCHITECTURE.md), [docs/2-LINEAGE.md](docs/2-LINEAGE.md), [docs/3-ANNOTATIONS.md](docs/3-ANNOTATIONS.md), [docs/4-DEVELOPMENT.md](docs/4-DEVELOPMENT.md), [docs/5-LIBRARY-API.md](docs/5-LIBRARY-API.md), [docs/CONTRIBUTING.md](docs/CONTRIBUTING.md).

2. **Split when the README would balloon.** Detail lives in dedicated numbered `.md` files under `docs/` (`0-GETTING-STARTED`, `1-ARCHITECTURE`, `2-LINEAGE`, `3-ANNOTATIONS`, `4-DEVELOPMENT`, `5-LIBRARY-API`). If a new area appears (e.g. a dedicated OFIQ-quality doc, a config reference, a per-modality deep-dive), create a focused numbered sub-doc rather than growing the README. Each split doc:
   - Has a clear title + one-line purpose at the top.
   - Is linked from the README's docs index.
   - Links back to the README and to sibling docs where relevant (bidirectional where it aids navigation).
   - Stays focused on one topic — don't recreate a second monolith.

3. **Links must resolve.** Use relative links from the repo root. After adding/renaming/moving a doc, check that every inbound link still resolves (grep for the old path). A broken docs index is worse than no index.

4. **Keep docs in sync with code.** When a session changes behavior — a new stage, a new script, a renamed config key, a changed CSV column, a changed sidecar JSON field, a new model, a new AI system — update the README and/or the relevant sub-doc **in the same chunk**, before marking the task done. This pairs with the refactor-to-objective quality gate (documentation is a non-negotiable gate, not an afterthought). If a script's `--help` changed, the doc's usage block must match it. If a CSV schema or sidecar JSON field changed, [docs/2-LINEAGE.md](docs/2-LINEAGE.md) / [docs/3-ANNOTATIONS.md](docs/3-ANNOTATIONS.md) must change in the same chunk. If a new automated component was added, the README AI Systems table + its model/system card must appear in the same chunk.

5. **Honest docs.** Mark what's done vs blocked-by-env plainly (e.g. CPU-only fallback, MPS-only on macOS, `tests/` documented but not yet present). Don't document aspirational behavior as if it works. This mirrors the loop's scope-honesty rule.

## When to apply

- At the end of every refactor-to-objective chunk: did the change affect install, usage, a stage, a config key, a CSV column, a sidecar field, a model, or an AI system? → update docs in the same chunk.
- When the README grows past ~250–300 lines or a section becomes a wall of text → split it (this README is already sizable — be wary of growing it further; prefer linking out).
- When adding a new top-level feature/area → create its sub-doc and add it to the README docs index in the same chunk.

## Quick checklist before marking a chunk done
- [ ] README still answers: what is this / how to install / how to use?
- [ ] README docs index lists and links every sub-doc that exists.
- [ ] No broken links (relative paths resolve).
- [ ] Changed stage/CLI/CSV/sidecar/model is reflected in the matching doc (or LINEAGE/ANNOTATIONS).
- [ ] Any new automated component has a README AI Systems table row + a model/system card.
- [ ] Blocked/aspirational features are marked honestly.

Related: `refactor-to-objective` (documentation quality gate), `socraticode-index-first`.