---
mode: agent
description: Start a goal-driven refactor/implementation session. Recalls the objective, indexes the codebase, picks the next chunk, implements it with quality gates, and stops for review before continuing. No auto-commit.
---

# Refactor-to-objective — session start

Follow the `refactor-to-objective` instructions (`.github/instructions/refactor-to-objective.instructions.md`) for the full loop protocol. This prompt triggers the resume sequence:

## 1. Recall the objective

Read `CLAUDE.md` § Objective (the 11 stages, FAIR + Annex IV, no-regression rule). The acceptance criterion lives there — do not edit it unilaterally.

## 2. Index first

Follow `socraticode-index-first` before navigating code:
- `codebase_status` → if stale, `codebase_index` and wait for 100%.
- Then use `codebase_search` / `codebase_symbol` / `codebase_impact` / `codebase_graph_circular` — not grep/glob — for structural questions.

## 3. Orient

- `git log --oneline -20` — recent history.
- Re-measure complexity ratchet baseline: `uv run python -m ruff check . --select C901 --config "lint.mccabe.max-complexity=20" --no-cache`
- Re-measure god-file sizes: `wc -l dardcollect/tracker.py dardcollect/quality.py dardcollect/pipeline_loggers.py`

## 4. Pick the next chunk

Based on the objective and git history, identify one concrete, scoped unit of work. State it explicitly before starting.

## 5. Implement

One chunk. Refactor opportunistically (duplication, naming, mixed responsibilities) while staying behavior-preserving. Do not add silent runtime fallbacks — surface failing paths to the user.

Dead-code review: for every candidate (unused import/function/class/branch), investigate why it's there (`git log -S`, blame) before deleting. If unclear, ask.

## 6. Quality gates (non-negotiable before marking done)

```
uv run python -m ruff check .
uv run python -m ruff format --check .
uv run python -m ty check
uv run python -m ruff check . --select C901 --config "lint.mccabe.max-complexity=20" --no-cache
uv run python -m pytest tests/ -q
```

Then the objective gate (~1–2 min on fixture):
```
uv run python scripts/run_pipeline.py --config config.test.yaml
uv run python scripts/golden_snapshot.py --dard-root DARD_test compare tests/fixtures/golden_manifest.json --validate
```

Both must exit 0. Hash drift is informational; hard-fail on missing CSV, sidecar volume out of bounds, broken provenance, or schema-invalid sidecar.

## 7. Stop and request review

Summarize: files changed, behavior delta, gate results, platform coverage. Hand the diff to the user. Do NOT run `git add`, `git commit`, or `git push`. Wait for the user's go-ahead before the next chunk.
