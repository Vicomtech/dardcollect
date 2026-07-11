# Copilot Tab Rules — DARDcollect

**Complete rules:** Read `CLAUDE.md` in full for the complete project context, objective, and working rules. These are quick practical checks for Tab code generation.

## Quick checks (before generating code)

This repo uses Python 3.12 + uv. Lint with ruff (E/W/I/RUF), type-check with ty.
Always run before marking done: uv run python -m ruff check . && uv run python -m ruff format --check . && uv run python -m ty check

Functions ≤ ~80 lines, files ≤ ~600 lines. Max cyclomatic complexity 20 (target 10).
Do NOT grow the god-files: dardcollect/tracker.py, dardcollect/quality.py, dardcollect/pipeline_loggers.py.

No silent runtime fallbacks. Do not add try/except that swallows errors and defaults to a degraded path without asking the user first.

Authoring language is English. Do not introduce non-English identifiers or comments.

The user commits — never run git add, git commit, or git push.
