# GitHub Copilot — DARDcollect project instructions

Read `CLAUDE.md` in full at the start of every session. It is the single authoritative source of project context, toolchain, working rules, objective, and code quality gates. 

**Critical rule:** When working on code chunks, ALL THREE of these gates are **mandatory, not optional** — do NOT mark a chunk done without ALL passing:
1. **Documentation gate:** Updated README + sub-docs if behavior/config/CLI/CSV/sidecar/model/AI system changed
2. **Fixture pipeline:** `uv run python scripts/run_pipeline.py --config config.test.yaml` → EXIT 0
3. **Golden snapshot:** `uv run python scripts/golden_snapshot.py --dard-root DARD_test compare tests/fixtures/golden_manifest.json --validate` → EXIT 0

Detailed protocols in `.claude/skills/` (delegated from `.github/instructions/`).

**Fallback rule (mandatory):** Do not add runtime fallbacks without explicit user approval first. Only the fallback exceptions listed in `CLAUDE.md` "Runtime fallback policy" are pre-approved; any new fallback proposal must be approved by the user before implementation.
