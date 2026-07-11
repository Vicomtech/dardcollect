# GitHub Copilot — DARDcollect project instructions

Read `CLAUDE.md` in full at the start of every session. It is the single authoritative source of project context, toolchain, working rules, objective, and code quality gates. 

**Critical rule:** When working on code chunks, the objective gate (§ Objective verification) is **mandatory, not optional** — do NOT mark a chunk done without running both:
1. `uv run python scripts/run_pipeline.py --config config.test.yaml` (must exit 0)
2. `uv run python scripts/golden_snapshot.py --dard-root DARD_test compare tests/fixtures/golden_manifest.json --validate` (must exit 0)

The three skills are loaded automatically via the instruction files in `.github/instructions/` — each one delegates to its canonical `.claude/skills/*/SKILL.md`. Consult them for detailed protocols.
