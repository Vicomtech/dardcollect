---
name: harness-self-improve
description: Audit the harness each cycle (rules, skills, scripts, docs) with structured exploration and rule retirement review. Proposals only, never silent changes.
---

# harness-self-improve

Keep the harness aligned with repo reality. Run at session start as a quick
check, when the harness changes, or on user request. Output is findings +
proposals; **never execute changes without explicit user confirmation**
(`AGENTS.md`: the user commits, the agent proposes).

## 1. Rules (`AGENTS.md` vs `docs/HARNESS_RULES.md`)

- Contradictory, obsolete, or duplicated rules.
- Every `HARNESS_RULES.md` row carries an Enforcement cell from the fixed
  vocabulary (`scripts/harness_extra.py`); a gated rule whose artifact is
  gone is a retirement candidate.
- New rules need a motivating failure stated in the same cycle; a rule
  without one is aspirational and is not adopted.

## 2. Skills and scripts

- Each skill reflects the current workflow; no redundant or missing skills.
- Every `scripts/*.py` is in `SCRIPT_MANIFEST` or is a `diag_*`
  diagnostic; leftovers are findings.
- Every `_check_*` in `validate_harness.py` is wired in `CHECK_REGISTRY`
  (coverage gate); `uv run python scripts/diag_mutation_probe.py` is the
  firing proof (diagnostic, never a gate).

## 3. Structured exploration

Sessions concentrate on a handful of files. From `scripts/cycle_metrics.py`
output (or `git log --name-only`), take the least-recently-touched group —
a whole directory family, not a single file — and inspect it for defects the
gates do not cover (prose accuracy, stale numbers, orphaned references).
Report what was inspected, including "nothing found".

## 4. Retirement review

Scan `HARNESS_RULES.md` for rules whose motivating failure no longer
applies. Mark candidates (retirement itself is a user decision); a retired
rule keeps its row, marked retired with reason. Rejected proposals go to
`MEMORY.md` Open items with reason + reopening condition (rejection memory).

## Output

Findings, consolidation proposals, updated rule rows for anything added /
changed / retired. Report findings (what changed, what was verified, what
failed), not bookkeeping.
