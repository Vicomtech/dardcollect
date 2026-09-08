---
description: Goal-driven refactor/implementation loop session - recalls objective, indexes codebase, picks next chunk, implements with quality gates, stops for review. No auto-commit.
---
Read and follow `.kilo/skills/refactor-to-objective/SKILL.md` in full (section Resume protocol). That file is the authoritative loop definition - quality gates, commands, dead-code review, and stop-and-review rules all live there.

This command triggers the resume sequence in order:

1. **Recall the objective** - read `AGENTS.md` section Objective.
2. **Index first** - follow `socraticode-index-first` (index the repo, then structural navigation, not grep/glob fan-outs).
3. **Orient** - `git log --oneline -20`; re-measure complexity ratchet and god-file sizes (commands in the SKILL.md).
4. **Pick the next chunk** - one concrete, scoped unit of work; state it explicitly before starting.
5. **Implement** - per SKILL.md section Implement one concrete chunk (opportunistic refactor + dead-code review).
6. **Quality gates** - run every gate listed in SKILL.md section Quality gates before marking done.
7. **Stop and request review** - summarize, hand diff to user, do NOT commit.

$ARGUMENTS