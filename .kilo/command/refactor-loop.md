---
description: Goal-driven refactor/implementation loop session - recalls objective, orients in the code, picks next chunk, implements with quality gates, stops for review. No auto-commit.
---
Read and follow `.kilo/skills/refactor-to-objective/SKILL.md` in full (section Resume protocol). That file is the authoritative loop definition - quality gates, commands, dead-code review, and stop-and-review rules all live there.

This command triggers the resume sequence in order:

1. **Recall the objective** - read `AGENTS.md` section Objective.
2. **Orient in the code** - locate the symbols a chunk touches with Grep + narrow `Read`; re-measure the complexity ratchet and god-file sizes (`git log --oneline -20`, commands in the SKILL.md).
3. **Pick the next chunk** - one concrete, scoped unit of work; state it explicitly before starting.
4. **Implement** - per SKILL.md section Implement one concrete chunk (opportunistic refactor + dead-code review).
5. **Quality gates** - run every gate listed in SKILL.md section Quality gates before marking done.
6. **Stop and request review** - summarize, hand diff to user, do NOT commit.

$ARGUMENTS