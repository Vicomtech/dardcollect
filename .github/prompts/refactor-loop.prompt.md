---
mode: agent
description: Start a goal-driven refactor/implementation session. Recalls the objective, indexes the codebase, picks the next chunk, implements it with quality gates, and stops for review before continuing. No auto-commit.
---

# Refactor-to-objective — session start

Read and follow `.claude/skills/refactor-to-objective/SKILL.md` in full (§ Resume protocol). That file is the authoritative loop definition — quality gates, commands, dead-code review, and stop-and-review rules all live there.

This prompt triggers the resume sequence in order:

1. **Recall the objective** — read `CLAUDE.md` § Objective.
2. **Index first** — follow `socraticode-index-first` (`codebase_status` → `codebase_index` if stale → use SocratiCode tools, not grep/glob).
3. **Orient** — `git log --oneline -20`; re-measure complexity ratchet and god-file sizes (commands in the SKILL.md).
4. **Pick the next chunk** — one concrete, scoped unit of work; state it explicitly before starting.
5. **Implement** — per SKILL.md § Implement one concrete chunk (opportunistic refactor + dead-code review).
6. **Quality gates** — run every gate listed in SKILL.md § Quality gates before marking done.
7. **Stop and request review** — summarize, hand diff to user, do NOT commit.
