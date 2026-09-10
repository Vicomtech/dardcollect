---
name: refactor-to-objective
description: Autonomous goal-driven loop to move the code toward the agreed objective (AGENTS.md § Objective). Resume protocol + dead-code review + blocked-task pivot. Gates, fallback policy, and chunk-done criteria live in AGENTS.md (single source of truth) — this skill only adds the loop mechanics.
---

# refactor-to-objective

Implement or refactor toward the agreed objective. **AGENTS.md is the authoritative source** for: the objective, all quality gates (CPU + size + complexity + circular + docs + objective gate), the runtime fallback policy, and the Chunk DONE / NOT DONE criteria. This skill describes only the **loop protocol** to apply them — it does not re-list them.

When the objective and the user's in-session input conflict, surface it and prefer the user's input for *requirements*, the objective for *implementation approach*. Do not edit the objective unilaterally.

This repo has tracked debt (god-files > 600 lines, C901 violations); opportunistically refactor while preserving behavior (CSV set, sidecar volume/schema/provenance intact). GPU inference (TensorRT/CUDA) is non-deterministic — the golden gate checks **structure + provenance + validity**, not byte-match (see `AGENTS.md` § Objective verification). The gate validates **correctness, not wall-time** — a performance optimization (e.g. model-loading, stage overlap) can't be gate-validated; measure it on a real large run before building, and prefer the simple-correct memory-safe design over a complex one built preemptively (see `docs/PROGRESSIVE-WORKERS.md`).

**Dispatcher vs god-method (don't relocate):** a function whose length comes from **arg-passing verbosity to already-extracted stage helpers** (e.g. `run_pipeline`'s stage-wiring, the `orchestrator_plan` builders) is a *dispatcher*, not a god-method of logic — its length is intrinsic, not a concentration problem. Do NOT force such a dispatcher under an arbitrary line cap via a context-object refactor: that trades explicit args for `ctx.attr` indirection (relocates complexity, high churn across many helper signatures, high regression risk) and is NOT a maintainability win. Accept a readable dispatcher as the wiring layer. The god-method gate is about **logic concentration**, not raw line count.

## Resume protocol (each session)

1. **Recall the objective.** Read `AGENTS.md` § Objective + the relevant sub-doc ([docs/0-GETTING-STARTED.md](../../../docs/0-GETTING-STARTED.md), [docs/1-ARCHITECTURE.md](../../../docs/1-ARCHITECTURE.md), [docs/2-LINEAGE.md](../../../docs/2-LINEAGE.md), [docs/3-ANNOTATIONS.md](../../../docs/3-ANNOTATIONS.md)) as output-format references, NOT the acceptance criterion. For refactor history, `git log`. The local auto-memory (`MEMORY.md`) is loaded automatically each session by the harness — don't hardcode its machine-specific path.
2. **Orient in the code.** Locate the symbols a chunk touches with Grep (context/limit, include-filters) and narrow `Read` (`offset`/`limit`) before reading whole files — don't fan out broad searches.
3. **Check tasks.** `TaskList` for in-progress/pending tasks; pick the next one. If empty/stale, rebuild the work list from the objective.
4. **Implement one concrete chunk** — not the whole project. Refactor opportunistically as you go: when the code you're touching (or the code it calls) shows duplication, poor naming, mixed responsibilities, or a missing pattern, improve it in the same chunk. Don't gold-plate unrelated code, but don't leave nearby rot that you just created or stepped through — fix it while the context is loaded. Keep refactors **behavior-preserving** (CSV set, sidecar volume, provenance links, schema-validity preserved — NOT byte-identical inference, which is GPU-non-deterministic) and **separate from new-feature logic** in the change so the diff reads cleanly.
   - **Do not add runtime fallbacks** without asking the user first — surface the failing path and let the user choose. Only the exceptions listed in `AGENTS.md` § Runtime fallback policy are pre-approved; any other fallback needs explicit user approval before implementing.
5. **Dead-code review — every chunk.** While the context for a chunk is loaded, hunt for dead code: unused imports / functions / variables / classes / modules, unreachable branches, commented-out blocks, aspirational stubs never called, leftovers from removed or renamed stages/models. Use Grep (references across the repo), `ruff` (F401/F811/F841), and caller/callee checks via targeted reads to find candidates. For each candidate, **investigate *why* it's there before deleting**: grep for references, check `git log -S` / blame for origin and intent, and weigh public-API (the library API in [docs/5-LIBRARY-API.md](../../../docs/5-LIBRARY-API.md)) or planned-future-use value. If the reason is unclear or it might be deliberate, **ask the user** — do not delete silently. If it has no justification, delete it in the same chunk and prune its references. Log what was removed (and what was kept-after-asking) in the chunk's progress note.
6. **Run the gates — MANDATORY before marking done.** Run every gate listed in `AGENTS.md` § Objective verification + § Refactor methodology. All must pass; C901 count must not increase from chunk start. A green `pytest` does NOT substitute for the objective gate.
7. **Verify before done — objective gate is MANDATORY, not optional.** Run `AGENTS.md` § Objective verification Step 1 (pipeline EXIT 0) + Step 2 (golden snapshot EXIT 0). If either fails, the chunk is NOT done — fix and re-run. **Triple-platform:** if claiming cross-platform parity, test on Linux + Windows + macOS; if a platform can't be exercised, surface that honestly.
8. **Update tasks** (`TaskUpdate`) and append a short, dated progress note to the loop memory (most-recent-first, absolute dates).
9. **Close the session** (AGENTS.md § Session closure): update `MEMORY.md` (compact one-line closure entries) and log the cycle with `uv run python scripts/cycle_metrics.py log --phases <phases> --files <files> --status <status>`. `uv run python scripts/validate_harness.py` must pass after the closure update.
10. **Stop and request review.** Once a functionality is implemented and the gates pass, PAUSE the loop. Do not auto-continue to the next chunk. **The user commits — never commit yourself, not even after they approve** (see `AGENTS.md` § "Working rule — the user commits, never the assistant"). Summarize what changed (files, behavior, lint/type/golden result, gates, platform coverage) and hand the diff to the user. Do not run `git add`, `git commit`, or `git push`. Wait for the user's decision before resuming.

   **Exception — queue scoping (AGENTS.md § Task scoping):** when the session's request covers a
   numbered queue (GitHub issues, multi-chunk plan), do NOT pause between chunks — the batched
   questions of the first turn + the final diff review are the approval points. The pause-and-ask
   protocol applies per queue, not per chunk. A plan open question with a written recommendation
   is a decision default: proceed and record it, don't block on it.

## Scope honesty

Each session does one concrete chunk. The Chunk DONE vs NOT DONE checklist is in `AGENTS.md` § Scope honesty — use it as the gate. Critical rule: **if the objective gate is pending or failing, the chunk is NOT done**, no matter how green the CPU gates are. Surface status honestly — "CPU gates ✅, objective gate 🔄 IN PROGRESS" or "objective gate ❌ FAILED (reason)" — do not invent a "done" state.

## Blocked-task pivot

When a task is blocked on validation (missing tooling, env, access, or an un-ratified baseline): do NOT burn the loop on unvalidatable hand-written work. Document the gap + fallback strategy in code and memory, keep the task pending/blocked, and pivot to unblocked tasks. Revisit when the blocker lifts (e.g. a new `snapshots/` baseline pending the user running the pipeline on test media).
