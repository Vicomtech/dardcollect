---
name: feature-intake
description: Take in a feature request or numbered queue as a design note before implementing (scope, FAIR impact, resumability, gates, evidence).
---

# feature-intake

A request implemented straight from its title arrives partial. Intake first,
implement after. Complements the Feature Request Protocol in `AGENTS.md`
(single source of truth for gates) and `.kilo/FEATURE_WORKFLOW.md` (the
checklist) — this skill is the invocable workflow.

## Step 1 — Write the design note (before any code)

- **Problem**: what is wrong or missing, with the observed evidence.
- **Non-goals**: what is explicitly out of scope.
- **Placement**: which modality (video/image/audio/document/orchestration),
  which stage (existing vs new — prefer existing), what it touches beside
  the target.
- **FAIR impact**: new CSVs or sidecar extensions? Schema + provenance chain
  implications (`source-manifest UUID → Download → Clip/Crop → Quality`).
- **Resumability**: `.done` sentinel? CSV dedup / skip logic?
- **Gates**: which `AGENTS.md` gates must pass (CPU + quality + objective
  gate fresh); new gates ship with their defect-planting test.
- **Evidence**: what the completion record will cite (gate outputs, golden
  diff) — never "it works".

## Step 2 — Scope a queue whole

For a numbered queue, build the full list in the first turn and batch
questions once; an open question with a written recommendation is a decision
default — proceed and record, do not block between chunks (`AGENTS.md`
§ Task scoping). A fix-it loop with no progress stops after four iterations,
then asks.

## Step 3 — Done criteria

Done when every gate in Step 1 passes on a fresh run
(`scripts/objective_gate.py`, never `--no-wipe`), the evidence is recorded,
and the report states findings (what changed, what was verified, what
failed) — not bookkeeping. Never downgrade an approval to a registration:
name the exact consequence of approval (files, implemented vs registered).
