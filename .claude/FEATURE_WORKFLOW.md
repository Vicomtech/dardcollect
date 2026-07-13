---
applyTo: "feature-requests"
---

# Feature Request Workflow — Claude/Copilot Protocol

**When evaluating a feature request, follow this checklist before implementation.** The quality gates, Chunk DONE / NOT DONE criteria, runtime fallback policy, and the objective all live in [CLAUDE.md](../CLAUDE.md) (single source of truth) — this file only adds the feature-specific intake + design-doc protocol.

## 1. Feature Intake (Pre-Implementation)

**Read the request and ask these questions:**

- [ ] **Scope clarity**: Is the request clearly scoped? If vague, request clarification in a Q&A format, don't guess.
- [ ] **Objective alignment**: Does it advance the project objective (§ CLAUDE.md § Objective)? Or is it orthogonal/supportive?
- [ ] **FAIR compliance**: Does it generate new CSVs/sidecars? If yes, verify a JSON schema exists and is documented.
- [ ] **Modality**: Which pipeline modality does it belong to (video/image/audio/document)?
- [ ] **Stage placement**: Which pipeline stage should it hook into, or is it a new stage?
- [ ] **Resumability**: Is the output resumable (`.done` sentinel, CSV idempotence)?

**If any question is unclear, STOP and ask for clarification. Do not proceed with implementation.**

## 2. Architecture Decision

**Before coding, decide:**

- **New stage** vs **integration into existing stage** (prefer existing)
- **New CSV** vs **sidecar extension** (prefer sidecars for per-item metadata)
- **CPU-only** vs **GPU** (specify if GPU required)
- **Resumable checkpoints** (required: `.done` sentinel, CSV dedupe, or skip logic)
- **File size budget**: Will output grow large? (e.g., masks + original images = storage concern?)

**Create a 1-page design doc (markdown) answering all of the above.**

## 3. Implementation

**Before marking done, ALL gates in [CLAUDE.md](../CLAUDE.md) § Chunk DONE ✅ must pass** — CPU gates (ruff, ty, pytest, C901), size/complexity, circular deps, dead-code review, documentation, config-file sync, objective gate (pipeline EXIT 0 + golden snapshot EXIT 0), platform testing. Do not re-list them here; CLAUDE.md is authoritative.

**If any gate fails, the feature is NOT done.** Surface the failure honestly: "CPU ✅, docs ❌ (AI Systems table)" or "objective ❌ (pipeline EXIT 1)".

## 4. Feature PR Template

**Commit message format:**

```
[FEATURE] <short title>: <1 line description>

Design Doc: <link to 1-page architecture.md>

Modality: <video|image|audio|document|orchestration>
Stage(s): <list of affected stages or "new stage">
CSVs/Sidecars: <list of new outputs or "none">

Gates:
  ✅ Size: <file sizes in lines>
  ✅ CPU: ruff, ty, pytest, C901
  ✅ Circular deps: 0
  ✅ Objective: pipeline EXIT 0, golden EXIT 0
  ✅ Platform: Windows + WSL tested

Testing:
  - Fixture: <result>
  - Full dataset: <result or "deferred">
```

## 5. Feature Review Checklist (For Contributors)

If a contributor submits a PR with a new feature:

- [ ] Does the PR include the design doc link?
- [ ] Are all gates listed in the commit message and marked ✅?
- [ ] If gates are ❌, has the contributor explained why they deferred?
- [ ] Is README + AI Systems table updated?
- [ ] Is the feature resumable (`.done` sentinel or CSV dedup)?
- [ ] Are new sidecars schema-validated at write time?

**If any of the above is missing, request changes before merging.**

## 6. Deferred vs Blocked

**Deferred (acceptable with justification):**
- Full dataset testing (if fixture passes + logic is sound)
- macOS testing (if Windows + WSL pass + code is portable)
- New sidecar schema (if design is sound + will be added in follow-up PR)

**Blocked (NOT acceptable):**
- CPU gates failing (ruff, ty, pytest, C901)
- Objective gate failing (pipeline EXIT ≠ 0 or golden EXIT ≠ 0)
- Dead-code not reviewed
- Documentation out of sync
- No design doc + clear scope

---

**Summary:** Feature request → pre-impl Q&A → design doc → implementation → gates (CLAUDE.md) → PR with checklist → review → merge.