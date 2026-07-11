---
applyTo: "feature-requests"
---

# Feature Request Workflow — Claude/Copilot Protocol

**When evaluating a feature request, follow this checklist before implementation.**

## 1. Feature Intake (Pre-Implementation)

**Read the request and ask these questions:**

- [ ] **Scope clarity**: Is the request clearly scoped? If vague, request clarification in a Q&A format, don't guess.
- [ ] **Objective alignment**: Does it advance the project objective (§ CLAUDE.md Objective)? Or is it orthogonal/supportive?
- [ ] **FAIR compliance**: Does it generate new CSVs/sidecars? If yes, verify JSON schema exists and is documented.
- [ ] **Modality**: Which pipeline modality does it belong to (video/image/audio/document)?
- [ ] **Stage placement**: Which of the 11 stages should it hook into, or is it a new stage?
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

## 3. Implementation Checklist

**Before marking done, ALL of these must pass:**

| Gate | Rule |
|---|---|
| **Size gate** | ✅ No god-files > 600 lines; functions ≤ 80 lines |
| **CPU gates** | ✅ `ruff check .` + `ruff format --check .` + `ty check` + `pytest` all green |
| **C901** | ✅ ≤ 20 (target ≤ 10); not increased from start |
| **Circular deps** | ✅ `codebase_graph_circular` = 0 |
| **Dead-code** | ✅ Unused imports/functions reviewed + pruned |
| **Documentation** | ✅ README + AI Systems table updated if behavior changed |
| **Config file sync** | ✅ `.vscode/launch.json`, `pyproject.toml`, `.github/instructions/` checked |
| **Objective gate** | ✅ `scripts/run_pipeline.py --config config.test.yaml` EXIT 0 |
| **Golden snapshot** | ✅ `scripts/golden_snapshot.py` EXIT 0; 0 hard-fail, 0 schema-invalid |
| **Platform** | ✅ Tested on Windows + (WSL or Linux or macOS); documented |
| **Design doc** | ✅ 1-page architecture + trade-offs linked from commit message |

**If any gate fails, feature is NOT done. Surface the failure honestly: "CPU ✅, docs ❌ (AI Systems table)" or "objective ❌ (pipeline EXIT 1)".**

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

## Example: Video Masks Feature

**Feature request:** "Extract N frames, detect faces, save bounding box masks (255 = face, 0 = background)."

**Pre-implementation check:**
- ✅ Scope clear: extract frames → detect → generate masks
- ✅ Objective: supports face crop quality (supportive)
- ✅ FAIR: masks are outputs; should they be tracked in provenance? → design doc must address
- ✅ Modality: video
- ✅ Stage: NEW stage after extract_frames, before face_crops

**Design doc (1 page):**
- Input: extracted frames from `extract_frames_from_videos`
- Output: `<frame_name>.png.mask` (same dir as frame)
- Mask format: uint8 [0, 255], white=face/head-shoulders, black=background
- Resumability: skip if `.mask` exists
- Schema: is there a sidecar? Or just files? → design doc clarifies

**Gates before marking done:**
1. CPU ✅
2. Objective ✅ (fixture runs, masks present, no schema errors)
3. Platform ✅ (Windows tested; WSL deferred if portable)
4. README updated ✅ (new stage documented)

---

**Summary:** Feature request → pre-impl Q&A → design doc → implementation → 9 gates → PR with checklist → review → merge.
