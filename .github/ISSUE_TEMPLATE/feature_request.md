# Feature Request: [Title]

## Description

**What problem does this solve or what capability does it add?**

[Replace this with a clear 1-2 sentence description of the feature.]

## Proposed Solution

**What specifically should be implemented?**

- [ ] Bullet 1: Clear description of first task
- [ ] Bullet 2: Clear description of second task
- [ ] Bullet 3: Clear description of third task

**Example (if applicable):**
```
Show an example of input/output or workflow here.
```

## Impact

**Which pipeline stages or modalities are affected?**

- Modality: [ ] Video [ ] Image [ ] Audio [ ] Document [ ] Orchestration
- New stage: [ ] Yes [ ] No (if yes, describe placement)
- New CSV/sidecar: [ ] Yes [ ] No (if yes, describe format)

**Does this advance the project objective?** (See [CLAUDE.md](../CLAUDE.md) § Objective)

- [ ] Yes, directly (core feature)
- [ ] Yes, indirectly (supportive)
- [ ] Orthogonal but useful

## Acceptance Criteria

**The feature is complete when:**

- [ ] All implementation tasks above are done
- [ ] Code quality gates pass (ruff, ty, pytest, C901, circular deps, dead-code)
- [ ] Documentation updated (README + sub-docs + AI Systems table if needed)
- [ ] Fixture pipeline runs end-to-end without regression (EXIT 0)
- [ ] Golden snapshot gate passes (EXIT 0, 0 hard-fail, 0 schema-invalid)
- [ ] Tested on Windows + [Linux/macOS] (specify which platform)

**For design clarification, see [.claude/FEATURE_WORKFLOW.md](../../.claude/FEATURE_WORKFLOW.md).**

## Design Notes (Optional)

*Use this space to sketch architecture, trade-offs, or open questions for implementers.*

---

**Before starting implementation, reviewers should:**
1. Ask for design doc if not provided
2. Ensure scope is clear
3. Check FAIR/schema implications
4. Confirm acceptance criteria align with project objective
