# Development System for DARDcollect — Quick Reference

## For Users Requesting Features

1. **Open an issue** using [`.github/ISSUE_TEMPLATE/feature_request.md`](.github/ISSUE_TEMPLATE/feature_request.md)
2. **Fill out the template** — be specific about what, why, and which pipeline stages
3. **Wait for review** before starting work
4. **Reviewers will ask clarifying questions** if needed

## For Developers (Humans)

**Read these in order:**
1. [docs/CONTRIBUTING.md](docs/CONTRIBUTING.md) — code style + feature workflow overview
2. [CLAUDE.md](CLAUDE.md) — project objective, toolchain, gates
3. [docs/4-DEVELOPMENT.md](docs/4-DEVELOPMENT.md) — GPU setup, dev workflow

**Before implementing a feature:**
1. Ensure the issue is **reviewed and approved**
2. Read [`.claude/FEATURE_WORKFLOW.md`](.claude/FEATURE_WORKFLOW.md) for design doc template
3. Write a 1-page design doc answering: scope, architecture, FAIR impact, resumability, test plan
4. Share the design doc in the issue for final approval

**When implementing:**
- Follow all gates listed in [CLAUDE.md](CLAUDE.md) § Chunk DONE ✅
- Run gates before marking done: CPU + complexity + circular deps + objective + documentation
- Test on Windows + (WSL or Linux or macOS)

**Before opening a PR:**
- Commit message must list all gates with ✅ or ❌ status
- If any gate is ❌, explain why (blocked/deferred with justification)
- Link to the design doc
- All ✅ gates required to merge (except explicitly deferred + approved)

## For Claude/Copilot Agents

**When a feature request arrives:**

1. **Read [`.claude/FEATURE_WORKFLOW.md`](.claude/FEATURE_WORKFLOW.md)** — this is your protocol
2. **Pre-implementation intake:**
   - Ask clarifying questions in a Q&A format
   - Check objective alignment, FAIR implications, stage placement
   - **Do NOT start coding until scope is 100% clear**
3. **Design doc:**
   - Help user write 1-page architecture doc (if not provided)
   - Cover: problem, architecture, FAIR impact, resumability, test plan
   - Save to `.claude/designs/` (temporary, user commits final version)
4. **Implementation:**
   - Follow the 9 gates in § Chunk DONE ✅
   - Run all verifications before marking done
   - Surface gate failures honestly: "CPU ✅, docs ❌"
5. **Final commit message:**
   - Include design doc link
   - List all gates with status (✅ or ❌)
   - State platform tested (Windows + WSL, etc.)

---

## Example: Video Masks Feature

**Request:** "Extract N frames, detect faces, save bounding box masks (255=white=face, 0=black=background)."

**Your response (Claude/Copilot):**

> Thanks for the request. Before I implement, let me clarify a few things:
> 
> 1. **Input frames:** Should these come from the existing `extract_frames_from_videos` stage, or is this part of a new workflow?
> 2. **Output location:** Should masks go alongside the frames or in a dedicated output dir?
> 3. **FAIR compliance:** Are these masks tracked in a sidecar JSON, or just output files?
> 4. **Resumability:** If a video is partially processed, should we skip existing `.mask` files?
> 5. **Test plan:** Should this be a standalone pipeline stage or integrated into `extract_frames`?
> 
> Once these are clear, I'll write a 1-page design doc for your review.

**After clarification, you write the design doc and implement with all gates verified.**

---

## File Structure — Where Everything Is

```
DARDcollect/
├── CLAUDE.md                           # This file: objective, gates, working rules
├── docs/
│   ├── CONTRIBUTING.md                 # Code style + feature workflow (human-facing)
│   ├── 0-GETTING-STARTED.md           # Installation + first steps
│   ├── 1-ARCHITECTURE.md              # Pipeline design
│   └── ... (other docs)
├── .claude/
│   ├── FEATURE_WORKFLOW.md            # Feature protocol for Claude/Copilot (you are here)
│   ├── skills/                        # Reusable automation patterns
│   │   ├── keep-docs-navigable/SKILL.md
│   │   ├── refactor-to-objective/SKILL.md
│   │   └── socraticode-index-first/SKILL.md
│   └── designs/                       # Temporary design docs (per-session)
├── .github/
│   ├── ISSUE_TEMPLATE/
│   │   └── feature_request.md         # Template for new features
│   └── instructions/                  # GitHub workflows + rules
└── (code + tests)
```

---

## Checklist: Is Your Feature Ready to Merge?

- [ ] Issue opened + approved by reviewer
- [ ] Design doc written + approved
- [ ] All CPU gates pass (ruff, ty, pytest, C901, circular deps, dead-code)
- [ ] Objective gate passes (pipeline EXIT 0, golden snapshot 0 hard-fail)
- [ ] Documentation updated (README, AI Systems table, sub-docs)
- [ ] Config files synced (.vscode/launch.json, pyproject.toml)
- [ ] Tested on Windows + (WSL | Linux | macOS)
- [ ] PR has design doc link + gate status in commit message
- [ ] All ✅ gates required; ❌ gates deferred with justification

**If all boxes are checked, feature is ready for merge.**
