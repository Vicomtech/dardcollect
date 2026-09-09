# Harness Rules Archive

Adopted harness rules keyed by the failure or incident that motivated them (pattern from
`F:\Vicomtech\Vision Tech\ai-harness-eng\knowledge\rule_index.md`). When a session turns a lesson
into a rule, add the row here in the same cycle. A rule whose motivating failure no longer applies
is a retirement candidate, not automatic deletion.

| Rule (codified in `AGENTS.md` or `.kilo/`) | Motivating failure or incident | Date |
|---|---|---|
| Task scoping — build the full work list + batch blocking questions in one call in the first turn; a plan open question with a written recommendation is a decision default, not a user-blocking question; do not pause between chunks of a queue | A 9-issue session delivered 2 chunks, paused per "open question" that already had a written recommendation, and waited for the user twice before executing the remaining 7 — the user's "solve everything" request was read as 2 items instead of the whole queue | 2026-09-09 |
| Session closure — update `MEMORY.md` + log the cycle every session | No durable memory existed in this repo: lessons (user prefs, runner flags, quirk workarounds) died with each session; the ai-harness-eng session_state pattern was reviewed but only partially adopted until the user asked why it wasn't done | 2026-09-09 |
| Memory budget — `MEMORY.md` stays under ~40 KB (fatal gate in validate_harness.py) | The ai-harness-eng origin saw its equivalent state file grow append-only to ~99 KB (~25K tokens of fixed per-session context) before the gate existed; adopted here to prevent the same regression | 2026-09-09 |
| Rules keyed by motivating failure (this archive) | Lessons lived only in session chat; every session that ignored one re-produced the same defect later (ai-harness-eng: the deliverable-edit lessons pattern) | 2026-09-09 |
| Golden drift policy — GPU non-determinism is informational; intended behavior changes re-capture the baseline and the user ratifies | Byte-exact goldens are impossible with TensorRT/CUDA; treating drift as failure would block every legitimate chunk (AGENTS.md § Objective verification) | 2026-09-08 |
| Test runner `uv run --no-sync` + ASCII-only script output on Windows | Plain `uv run` re-resolves torch and hangs minutes; non-ASCII print output crashes on cp1252 consoles | 2026-09-08 |
| Fake subprocess tools in tests use `.bat` + patched `_ffmpeg_exe`, never fake `.exe` | Windows CreateProcess requires a real PE binary for `.exe` (WinError 216) — content-scripted fakes must use `.bat` and module-attr patching | 2026-09-09 |