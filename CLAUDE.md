@AGENTS.md

# Claude Code notes

Everything above is the shared project contract. The points below only matter to Claude Code.

- **Skills:** edit the canonical copy in `.agents/skills/<name>/SKILL.md`. `.claude/skills/` is the mirror Claude Code discovers; the pre-commit hook refreshes it, or run `python scripts/skill_mounts.py --sync`.
- **Permissions:** the committed allowlist and the git `deny` rules live in `.claude/settings.json`. Personal overrides go in `.claude/settings.local.json`, which is never committed.
- **Commits:** the user commits. The `deny` rules block `git add`, `git commit` and `git push`; do not work around them.
- **Memory:** read and update `MEMORY.md` at the repo root as AGENTS.md § Session closure says. Claude's own auto-memory directory is separate and does not replace it.
