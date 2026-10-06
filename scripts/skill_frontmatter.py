#!/usr/bin/env python3
"""Agent Skills frontmatter contract for one `SKILL.md`.

WHY THIS EXISTS
    Adopted from the ai-harness-eng harness (2026-10-06) when the canonical skill
    tree moved to `.agents/skills/`. The harness claims five agent hosts; a
    permissive reader accepts a `SKILL.md` whose frontmatter parses locally while
    it is unreadable at every packaging boundary (the silent-loss class), so the
    offline packaging-critical subset is checked strictly here.

    This module owns the strict-YAML contract (validator check `skill
    frontmatter`): the file opens with a terminated YAML block, the block parses
    strictly, and the mapping carries the Agent Skills required fields with a
    spec-legal name. It is deliberately NOT a reimplementation of the full
    specification: the reference library (`skills-ref`, agentskills/agentskills)
    is the authority. PyYAML is already a runtime dependency of this repo, so it
    is imported directly (the portable-core variant defers this to the adopter).

USAGE
    import skill_frontmatter
    errors = skill_frontmatter.frontmatter_errors(skill_md, label)
"""

from __future__ import annotations

import importlib
from pathlib import Path
from types import ModuleType

try:
    yaml: ModuleType | None = importlib.import_module("yaml")
except ImportError:  # pragma: no cover - exercised by uninstalling PyYAML
    yaml = None

_AGENT_SKILLS_REQUIRED = ("name", "description")
_AGENT_SKILLS_ALLOWED = {
    "name",
    "description",
    "license",
    "compatibility",
    "metadata",
    "allowed-tools",
}


def _err(label: str, kind: str, message: str, fix: str) -> str:
    """Remediation-injecting error string, in the repository's message shape."""
    return f"[{kind}] {label}: {message} -> {fix}"


def name_rule_errors(name: str, dirname: str, label: str) -> list[str]:
    """Directory-match and character rules for one declared skill name."""
    errors: list[str] = []
    if name != dirname:
        errors.append(
            _err(
                label,
                "skill_name_directory_mismatch",
                f"declares name '{name}' but lives in directory '{dirname}'",
                "rename the directory or the `name` so they match (the spec "
                "requires the name to equal the parent directory name)",
            )
        )
    if (
        name != name.lower()
        or "--" in name
        or name.startswith("-")
        or name.endswith("-")
        or not all(c.isalnum() or c == "-" for c in name)
    ):
        errors.append(
            _err(
                label,
                "skill_name_invalid",
                f"declares name '{name}', which the spec does not allow",
                "use lowercase letters, digits and single hyphens only, not "
                "starting or ending with a hyphen",
            )
        )
    if len(name) > 64:
        errors.append(
            _err(
                label,
                "skill_name_too_long",
                f"name is {len(name)} characters",
                "shorten the name to 64 characters or fewer",
            )
        )
    return errors


def meta_field_errors(meta: dict, dirname: str, label: str) -> list[str]:
    """Field allowlist, required fields, name and description rules."""
    errors: list[str] = []
    extra = sorted(set(meta) - _AGENT_SKILLS_ALLOWED)
    if extra:
        errors.append(
            _err(
                label,
                "skill_frontmatter_unknown_fields",
                f"frontmatter declares field(s) the Agent Skills spec does not "
                f"define: {', '.join(extra)}",
                f"remove them or move them under `metadata` (allowed: "
                f"{', '.join(sorted(_AGENT_SKILLS_ALLOWED))})",
            )
        )
    for required in _AGENT_SKILLS_REQUIRED:
        if not meta.get(required):
            errors.append(
                _err(
                    label,
                    "skill_frontmatter_missing_field",
                    f"frontmatter has no non-empty `{required}`",
                    f"add `{required}:` with a value",
                )
            )
    name = meta.get("name")
    if isinstance(name, str) and name.strip():
        errors.extend(name_rule_errors(name.strip(), dirname, label))
    description = meta.get("description")
    if isinstance(description, str) and len(description) > 1024:
        errors.append(
            _err(
                label,
                "skill_description_too_long",
                f"description is {len(description)} characters",
                "shorten the description to 1024 characters or fewer",
            )
        )
    return errors


def frontmatter_errors(skill_md: Path, label: str) -> list[str]:
    """Strict-YAML frontmatter check for one `SKILL.md` (Agent Skills spec)."""
    errors: list[str] = []
    try:
        text = skill_md.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        errors.append(
            _err(
                label,
                "skill_frontmatter_unreadable",
                f"cannot read it: {exc}",
                "restore the file (git checkout) or remove the skill directory",
            )
        )
        return errors
    if not text.startswith("---"):
        errors.append(
            _err(
                label,
                "skill_frontmatter_missing",
                "does not open with a YAML frontmatter block",
                "add a `---` block with at least `name` and `description` as the "
                "first lines of the file",
            )
        )
        return errors
    end = text.find("\n---", 3)
    if end == -1:
        errors.append(
            _err(
                label,
                "skill_frontmatter_unterminated",
                "frontmatter is never closed with `---`",
                "close the frontmatter block with a line containing only `---`",
            )
        )
        return errors
    if yaml is None:
        errors.append(
            _err(
                label,
                "skill_frontmatter_unchecked",
                "could not be parsed: PyYAML is not importable, so a frontmatter "
                "that is invalid YAML would load here (tolerant reader) and fail "
                "at every packaging boundary",
                "install PyYAML for the Python that runs this check, then re-run",
            )
        )
        return errors
    try:
        meta = yaml.safe_load(text[4 : end + 1])
    except Exception as exc:
        first = str(exc).split("\n")[0]
        errors.append(
            _err(
                label,
                "skill_frontmatter_invalid_yaml",
                f"frontmatter is not valid YAML: {first}. A tolerant reader may "
                f"still load this skill, which is why the defect reaches the "
                f"packaging boundary unnoticed",
                "quote the offending value or replace the character that breaks "
                "the block (an unquoted `: ` inside a scalar is the common "
                "cause), then re-run",
            )
        )
        return errors
    if not isinstance(meta, dict):
        errors.append(
            _err(
                label,
                "skill_frontmatter_not_mapping",
                "frontmatter is not a YAML mapping",
                "make the frontmatter a `key: value` mapping with `name` and `description`",
            )
        )
        return errors
    errors.extend(meta_field_errors(meta, skill_md.parent.name, label))
    return errors
