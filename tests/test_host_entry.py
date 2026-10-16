"""Host entry files and portable vocabulary (validate_harness checks 17-18).

Each test breaks one real defect and asserts the gate names it, so a green
validator run cannot hide a Claude Code entry point that stopped importing
AGENTS.md or a portable layer that started naming a host-only tool.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from harness_fixture import _write_skills, vh

GIT_DENY = ["Bash(git add:*)", "Bash(git commit:*)", "Bash(git push:*)"]


def _write_claude(root: Path, *, import_line: str = "@AGENTS.md", deny=None) -> None:
    (root / "CLAUDE.md").write_text(f"{import_line}\n\n# Claude notes\n", encoding="utf-8")
    (root / ".claude").mkdir(exist_ok=True)
    settings = {
        "permissions": {"allow": ["Bash(uv:*)"], "deny": GIT_DENY if deny is None else deny}
    }
    (root / ".claude" / "settings.json").write_text(json.dumps(settings), encoding="utf-8")


@pytest.fixture
def root(tmp_path: Path) -> Path:
    (tmp_path / "AGENTS.md").write_text("Shared contract.\n", encoding="utf-8")
    return tmp_path


def test_host_entry_passes_when_claude_imports_and_denies_git(root):
    _write_claude(root)
    assert vh.harness_extra.check_host_entry_files(root) == []


def test_missing_claude_md_names_the_file(root):
    _write_claude(root)
    (root / "CLAUDE.md").unlink()
    errors = vh.harness_extra.check_host_entry_files(root)
    assert any("missing host entry file: CLAUDE.md" in e for e in errors)


def test_claude_md_without_import_is_reported(root):
    _write_claude(root, import_line="Read AGENTS.md first")
    errors = vh.harness_extra.check_host_entry_files(root)
    assert any("must import the shared contract" in e for e in errors)


def test_commented_out_import_does_not_count(root):
    _write_claude(root, import_line="# @AGENTS.md")
    errors = vh.harness_extra.check_host_entry_files(root)
    assert any("must import the shared contract" in e for e in errors)


@pytest.mark.parametrize("missing", GIT_DENY)
def test_each_git_write_must_be_denied_in_claude_settings(root, missing):
    _write_claude(root, deny=[d for d in GIT_DENY if d != missing])
    errors = vh.harness_extra.check_host_entry_files(root)
    assert any(f"must deny '{missing}'" in e for e in errors)


def test_invalid_claude_settings_json_is_reported(root):
    _write_claude(root)
    (root / ".claude" / "settings.json").write_text("{not json", encoding="utf-8")
    errors = vh.harness_extra.check_host_entry_files(root)
    assert any("invalid JSON in .claude/settings.json" in e for e in errors)


def test_portable_vocabulary_clean_on_neutral_text(root):
    _write_skills(root)
    assert vh.harness_extra.check_portable_vocabulary(root) == []


@pytest.mark.parametrize("token", list(vh.harness_extra.HOST_ONLY_TOKENS))
def test_host_only_tool_in_agents_md_is_reported(root, token):
    _write_skills(root)
    (root / "AGENTS.md").write_text(f"Ask with {token} first.\n", encoding="utf-8")
    errors = vh.harness_extra.check_portable_vocabulary(root)
    assert any(token in e and "AGENTS.md" in e for e in errors)


def test_host_only_tool_in_a_skill_is_reported(root):
    _write_skills(root)
    skill = root / ".agents" / "skills" / "feature-intake" / "SKILL.md"
    skill.write_text(
        "---\nname: feature-intake\ndescription: x\n---\nUse AskUserQuestion.\n", encoding="utf-8"
    )
    errors = vh.harness_extra.check_portable_vocabulary(root)
    assert any("feature-intake/SKILL.md" in e and "AskUserQuestion" in e for e in errors)
