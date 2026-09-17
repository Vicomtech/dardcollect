"""CPU-only tests for scripts/validate_harness.py structural checks.

The harness validator turns judgment-only rules (links resolve, harness files
exist, god-file ratchet, no retired-harness residue) into mechanical gates.
These tests exercise each check function against a synthetic repo tree via
monkeypatched REPO_ROOT so the suite stays fast and hermetic.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


def _load_validator():
    validator_path = Path(__file__).resolve().parent.parent / "scripts" / "validate_harness.py"
    spec = importlib.util.spec_from_file_location("validate_harness", validator_path)
    if spec is None or spec.loader is None:  # pragma: no cover - import machinery
        raise ImportError(f"cannot load scripts/validate_harness.py from {validator_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["validate_harness"] = module
    spec.loader.exec_module(module)
    return module


vh = _load_validator()


def _make_repo(tmp_path: Path, *, kilo_config: bool = True) -> None:
    """Build a minimal harness layout the validator expects."""
    (tmp_path / "AGENTS.md").write_text(
        "`refactor-to-objective` skill is authoritative.\n", encoding="utf-8"
    )
    (tmp_path / "README.md").write_text("# t\n[docs](docs/6-HARNESS.md)\n", encoding="utf-8")
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "6-HARNESS.md").write_text("# harness\n", encoding="utf-8")
    (tmp_path / "docs" / "HARNESS_RULES.md").write_text("# rules\n", encoding="utf-8")
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts" / "cycle_metrics.py").write_text("# metrics\n", encoding="utf-8")
    (tmp_path / "MEMORY.md").write_text("# session state\n", encoding="utf-8")
    (tmp_path / ".kilo" / "skills" / "refactor-to-objective").mkdir(parents=True)
    (tmp_path / ".kilo" / "skills" / "refactor-to-objective" / "SKILL.md").write_text(
        "---\nname: refactor-to-objective\n---\n", encoding="utf-8"
    )
    for name in ("keep-docs-navigable",):
        (tmp_path / ".kilo" / "skills" / name).mkdir(parents=True)
        (tmp_path / ".kilo" / "skills" / name / "SKILL.md").write_text(
            f"---\nname: {name}\n---\n", encoding="utf-8"
        )
    (tmp_path / ".kilo" / "command").mkdir()
    (tmp_path / ".kilo" / "command" / "refactor-loop.md").write_text("x", encoding="utf-8")
    (tmp_path / ".kilo" / "FEATURE_WORKFLOW.md").write_text("x", encoding="utf-8")
    (tmp_path / ".kilo" / ".gitignore").write_text(
        "agent-manager.json\nworktrees/\n__pycache__/\n", encoding="utf-8"
    )
    if kilo_config:
        (tmp_path / "kilo.json").write_text("{}", encoding="utf-8")


@pytest.fixture()
def repo(tmp_path, monkeypatch):
    _make_repo(tmp_path)
    monkeypatch.setattr(vh, "REPO_ROOT", tmp_path)
    return tmp_path


def test_all_checks_pass_on_clean_repo(repo):
    assert vh.main() == 0


def test_broken_markdown_link_is_reported_with_remediation(repo, monkeypatch):
    doc = repo / "docs" / "6-HARNESS.md"
    doc.write_text("[x](missing.md)\n", encoding="utf-8")
    errors = vh._check_markdown_links()
    assert any("missing.md" in e and "fix" in e for e in errors)


def test_missing_harness_file_names_the_file(repo, monkeypatch):
    (repo / "kilo.json").unlink()
    errors = vh._check_harness_files()
    assert any("kilo.json" in e for e in errors)


def test_agents_skill_reference_points_at_existing_skill(repo):
    assert vh._check_agent_skills_reference() == []
    (repo / "AGENTS.md").write_text("`no-such-skill` skill.\n", encoding="utf-8")
    errors = vh._check_agent_skills_reference()
    assert any("no-such-skill" in e for e in errors)


def test_residue_file_and_reference_are_flagged(repo):
    (repo / "docs" / "other.md").write_text("read CLAUDE.md\n", encoding="utf-8")
    errors = vh._check_residue()
    assert any("other" in e and "claude" in e.lower() for e in errors)
    (repo / "CLAUDE.md").write_text("x", encoding="utf-8")
    errors = vh._check_residue()
    assert any("claude" in e.lower() for e in errors)


def test_god_file_over_hard_cap_fails_with_remediation(repo, monkeypatch):
    py = repo / "dardcollect"
    py.mkdir()
    big = py / "big.py"
    big.write_text("\n" * 601, encoding="utf-8")
    errors = vh._check_god_files()
    assert any("big.py" in e and "600" in e for e in errors)
    # baseline-grown file reports growth; shrunk file advises lowering
    tracked = py / "tracked.py"
    tracked.write_text("\n" * 11, encoding="utf-8")
    monkeypatch.setattr(vh, "GOD_FILE_BASELINES", {"dardcollect/tracked.py": 10})
    errors = vh._check_god_files()
    assert any("tracked.py" in e and "baseline 10" in e for e in errors)
    tracked.write_text("\n" * 5, encoding="utf-8")
    errors = vh._check_god_files()
    assert not any("tracked.py" in e for e in errors)


def test_launch_json_stale_program_path_is_flagged(repo):
    vscode = repo / ".vscode"
    vscode.mkdir()
    (vscode / "launch.json").write_text(
        '{"configurations": [{"name": "s", "program": "scripts/gone.py"}]}',
        encoding="utf-8",
    )
    errors = vh._check_launch_json()
    assert any("scripts/gone.py" in e for e in errors)


def test_kilo_gitignore_missing_local_state_exclusion(repo):
    (repo / ".kilo" / ".gitignore").write_text("worktrees/\n", encoding="utf-8")
    errors = vh._check_kilo_config()
    assert any("agent-manager.json" in e for e in errors)


def test_residue_check_skips_harness_self_documentation(repo):
    (repo / "AGENTS.md").write_text("validator rejects Claude/Copilot residue\n", encoding="utf-8")
    assert vh._check_residue() == []


def test_session_state_missing_detected_in_patched_repo(repo):
    """Regression (2026-09-16): check 8 must inspect the monkeypatched REPO_ROOT,
    not an ambient repo — an import-time SESSION_STATE constant leaked the real
    machine's MEMORY.md into these tests (green on machines with a root
    MEMORY.md, red on fresh clones even though the fixture repo has its own)."""
    (repo / "MEMORY.md").unlink()
    errors = vh._check_session_state_size()
    assert any("missing session handoff" in e for e in errors)


def test_session_state_over_budget_detected_in_patched_repo(repo):
    (repo / "MEMORY.md").write_text("x" * (vh.SESSION_STATE_MAX_BYTES + 1), encoding="utf-8")
    errors = vh._check_session_state_size()
    assert any("too large" in e for e in errors)


def test_main_returns_1_and_prints_remediation_on_failure(repo, monkeypatch, capsys):
    (repo / "kilo.json").unlink()
    rc = vh.main()
    out = capsys.readouterr().out
    assert rc == 1
    assert "re-run" in out


def test_privacy_scan_flags_home_dir_path_in_docs(repo):
    (repo / "docs" / "6-HARNESS.md").write_text("run from /home/lunzueta/repo\n", encoding="utf-8")
    assert any("home-directory" in w for w in vh._check_privacy_scan())
    (repo / "docs" / "6-HARNESS.md").write_text(
        "run from C:\\Users\\lunzueta\\repo\n", encoding="utf-8"
    )
    assert any("home-directory" in w for w in vh._check_privacy_scan())
    (repo / "docs" / "6-HARNESS.md").write_text("clean\n", encoding="utf-8")
    assert vh._check_privacy_scan() == []


def test_component_docs_flags_unnamed_pipeline_stage(repo):
    pipeline = repo / "pipeline"
    pipeline.mkdir()
    (pipeline / "stage_a.py").write_text("x", encoding="utf-8")
    # Not named anywhere -> flagged
    errors = vh._check_component_docs()
    assert any("stage_a.py" in e for e in errors)
    # Named in docs -> clean
    (repo / "docs" / "6-HARNESS.md").write_text("pipeline/stage_a.py runs\n", encoding="utf-8")
    assert vh._check_component_docs() == []
    # Or named in launch.json -> clean
    (repo / "docs" / "6-HARNESS.md").write_text("# harness\n", encoding="utf-8")
    vscode = repo / ".vscode"
    vscode.mkdir()
    (vscode / "launch.json").write_text(
        '{"configurations": [{"name": "s", "program": "pipeline/stage_a.py"}]}',
        encoding="utf-8",
    )
    assert vh._check_component_docs() == []


def test_warnings_do_not_fail_validation_but_set_exit_2(repo):
    (repo / "docs" / "6-HARNESS.md").write_text("see /home/lunzueta/repo\n", encoding="utf-8")
    rc = vh.main()
    assert rc == 2


def test_check_mode_quiet_on_warnings(repo, capsys):
    (repo / "docs" / "6-HARNESS.md").write_text("see /home/lunzueta/repo\n", encoding="utf-8")
    rc = vh.main(["--check"])
    assert rc == 2
    captured = capsys.readouterr()
    assert captured.out == ""  # quiet: warnings print nothing in --check mode


def test_check_mode_returns_1_and_writes_stderr_on_errors(repo, capsys):
    (repo / "kilo.json").unlink()
    rc = vh.main(["--check"])
    assert rc == 1
    captured = capsys.readouterr()
    assert "kilo.json" in captured.err
    assert captured.out == ""
