"""CPU-only tests for scripts/validate_harness.py structural checks.

The harness validator turns judgment-only rules (links resolve, harness files
exist, god-file ratchet, no retired-harness residue) into mechanical gates.
These tests exercise each check function against a synthetic repo tree via
monkeypatched REPO_ROOT so the suite stays fast and hermetic.
"""

from __future__ import annotations

import importlib.util
import subprocess
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

# Drive-absolute fixtures are assembled from parts, never written literally:
# this test module is itself tracked and scanned, so a literal drive path here
# would be flagged by the very check it exercises (the self-reference the
# production module avoids the same way). `_B` is one backslash, `_S` one
# forward slash.
_B = chr(92)
_S = "/"
_CU = "C:" + _B + "Users" + _B + "jdoe"
_FW = "F:" + _B + "Work" + _B + "secret-project"
_HOME = _S + "home" + _S + "jdoe"
_FAKE_HOME = _S + "home" + _S + "you"


def _drive_path(*parts: str) -> str:
    """Build a drive-absolute literal from parts (avoids self-reference)."""
    return _FW + _B + _B.join(parts)


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
    (tmp_path / "scripts" / "privacy_scan.py").write_text("# privacy scan\n", encoding="utf-8")
    (tmp_path / "scripts" / "quality_gates.py").write_text("# quality gates\n", encoding="utf-8")
    # Empty quality baseline: the synthetic repo has no violations, so the
    # code-quality ratchet check passes (it is a real gate, not a stub).
    (tmp_path / "scripts" / "quality_baselines.json").write_text("{}\n", encoding="utf-8")
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


def test_main_returns_1_and_prints_remediation_on_failure(repo, monkeypatch, capsys):
    (repo / "kilo.json").unlink()
    rc = vh.main()
    out = capsys.readouterr().out
    assert rc == 1
    assert "re-run" in out


def test_privacy_scan_flags_real_home_dir_path(repo):
    """A real-looking user name in a home path is a hit (2026-09-22)."""
    (repo / "docs" / "6-HARNESS.md").write_text(f"run from {_HOME}{_B}repo\n", encoding="utf-8")
    assert any("home-directory" in w for w in vh._check_privacy_scan())
    (repo / "docs" / "6-HARNESS.md").write_text(f"run from {_CU}{_B}repo\n", encoding="utf-8")
    assert any("home-directory" in w for w in vh._check_privacy_scan())
    (repo / "docs" / "6-HARNESS.md").write_text("clean\n", encoding="utf-8")
    assert vh._check_privacy_scan() == []


def test_privacy_scan_flags_drive_absolute_path_any_root(repo):
    """NEGATIVE TEST for the 2026-09-22 leak: the class that was missed.

    The committed leak was a drive-absolute path under a non-home root, in
    prose. The old scan matched only the Windows `Users` form and only
    README/AGENTS/docs, so it passed. This test fails if that hole returns.
    """
    leak = _drive_path("knowledge", "rule_index.md")
    (repo / "docs" / "6-HARNESS.md").write_text(f"see the rules in {leak}\n", encoding="utf-8")
    hits = vh._check_privacy_scan()
    assert any("drive-absolute" in w for w in hits), hits


def test_privacy_scan_covers_files_outside_docs(repo):
    """NEGATIVE TEST for the SCOPE hole: a hit in tests/ must be found.

    The committed leak embedded a user name in tests/test_validate_harness.py,
    which the scan never read.
    """
    (repo / "tests").mkdir(exist_ok=True)
    (repo / "tests" / "test_thing.py").write_text(f'PATH = "{_CU}{_B}repo"\n', encoding="utf-8")
    hits = vh._check_privacy_scan()
    assert any("test_thing.py" in w for w in hits), hits


def test_privacy_scan_flags_utf16_encoded_text_file(repo):
    """NEGATIVE TEST for the ENCODING hole (2026-09-22).

    A tracked UTF-16 `*.txt` (the class `temp_pipeline_output.txt` belonged to)
    hid its drive-absolute paths from a UTF-8-only read. The scan now decodes
    UTF-16, so the leak inside is found.
    """
    leak = _drive_path("knowledge", "rule_index.md")
    (repo / "docs" / "6-HARNESS.md").unlink()
    (repo / "notes.txt").write_text(f"see {leak}\n", encoding="utf-16")
    hits = vh._check_privacy_scan()
    assert any("drive-absolute" in w for w in hits), hits


def test_privacy_scan_reports_unscannable_text_file(repo):
    """A text-suffixed file that cannot be decoded is reported, not skipped."""
    (repo / "docs" / "6-HARNESS.md").unlink()
    (repo / "blob.txt").write_bytes(b"\x00\x01\x02\xff\xfe\x00binary\x00\x00")
    hits = vh._check_privacy_scan()
    assert any("unscannable" in w and "blob.txt" in w for w in hits), hits


def test_privacy_scan_allows_documented_examples_and_fixtures(repo):
    """The allowances must keep legitimate literals quiet, or the gate is noise.

    Each line here is a real literal from this repository (2026-09-22) that does
    not identify a machine: documented example roots, platform-invariant vendor
    install directories, placeholders, bare-root fixture tokens, and synthetic
    user names. Assembled from parts for the same self-reference reason.
    """
    legit = [
        'root: "C:/data/DARD"',
        "tensorrt_lib: 'C:" + _B * 2 + "TensorRT-10.14.1.48" + _B * 2 + "lib'",
        "cuda_bin: 'C:"
        + _B * 2
        + "Program Files"
        + _B * 2
        + "NVIDIA GPU Computing Toolkit"
        + _B * 2
        + "CUDA"
        + _B * 2
        + "v12.1"
        + _B * 2
        + "bin'",
        "use Z:" + _B * 2 + "... paths",
        '{"data_root": "C:/First", "use_server_proxy": true}',
        'write_text("run from C:' + _B * 2 + "Users" + _B * 2 + "testuser" + _B * 2 + 'repo")',
        "run from " + _FAKE_HOME + "/project",
    ]
    (repo / "docs" / "6-HARNESS.md").write_text("\n".join(legit) + "\n", encoding="utf-8")
    hits = vh._check_privacy_scan()
    assert hits == [], hits


def test_compat_marker_is_flagged(repo):
    """A backward-compat shim marker in tracked Python code is an error."""
    py = repo / "dardcollect"
    py.mkdir()
    (py / "shim.py").write_text(
        "from new_module import X  # kept for compatibility\n", encoding="utf-8"
    )
    errors = vh._check_compat_markers()
    assert any("shim.py" in e and "compatibility" in e for e in errors)


def test_compat_marker_allowlist_pins_legitimate_use(repo, monkeypatch):
    """An allowlisted line is not flagged (real contract, user-confirmed)."""
    py = repo / "dardcollect"
    py.mkdir()
    line = "PUBLIC_ALIAS = object()  # legacy name, documented in 5-LIBRARY-API"
    (py / "pub.py").write_text(line + "\n", encoding="utf-8")
    assert any("pub.py" in e for e in vh._check_compat_markers())
    monkeypatch.setattr(vh, "COMPAT_ALLOWLIST", {("dardcollect/pub.py", line): "reason"})
    assert vh._check_compat_markers() == []


def test_compat_check_clean_when_no_markers(repo):
    py = repo / "dardcollect"
    py.mkdir()
    (py / "clean.py").write_text("import os\nVALUE = 1\n", encoding="utf-8")
    assert vh._check_compat_markers() == []


def _git(cwd: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True)


def test_compat_check_scans_tracked_path_with_space(tmp_path, monkeypatch):
    """A tracked .py whose path contains a space is scanned, not silently skipped.

    Regression guard: the first version split `git ls-files` output on
    whitespace, so `a b.py` became two tokens that resolved to no file and was
    dropped — the check claimed full coverage while skipping it.
    """
    monkeypatch.setattr(vh, "REPO_ROOT", tmp_path)
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "config", "user.email", "t@example.invalid")
    _git(tmp_path, "config", "user.name", "t")
    py = tmp_path / "dardcollect"
    py.mkdir()
    (py / "has space.py").write_text("X = 1  # kept for compatibility\n", encoding="utf-8")
    _git(tmp_path, "add", "-A")

    errors = vh._check_compat_markers()
    assert any("has space.py" in e for e in errors), errors


def test_compat_check_scans_tracked_non_ascii_path(tmp_path, monkeypatch):
    """A tracked .py whose path is non-ASCII is scanned, not silently skipped.

    Regression guard: `subprocess.run(text=True)` decodes git output with the
    locale codec (cp1252 on Windows), so `café.py` became a mojibake string
    that resolved to no file and was dropped. The walker must decode UTF-8.
    """
    monkeypatch.setattr(vh, "REPO_ROOT", tmp_path)
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "config", "user.email", "t@example.invalid")
    _git(tmp_path, "config", "user.name", "t")
    py = tmp_path / "dardcollect"
    py.mkdir()
    (py / "caf\u00e9.py").write_text("X = 1  # kept for compatibility\n", encoding="utf-8")
    _git(tmp_path, "add", "-A")

    errors = vh._check_compat_markers()
    assert any("caf\u00e9.py" in e for e in errors), errors


def test_compat_check_excludes_local_and_vendored_state(tmp_path, monkeypatch):
    """Markers under .venv/ (never published) do not fail the gate."""
    monkeypatch.setattr(vh, "REPO_ROOT", tmp_path)
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "config", "user.email", "t@example.invalid")
    _git(tmp_path, "config", "user.name", "t")
    vendored = tmp_path / ".venv" / "lib"
    vendored.mkdir(parents=True)
    (vendored / "third_party.py").write_text("Y = 1  # kept for compatibility\n", encoding="utf-8")
    _git(tmp_path, "add", "-Af")

    assert vh._check_compat_markers() == []


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
    (repo / "docs" / "6-HARNESS.md").write_text(f"see {_HOME}{_B}repo\n", encoding="utf-8")
    rc = vh.main()
    assert rc == 2


def test_check_mode_quiet_on_warnings(repo, capsys):
    (repo / "docs" / "6-HARNESS.md").write_text(f"see {_HOME}{_B}repo\n", encoding="utf-8")
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
