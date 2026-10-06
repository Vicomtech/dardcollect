"""Unit tests for the agent-host skill-mount gate (`scripts/skill_mounts.py`).

The gate is the portability claim's only enforcement: every agent host the
harness claims must be wired to a real skill-discovery directory, and every
distinct mount must mirror the canonical `.agents/skills/` tree file by file.
These tests plant each defect the gate is named for (drift, a missing/extra
file, an unwired host, a prose/registry mismatch, an absent registry) and
require it to fire, plus the `--sync` repair and the frontmatter walk.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent


def _load():
    path = _REPO / "scripts" / "skill_mounts.py"
    spec = importlib.util.spec_from_file_location("skill_mounts", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


sm = _load()

_HOSTS = ["Kilo Code", "pi", "Claude Code", "Codex", "GitHub Copilot"]


def _skill(root: Path, tree: str, name: str, body: str | None = None) -> None:
    d = root / tree / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "SKILL.md").write_text(
        body or f"---\nname: {name}\ndescription: d.\n---\n", encoding="utf-8"
    )


def _write_registry(root: Path, hosts: dict | None = None, claim: list | None = None) -> Path:
    data: dict = {
        "canonical": ".agents/skills",
        "hostMounts": hosts
        if hosts is not None
        else {
            "Kilo Code": [".agents/skills"],
            "pi": [".agents/skills"],
            "Claude Code": [".claude/skills"],
            "Codex": [".agents/skills"],
            "GitHub Copilot": [".agents/skills"],
        },
    }
    if claim is not None:
        (root / "AGENTS.md").write_text(
            "This harness runs on the " + ", ".join(claim) + " agent hosts.\n",
            encoding="utf-8",
        )
        data["hostClaim"] = {
            "doc": "AGENTS.md",
            "start": "This harness runs on the ",
            "end": " agent hosts.",
        }
    (root / "scripts").mkdir(parents=True, exist_ok=True)
    path = root / "scripts" / "host_surfaces.json"
    path.write_text(json.dumps(data) + "\n", encoding="utf-8")
    return path


def _tree(tmp_path: Path) -> Path:
    for name in ("alpha", "beta"):
        _skill(tmp_path, ".agents/skills", name)
        _skill(tmp_path, ".claude/skills", name)
    _write_registry(tmp_path, claim=_HOSTS)
    return tmp_path


def test_clean_tree_is_quiet(tmp_path):
    root = _tree(tmp_path)
    assert sm.mount_errors(root / "scripts" / "host_surfaces.json", root) == []


def test_drift_in_mount_is_reported(tmp_path):
    root = _tree(tmp_path)
    (root / ".claude" / "skills" / "beta" / "SKILL.md").write_text("drifted\n", encoding="utf-8")
    errors = sm.mount_errors(root / "scripts" / "host_surfaces.json", root)
    assert any("drift" in e.lower() and "beta" in e for e in errors), errors


def test_missing_file_in_mount_is_reported(tmp_path):
    root = _tree(tmp_path)
    (root / ".claude" / "skills" / "beta" / "SKILL.md").unlink()
    errors = sm.mount_errors(root / "scripts" / "host_surfaces.json", root)
    assert any("missing" in e.lower() and "beta" in e for e in errors), errors


def test_extra_file_in_mount_is_reported(tmp_path):
    root = _tree(tmp_path)
    (root / ".claude" / "skills" / "beta" / "notes.md").write_text("x\n", encoding="utf-8")
    errors = sm.mount_errors(root / "scripts" / "host_surfaces.json", root)
    assert any("extra" in e.lower() and "notes.md" in e for e in errors), errors


def test_extra_skill_in_mount_is_reported(tmp_path):
    root = _tree(tmp_path)
    _skill(root, ".claude/skills", "orphan")
    errors = sm.mount_errors(root / "scripts" / "host_surfaces.json", root)
    assert any("orphan" in e for e in errors), errors


def test_unwired_host_is_reported(tmp_path):
    root = _tree(tmp_path)
    registry = _write_registry(root, hosts={"pi": [], "Claude Code": [".claude/skills"]})
    errors = sm.mount_errors(registry, root)
    assert any("pi" in e and "no mount" in e for e in errors), errors


def test_absolute_mount_is_refused(tmp_path):
    root = _tree(tmp_path)
    registry = _write_registry(root, hosts={"pi": ["/abs/skills"]})
    errors = sm.mount_errors(registry, root)
    assert any("absolute" in e for e in errors), errors


def test_absent_registry_fails_loud(tmp_path):
    root = _tree(tmp_path)
    errors = sm.mount_errors(root / "scripts" / "missing.json", root)
    assert any("absent" in e for e in errors), errors


def test_claim_mismatch_is_reported(tmp_path):
    root = _tree(tmp_path)
    registry = _write_registry(root, claim=[*_HOSTS, "Extra Host"])
    errors = sm.mount_errors(registry, root)
    assert any("Extra Host" in e for e in errors), errors


def test_sync_repairs_a_drifted_mount(tmp_path):
    root = _tree(tmp_path)
    mirror = root / ".claude" / "skills" / "beta" / "SKILL.md"
    mirror.write_text("drifted\n", encoding="utf-8")
    registry = root / "scripts" / "host_surfaces.json"
    assert sm.mount_errors(registry, root), "precondition: drift must be reported"
    written = sm.sync_mounts(registry, root)
    assert written == [".claude/skills"]
    assert mirror.read_text(encoding="utf-8") == (
        root / ".agents" / "skills" / "beta" / "SKILL.md"
    ).read_text(encoding="utf-8")
    assert sm.mount_errors(registry, root) == []


def test_frontmatter_walk_reports_missing_skill_md(tmp_path):
    root = _tree(tmp_path)
    (root / ".agents" / "skills" / "alpha" / "SKILL.md").unlink()
    calls: list[str] = []
    errors = sm.frontmatter_tree_errors(
        root / ".agents" / "skills",
        lambda skill_md, label: calls.append(label) or [],
        lambda p: p.relative_to(root).as_posix(),
    )
    assert any("alpha" in e and "SKILL.md" in e for e in errors), errors
    assert len(calls) == 1  # only the skill that still has a SKILL.md


def test_frontmatter_walk_calls_checker_per_skill(tmp_path):
    root = _tree(tmp_path)
    seen: list[str] = []

    def checker(_skill_md, label):
        seen.append(label)
        return []

    errors = sm.frontmatter_tree_errors(
        root / ".agents" / "skills", checker, lambda p: p.relative_to(root).as_posix()
    )
    assert errors == []
    assert sorted(seen) == [".agents/skills/alpha/SKILL.md", ".agents/skills/beta/SKILL.md"]


def test_frontmatter_walk_empty_tree_fails_loud(tmp_path):
    root = _tree(tmp_path)
    empty = root / "empty-skills"
    empty.mkdir()
    errors = sm.frontmatter_tree_errors(empty, lambda *_: [], lambda p: str(p))
    assert any("empty" in e.lower() for e in errors), errors
