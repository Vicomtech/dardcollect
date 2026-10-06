"""Shared fixtures for the harness-validator tests.

`scripts/validate_harness.py` is loaded by path (it is a script, not an
importable package) and a minimal synthetic harness repo is built here, so the
validator tests in `test_validate_harness.py` and the skill-mount integration
tests in `test_skill_mounts.py` share one tree instead of duplicating it. Kept
out of the test modules so each stays under the god-file cap and `build_repo`
under the function-length cap.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

_SKILL_NAMES = (
    "refactor-to-objective",
    "keep-docs-navigable",
    "feature-intake",
    "harness-self-improve",
    "originality-guard",
)
_HOSTS = ("Kilo Code", "pi", "Claude Code", "Codex", "GitHub Copilot")


def load_validator():
    """Load `scripts/validate_harness.py` by path, registering it in sys.modules."""
    validator_path = Path(__file__).resolve().parent.parent / "scripts" / "validate_harness.py"
    spec = importlib.util.spec_from_file_location("validate_harness", validator_path)
    if spec is None or spec.loader is None:  # pragma: no cover - import machinery
        raise ImportError(f"cannot load scripts/validate_harness.py from {validator_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["validate_harness"] = module
    spec.loader.exec_module(module)
    return module


vh = load_validator()


def _write_core(root: Path) -> None:
    """AGENTS.md, README, docs, and the session-state handoff."""
    (root / "AGENTS.md").write_text(
        "`refactor-to-objective` skill is authoritative.\n"
        "Layout: .agents/ .claude/ .kilo/ .kilo/command/ .vscode/ docs/ pipeline/ "
        "scripts/\n"
        "The harness runs on any supported agent host (Kilo Code, pi, Claude Code, "
        "Codex, GitHub Copilot).\n",
        encoding="utf-8",
    )
    (root / "README.md").write_text("# t\n[docs](docs/6-HARNESS.md)\n", encoding="utf-8")
    (root / "docs").mkdir()
    (root / "docs" / "6-HARNESS.md").write_text("# harness\n", encoding="utf-8")
    (root / "docs" / "HARNESS_RULES.md").write_text(
        "# rules\n\n| Rule | Failure | Date | Enforcement |\n|---|---|---|---|\n"
        "| Test rule | test incident | 2026-09-25 | advisory |\n",
        encoding="utf-8",
    )
    (root / "MEMORY.md").write_text("# session state\n", encoding="utf-8")


def _write_scripts(root: Path) -> None:
    """Script stubs plus the documented-surfaces and host-mount registries."""
    (root / "scripts").mkdir()
    for name in (
        "cycle_metrics.py",
        "privacy_scan.py",
        "quality_gates.py",
        "harness_extra.py",
        "license_scan.py",
        "diag_mutation_probe.py",
        "component_inventory.py",
        "skill_mounts.py",
        "skill_frontmatter.py",
    ):
        (root / "scripts" / name).write_text(f"# {name}\n", encoding="utf-8")
    (root / "scripts" / "host_surfaces.json").write_text(
        json.dumps(
            {
                "canonical": ".agents/skills",
                "hostMounts": {
                    "Kilo Code": [".agents/skills"],
                    "pi": [".agents/skills"],
                    "Claude Code": [".claude/skills"],
                    "Codex": [".agents/skills"],
                    "GitHub Copilot": [".agents/skills"],
                },
                "hostClaim": {
                    "doc": "AGENTS.md",
                    "start": "The harness runs on any supported agent host (",
                    "end": ").",
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    # Every depth-1 class the synthetic repo has, declared with a doc that names
    # it (token-checked by the discovery gate).
    (root / "scripts" / "documented_surfaces.json").write_text(
        json.dumps(
            {
                "roots": ["", ".kilo"],
                "surfaces": {
                    ".agents": {"doc": "AGENTS.md", "token": ".agents/"},
                    ".claude": {"doc": "AGENTS.md", "token": ".claude/"},
                    ".kilo": {"doc": "AGENTS.md", "token": ".kilo/"},
                    ".kilo/command": {"doc": "AGENTS.md", "token": ".kilo/command/"},
                    ".vscode": {"doc": "AGENTS.md", "token": ".vscode/"},
                    "docs": {"doc": "AGENTS.md", "token": "docs/"},
                    "pipeline": {"doc": "AGENTS.md", "token": "pipeline/"},
                    "scripts": {"doc": "AGENTS.md", "token": "scripts/"},
                },
                "exempt": {},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    # Empty quality baseline: the synthetic repo has no violations, so the
    # code-quality ratchet check passes (it is a real gate, not a stub).
    (root / "scripts" / "quality_baselines.json").write_text("{}\n", encoding="utf-8")


def _write_skills(root: Path) -> None:
    """Canonical `.agents/skills/` tree + the byte-identical Claude mirror."""
    for name in _SKILL_NAMES:
        body = f"---\nname: {name}\ndescription: Test skill.\n---\n"
        (root / ".agents" / "skills" / name).mkdir(parents=True)
        (root / ".agents" / "skills" / name / "SKILL.md").write_text(body, encoding="utf-8")
        (root / ".claude" / "skills" / name).mkdir(parents=True)
        (root / ".claude" / "skills" / name / "SKILL.md").write_text(body, encoding="utf-8")


def _write_kilo_and_pipeline(root: Path, *, kilo_config: bool) -> None:
    (root / ".kilo" / "command").mkdir(parents=True)
    (root / ".kilo" / "command" / "refactor-loop.md").write_text("x", encoding="utf-8")
    (root / ".kilo" / "FEATURE_WORKFLOW.md").write_text("x", encoding="utf-8")
    (root / ".kilo" / ".gitignore").write_text(
        "agent-manager.json\nworktrees/\n__pycache__/\n", encoding="utf-8"
    )
    (root / "pipeline").mkdir()
    (root / "pipeline" / "probe_stage.py").write_text("# stage\n", encoding="utf-8")
    with open(root / "docs" / "6-HARNESS.md", "a", encoding="utf-8") as fh:
        fh.write("pipeline/probe_stage.py runs\n")
    if kilo_config:
        (root / "kilo.json").write_text("{}", encoding="utf-8")


def build_repo(root: Path, *, kilo_config: bool = True) -> None:
    """Build a minimal harness layout the validator expects."""
    _write_core(root)
    _write_scripts(root)
    _write_skills(root)
    _write_kilo_and_pipeline(root, kilo_config=kilo_config)
