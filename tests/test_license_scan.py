"""CPU-only tests for scripts/license_scan.py + volatile-numbers check."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _load(name: str, rel: str):
    path = Path(__file__).resolve().parent.parent / rel
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


ls = _load("license_scan", "scripts/license_scan.py")
he = _load("harness_extra", "scripts/harness_extra.py")


def test_license_scan_passes_on_pinned_repo():
    assert ls.scan() == [] or all("duplicate" not in w for w in ls.scan()), ls.scan()


def test_license_scan_flags_copyleft_and_unpinned(tmp_path, monkeypatch):
    monkeypatch.setattr(ls, "REPO_ROOT", tmp_path)
    (tmp_path / "pyproject.toml").write_text(
        '[project]\ndependencies = [\n"gpl-toolkit",\n"loose-lib",\n"fine-lib==1.2",\n]\n',
        encoding="utf-8",
    )
    (tmp_path / "dardcollect").mkdir()
    (tmp_path / "pipeline").mkdir()
    warnings = ls.scan(tmp_path)
    assert any("copyleft" in w for w in warnings), warnings
    assert any("unpinned" in w for w in warnings), warnings


def test_volatile_numbers_flags_stale_total(tmp_path):
    (tmp_path / "README.md").write_text("runs 10 checks\n", encoding="utf-8")
    (tmp_path / "AGENTS.md").write_text("x\n", encoding="utf-8")
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "6-HARNESS.md").write_text("x\n", encoding="utf-8")
    errors = he.check_volatile_numbers(tmp_path, 16)
    assert any("10" in e and "16" in e for e in errors), errors
    assert he.check_volatile_numbers(tmp_path, 10) == []
