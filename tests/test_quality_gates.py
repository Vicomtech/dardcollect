"""CPU-only tests for scripts/quality_gates.py (the code-quality ratchet).

The ratchet must fail on a NEW or WORSENED violation and only note resolved or
improved ones, so the baseline stays user-owned debt instead of a silent floor.
The comparison is pure (dicts in, errors/notes out), so these tests are hermetic;
one test exercises the real collector to prove the wiring runs end-to-end.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def _load():
    path = REPO_ROOT / "scripts" / "quality_gates.py"
    spec = importlib.util.spec_from_file_location("quality_gates", path)
    if spec is None or spec.loader is None:  # pragma: no cover - import machinery
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["quality_gates"] = module
    spec.loader.exec_module(module)
    return module


qg = _load()


def test_new_violation_is_an_error():
    errors, notes = qg.compare({"c901": {"a.py|f": 12}}, {"c901": {}})
    assert len(errors) == 1
    assert "NEW c901" in errors[0]
    assert notes == []


def test_worsened_violation_is_an_error():
    errors, _ = qg.compare({"fnlen": {"a.py|f": 120}}, {"fnlen": {"a.py|f": 90}})
    assert len(errors) == 1
    assert "WORSENED fnlen" in errors[0] and "90 -> 120" in errors[0]


def test_equal_violation_is_not_an_error():
    errors, notes = qg.compare({"c901": {"a.py|f": 14}}, {"c901": {"a.py|f": 14}})
    assert errors == [] and notes == []


def test_resolved_and_improved_are_notes_not_errors():
    errors, notes = qg.compare(
        {"c901": {"a.py|f": 9}},
        {"c901": {"a.py|f": 12, "b.py|g": 11}},
    )
    assert errors == []
    assert any("resolved" in n for n in notes)
    assert any("improved 12 -> 9" in n for n in notes)


def test_metric_absent_from_baseline_with_no_current_is_clean():
    errors, notes = qg.compare({}, {})
    assert errors == [] and notes == []


def test_every_bugbear_code_maps_to_the_metric():
    """The whole `B` family is covered, not just the codes present in the baseline.

    Regression guard: mapping only B007/B905 let a genuinely new antipattern
    (e.g. B006 mutable default) pass silently while the gate claimed to cover `B`.
    """
    assert qg._metric_for_code("B006") == "bugbear"
    assert qg._metric_for_code("B008") == "bugbear"
    assert qg._metric_for_code("B905") == "bugbear"
    assert qg._metric_for_code("C901") == "c901"
    assert qg._metric_for_code("E501") is None


def test_arg_codes_map_to_unusedarg():
    """Every `ARG` code is covered (the unused-parameter category)."""
    assert qg._metric_for_code("ARG001") == "unusedarg"
    assert qg._metric_for_code("ARG002") == "unusedarg"
    assert qg._metric_for_code("ARG005") == "unusedarg"


def test_unused_arg_name_is_extracted_from_the_message():
    assert qg._unused_arg_name("Unused function argument: `count`") == "count"
    assert qg._unused_arg_name("Unused method argument: `path`") == "path"
    assert qg._unused_arg_name("Unused lambda argument: `k`") == "k"
    assert qg._unused_arg_name("some other message") == ""


def test_arg_key_is_function_and_argument(tmp_path, monkeypatch):
    """An unused-argument key is `<function>(<arg>)`, stable across line shifts.

    Regression guard: ARG was not selected at all, so a dead parameter
    (`reorganize_for_fair`'s `schema_type`, `poser.mode`, ...) went unnoticed.
    """
    repo = tmp_path
    (repo / "mod.py").write_text("def takes(used, dead):\n    return used\n", encoding="utf-8")
    monkeypatch.setattr(qg, "RUFF_SELECT", "ARG")
    findings = qg._ruff_findings(repo)
    assert "mod.py|takes(dead)" in findings.get("unusedarg", {}), findings


def test_bugbear_key_is_the_enclosing_function_not_the_line(tmp_path):
    """A bugbear key names the enclosing function, so a line shift does not move it.

    Regression guard: keying on `line <n>: <text>` made an unrelated edit above a
    baselined hit report a spurious NEW violation and a bogus "resolved" note.
    """
    src = (
        "def inner(x):\n"
        "    return zip(x, x)\n"  # B905 on line 2, inside inner()
        "\n"
        "def outer():\n"
        "    return 1\n"
    )
    path = tmp_path / "mod.py"
    path.write_text(src, encoding="utf-8")
    assert qg._enclosing_symbol(path, 2) == "inner"

    shifted = "# a comment above everything\n" + src
    path.write_text(shifted, encoding="utf-8")
    assert qg._enclosing_symbol(path, 3) == "inner"  # same key after the shift


def test_enclosing_symbol_deepest_function_wins(tmp_path):
    src = "def outer():\n    def inner():\n        return 1\n    return inner\n"
    path = tmp_path / "nested.py"
    path.write_text(src, encoding="utf-8")
    assert qg._enclosing_symbol(path, 3) == "inner"


def test_collect_runs_end_to_end_on_the_repo():
    """The collector must run against the real tree (ruff + vulture wired)."""
    current = qg.collect(REPO_ROOT)
    assert isinstance(current, dict)
    # The repository has known debt, so at least one metric must be present.
    assert current, "collector returned no metrics; the tools are probably not running"
    flat_keys = {k for bucket in current.values() for k in bucket}
    assert all("|" in k for k in flat_keys)


def test_baseline_matches_current_tree():
    """With no code change, the committed baseline must not report errors."""
    current = qg.collect(REPO_ROOT)
    errors, _ = qg.compare(current, qg.load_baseline(REPO_ROOT))
    assert errors == [], errors
