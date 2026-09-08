"""CPU-only tests for scripts/make_test_config.py substitutions.

The fixture-gate config generator redirects both path conventions — literal
``DARD/...`` strings and ``root:`` + ``{root}/...`` templating — to
``tests/fixtures/media/`` + ``DARD_test/``. These tests pin the substitution
set against the two historical config shapes so the generated
``configs/config.test.yaml`` never goes stale/incoherent again.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "make_test_config",
        Path(__file__).resolve().parent.parent / "scripts" / "make_test_config.py",
    )
    if spec is None or spec.loader is None:  # pragma: no cover - import machinery
        raise ImportError("cannot load scripts/make_test_config.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["make_test_config"] = module
    spec.loader.exec_module(module)
    return module


mtc = _load_module()


def test_templated_paths_redirect_to_fixture_and_dard_test():
    src = (
        'root: "C:/data/DARD"\n'
        'base_output_dir: "C:/data/DARD/archive_org_public_domain"\n'
        '  input_dir: "{root}/archive_org_public_domain/videos"\n'
        '  output_clips_dir: "{root}/extracted_person_clips"\n'
        '  input_dir: "{root}/filtered_video_face_crops"\n'
    )
    out = src
    for old, new in mtc.SUBSTITUTIONS:
        out = out.replace(old, new)
    assert 'root: "DARD"' in out
    assert 'base_output_dir: "tests/fixtures/media"' in out
    assert 'input_dir: "tests/fixtures/media/videos"' in out
    assert 'output_clips_dir: "DARD_test/extracted_person_clips"' in out
    assert 'input_dir: "DARD_test/filtered_video_face_crops"' in out
    assert "C:/data" not in out
    assert "{root}" not in out


def test_literal_paths_redirect():
    src = (
        'base_output_dir: "DARD/archive_org_public_domain"\n'
        '  input_dir: "DARD/archive_org_public_domain/images"\n'
        '  output_dir: "DARD/audio_transcriptions"\n'
    )
    out = src
    for old, new in mtc.SUBSTITUTIONS:
        out = out.replace(old, new)
    assert 'base_output_dir: "tests/fixtures/media"' in out
    assert 'input_dir: "tests/fixtures/media/images"' in out
    assert 'output_dir: "DARD_test/audio_transcriptions"' in out


def test_absolute_base_output_dir_redirects_before_literal_rule_bites():
    """The absolute C:/data/DARD/... base_output_dir contains the literal
    DARD/archive_org_public_domain substring; the dedicated rule must run
    first or a C:/data/tests/... prefix leaks into the test config."""
    src = 'base_output_dir: "C:/data/DARD/archive_org_public_domain"\n'
    out = src
    for old, new in mtc.SUBSTITUTIONS:
        out = out.replace(old, new)
    assert 'base_output_dir: "tests/fixtures/media"' in out
    assert "C:/data" not in out


def test_generated_config_is_yaml_loadable_with_no_production_paths(tmp_path, monkeypatch):
    import yaml

    monkeypatch.setattr(mtc, "REPO_ROOT", tmp_path)
    configs = tmp_path / "configs"
    configs.mkdir()
    (configs / "config.archive_all.yaml").write_text(
        'root: "C:/data/DARD"\n'
        'base_output_dir: "C:/data/DARD/archive_org_public_domain"\n'
        'media_types: ["video"]\n'
        "person_extraction:\n"
        '  input_dir: "{root}/archive_org_public_domain/videos"\n'
        '  output_clips_dir: "{root}/extracted_person_clips"\n',
        encoding="utf-8",
    )
    rc = mtc.main()
    assert rc == 0
    generated = yaml.safe_load((configs / "config.test.yaml").read_text(encoding="utf-8"))
    assert generated["person_extraction"]["input_dir"] == "tests/fixtures/media/videos"
    assert generated["person_extraction"]["output_clips_dir"] == "DARD_test/extracted_person_clips"
    assert generated["base_output_dir"] == "tests/fixtures/media"
