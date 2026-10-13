"""CPU-only tests for the reproducibility producer block.

The block records the pipeline commit and the config hash that produced a
sidecar. Both are optional and injected by `add_fair_metadata`; the schemas must
accept the block (they are closed).
"""

import hashlib

from dardcollect import reproducibility as rp
from dardcollect.fair import add_fair_metadata, generate_uuid, validate_against_schema


def _reset_caches():
    rp.pipeline_commit.cache_clear()
    rp._config_hash.cache_clear()


def test_config_sha256_hashes_the_env_config(tmp_path, monkeypatch):
    cfg = tmp_path / "c.yaml"
    cfg.write_text("a: 1\n", encoding="utf-8")
    monkeypatch.setenv("DARDCOLLECT_CONFIG", str(cfg))
    _reset_caches()

    assert rp.config_sha256() == hashlib.sha256(b"a: 1\n").hexdigest()


def test_producer_block_uses_commit_override(monkeypatch):
    monkeypatch.setenv("DARD_PIPELINE_COMMIT", "abc1234")
    monkeypatch.delenv("DARDCOLLECT_CONFIG", raising=False)
    _reset_caches()

    block = rp.producer_block()
    assert block["pipeline_commit"] == "abc1234"
    assert "config_sha256" not in block


def test_add_fair_metadata_injects_and_validates_producer(tmp_path, monkeypatch):
    cfg = tmp_path / "c.yaml"
    cfg.write_text("x: 1\n", encoding="utf-8")
    monkeypatch.setenv("DARDCOLLECT_CONFIG", str(cfg))
    monkeypatch.setenv("DARD_PIPELINE_COMMIT", "deadbee")
    _reset_caches()

    data = add_fair_metadata(
        {
            "uuid": generate_uuid(),
            "source_video": "v.mp4",
            "start_frame": 0,
            "end_frame": 1,
            "duration_seconds": 0.1,
        },
        schema_type="person_clip",
    )

    assert data["producer"]["pipeline_commit"] == "deadbee"
    assert data["producer"]["config_sha256"] == hashlib.sha256(b"x: 1\n").hexdigest()
    assert validate_against_schema(data, "person_clip") is True
