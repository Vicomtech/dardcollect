"""Tests for the manipulation provenance schema + helpers.

Covers the hybrid traceability contract in docs/DESIGN_manipulation_provenance.md:
a live parent link PLUS a self-contained provenance_chain that survives loss of
the ancestor artifacts, with manipulation-over-manipulation (root_uuid /
manipulation_depth) and the external (unknown-history) edge case. Every
generated sidecar is validated against schemas/manipulation_schema.json.
"""

from dardcollect.fair import generate_uuid, validate_against_schema
from dardcollect.manipulation import (
    build_manipulation_metadata,
    cumulative_fake_regions,
    extend_provenance,
    walk_provenance,
)


def _face_crop_parent() -> dict:
    """A minimal non-manipulation parent (as an image face crop would look)."""
    return {
        "uuid": generate_uuid(),
        "schema_version": "1.0",
        "source": {"archive_org_id": "some_film_1959", "license": "public-domain"},
        "output_size": {"width": 616, "height": 616},
    }


def _mask(bbox: list[float]) -> dict:
    return {"file": "m.png", "type": "ofiq_crop_bbox", "bbox": bbox}


def _generator(prompt: str, seed: int) -> dict:
    return {"name": "sdxl-inpaint", "prompt": prompt, "seed": seed}


def test_first_manipulation_on_real_image_validates() -> None:
    parent = _face_crop_parent()
    meta = build_manipulation_metadata(
        manipulation_type="inpaint",
        generator=_generator("add glasses", 111),
        parent_metadata=parent,
        parent_type="face_crop",
        parent_file="crop.json",
        mask=_mask([10, 20, 100, 120]),
        source_image_size={"width": 616, "height": 616},
        output_size={"width": 616, "height": 616},
        label={"class": "manipulated", "fake_region": "mask"},
    )

    validate_against_schema(meta, "manipulation")
    assert meta["manipulation_depth"] == 0
    assert meta["root_uuid"] == parent["uuid"]
    assert meta["parent"]["uuid"] == parent["uuid"]
    assert meta["parent"]["type"] == "face_crop"
    assert len(meta["provenance_chain"]) == 1
    assert meta["provenance_chain"][0]["uuid"] == parent["uuid"]
    # pristine root attribution is inherited
    assert meta["source"]["license"] == "public-domain"


def test_manipulation_over_manipulation_inherits_root_and_chain() -> None:
    parent = _face_crop_parent()
    b = build_manipulation_metadata(
        manipulation_type="inpaint",
        generator=_generator("add glasses", 111),
        parent_metadata=parent,
        parent_type="face_crop",
        parent_file="crop.json",
        mask=_mask([10, 20, 100, 120]),
    )
    c = build_manipulation_metadata(
        manipulation_type="edit",
        generator=_generator("change background", 222),
        parent_metadata=b,
        parent_type="manipulation",
        parent_file="b.json",
        mask=_mask([0, 0, 616, 616]),
    )

    validate_against_schema(c, "manipulation")
    assert c["manipulation_depth"] == 1
    assert c["root_uuid"] == parent["uuid"]  # not b's uuid — the pristine root
    assert c["parent"]["uuid"] == b["uuid"]
    assert c["parent"]["type"] == "manipulation"
    # chain is self-contained: root + the intermediate manipulation B
    assert [e["uuid"] for e in c["provenance_chain"]] == [parent["uuid"], b["uuid"]]
    assert c["source"]["archive_org_id"] == "some_film_1959"


def test_deep_chain_is_self_contained_without_ancestor_files() -> None:
    parent = _face_crop_parent()
    b = build_manipulation_metadata(
        manipulation_type="inpaint",
        generator=_generator("op B", 1),
        parent_metadata=parent,
        parent_type="face_crop",
        mask=_mask([1, 1, 10, 10]),
    )
    c = build_manipulation_metadata(
        manipulation_type="edit",
        generator=_generator("op C", 2),
        parent_metadata=b,
        parent_type="manipulation",
        mask=_mask([2, 2, 20, 20]),
    )
    d = build_manipulation_metadata(
        manipulation_type="faceswap",
        generator=_generator("op D", 3),
        parent_metadata=c,
        parent_type="manipulation",
        mask=_mask([3, 3, 30, 30]),
    )

    validate_against_schema(d, "manipulation")
    assert d["manipulation_depth"] == 2
    assert d["root_uuid"] == parent["uuid"]

    # The full history is reconstructable from D alone: mask geometry and
    # generator params of B and C are embedded, not just references.
    chain = walk_provenance(d)
    assert [e["uuid"] for e in chain] == [parent["uuid"], b["uuid"], c["uuid"], d["uuid"]]
    b_entry = next(e for e in chain if e["uuid"] == b["uuid"])
    assert b_entry["op"] == "inpaint"
    assert b_entry["mask"]["bbox"] == [1, 1, 10, 10]
    assert b_entry["generator"]["prompt"] == "op B"


def test_cumulative_fake_regions_unions_all_edits() -> None:
    parent = _face_crop_parent()
    b = build_manipulation_metadata(
        manipulation_type="inpaint",
        generator=_generator("op B", 1),
        parent_metadata=parent,
        parent_type="face_crop",
        mask=_mask([1, 1, 10, 10]),
    )
    c = build_manipulation_metadata(
        manipulation_type="edit",
        generator=_generator("op C", 2),
        parent_metadata=b,
        parent_type="manipulation",
        mask=_mask([2, 2, 20, 20]),
    )

    regions = cumulative_fake_regions(c)
    assert [r["bbox"] for r in regions] == [[1, 1, 10, 10], [2, 2, 20, 20]]


def test_external_manipulation_starts_fresh_chain() -> None:
    meta = build_manipulation_metadata(
        manipulation_type="unknown",
        parent_metadata=None,
    )

    validate_against_schema(meta, "manipulation")
    assert meta["parent"] is None
    assert meta["root_uuid"] is None
    assert meta["manipulation_depth"] == 0
    assert meta["provenance"] == "external"
    assert meta["provenance_chain"] == []


def test_fair_fields_are_ordered_first() -> None:
    parent = _face_crop_parent()
    meta = build_manipulation_metadata(
        manipulation_type="inpaint",
        generator=_generator("x", 1),
        parent_metadata=parent,
        parent_type="face_crop",
        mask=_mask([0, 0, 1, 1]),
    )
    keys = list(meta.keys())
    assert keys[0] == "uuid"
    assert keys[1] == "schema_version"
    assert keys.index("provenance_chain") < keys.index("manipulation_type")


def test_extend_provenance_rejects_bad_parent_type() -> None:
    parent = _face_crop_parent()
    try:
        extend_provenance({}, parent, parent_type="not_a_type")
    except ValueError as e:
        assert "parent_type" in str(e)
    else:
        raise AssertionError("expected ValueError for invalid parent_type")
