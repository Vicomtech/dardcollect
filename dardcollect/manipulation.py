"""Provenance for manipulated (edited / generatively-inpainted) media artifacts.

Traceability here is HYBRID, so the lineage of an edited image survives even
when the intermediate artifacts (or their sidecars) are no longer available:

* ``parent`` — a live link to the IMMEDIATE input of this operation, by UUID.
  Fast to traverse while everything is present in the repo. May itself point at
  another ``manipulation`` (manipulation-over-manipulation).
* ``provenance_chain`` — a self-contained, append-only copy of the whole
  lineage. Each manipulation INHERITS its parent's chain and appends the
  parent's own compact record, so this one sidecar carries the full history:
  every ancestor's uuid, operation, generator (prompt / seed / model), mask
  GEOMETRY (bbox / quad — recoverable without the mask PNG) and sha256.
* ``root_uuid`` / ``manipulation_depth`` — denormalized shortcuts to the
  pristine root and the number of edit hops (0 = first edit on a real image).

If the immediate input has no known history (an externally-sourced fake), pass
``parent_metadata=None``: the artifact is marked ``provenance="external"`` and
starts a fresh chain rather than inventing ancestors.

See ``docs/DESIGN_manipulation_provenance.md`` and
``docs/3-ANNOTATIONS.md`` §11 for the full contract.
"""

from __future__ import annotations

from dardcollect.fair import add_fair_metadata, reorganize_for_fair
from dardcollect.provenance import now_iso

# Artifact types that can appear as a manipulation's parent / chain entry.
PARENT_TYPES = ("image_detection", "face_crop", "manipulation")


def build_provenance_entry(metadata: dict, artifact_type: str, file: str | None = None) -> dict:
    """Build one compact, self-contained lineage record for an artifact.

    The record is deliberately independent of the artifact's file: it embeds the
    mask GEOMETRY and the generator parameters, so "what was done, where" is
    recoverable even if the artifact and its mask PNG are gone.

    Args:
        metadata: The artifact's full sidecar dict.
        artifact_type: One of PARENT_TYPES.
        file: Optional filename of the artifact's sidecar / media.

    Returns:
        dict: Compact record for inclusion in a ``provenance_chain``.
    """
    entry: dict = {"uuid": metadata.get("uuid"), "type": artifact_type}
    if file:
        entry["file"] = file

    size = (
        metadata.get("output_size")
        or metadata.get("source_image_size")
        or metadata.get("image_size")
    )
    if size:
        entry["size"] = size
    if metadata.get("sha256"):
        entry["sha256"] = metadata["sha256"]

    if artifact_type == "manipulation":
        if metadata.get("manipulation_type"):
            entry["op"] = metadata["manipulation_type"]
        if metadata.get("generator"):
            entry["generator"] = dict(metadata["generator"])
        if metadata.get("mask"):
            entry["mask"] = dict(metadata["mask"])
        if metadata.get("manipulated_at"):
            entry["at"] = metadata["manipulated_at"]

    return entry


def extend_provenance(
    data: dict,
    parent_metadata: dict | None,
    parent_type: str | None = None,
    parent_file: str | None = None,
) -> dict:
    """Populate parent link, root/depth shortcuts and the inherited chain in-place.

    Args:
        data: The manipulation dict being built. Modified in-place.
        parent_metadata: The immediate input artifact's full sidecar dict, or
            None when the input's history is unknown (external).
        parent_type: Artifact type of the parent (one of PARENT_TYPES). Required
            when ``parent_metadata`` is given.
        parent_file: Optional filename of the parent artifact.

    Returns:
        dict: The same dict, mutated in-place.

    Raises:
        ValueError: If ``parent_metadata`` is given without a valid ``parent_type``.
    """
    if parent_metadata is None:
        data["parent"] = None
        data["root_uuid"] = None
        data["manipulation_depth"] = 0
        data["provenance"] = "external"
        data.setdefault("provenance_chain", [])
        return data

    if parent_type not in PARENT_TYPES:
        raise ValueError(f"parent_type must be one of {PARENT_TYPES!r}, got {parent_type!r}")

    parent_uuid = parent_metadata.get("uuid")
    parent_link: dict = {"uuid": parent_uuid, "type": parent_type}
    if parent_file is not None:
        parent_link["file"] = parent_file
    data["parent"] = parent_link

    if parent_type == "manipulation":
        data["root_uuid"] = parent_metadata.get("root_uuid") or parent_uuid
        data["manipulation_depth"] = int(parent_metadata.get("manipulation_depth", 0)) + 1
        inherited = list(parent_metadata.get("provenance_chain", []))
    else:
        data["root_uuid"] = parent_uuid
        data["manipulation_depth"] = 0
        inherited = []

    inherited.append(build_provenance_entry(parent_metadata, parent_type, parent_file))
    data["provenance_chain"] = inherited

    # Pristine root attribution is inherited down the whole chain.
    if "source" not in data and parent_metadata.get("source"):
        data["source"] = dict(parent_metadata["source"])

    return data


def build_manipulation_metadata(
    *,
    manipulation_type: str,
    generator: dict | None = None,
    parent_metadata: dict | None = None,
    parent_type: str | None = None,
    parent_file: str | None = None,
    original_path: str | None = None,
    mask: dict | None = None,
    source_image_size: dict | None = None,
    output_size: dict | None = None,
    label: dict | None = None,
    output_sha256: str | None = None,
    source: dict | None = None,
) -> dict:
    """Assemble a schema-ready ``manipulation`` sidecar dict.

    Injects the FAIR identity (uuid, schema_version), the hybrid provenance
    (parent link + inherited self-contained chain + root/depth shortcuts) and
    the manipulation payload, then orders FAIR fields first. The caller is
    responsible for ``validate_against_schema(result, "manipulation")`` and
    writing the JSON, per the project's validate-at-write contract.

    Args:
        manipulation_type: One of the schema's manipulation_type enum values.
        generator: The generative AI system + reproducibility params
            (name/version/provider/prompt/seed/...). Omit for external artifacts.
        parent_metadata: Immediate input's full sidecar dict, or None (external).
        parent_type: Parent artifact type (required when parent_metadata given).
        parent_file: Filename of the parent artifact.
        original_path: Convenience path to the immediate input image.
        mask: Mask object carrying reference (file/uuid) AND geometry (bbox/quad).
        source_image_size / output_size: {"width", "height"} dicts.
        label: Ground-truth object, e.g. {"class": "manipulated", "fake_region": "mask"}.
        output_sha256: SHA-256 of the manipulated output file (fixity).
        source: Explicit root attribution; if omitted it is inherited from parent.

    Returns:
        dict: FAIR-ordered manipulation sidecar, ready to validate and write.
    """
    data: dict = {
        "manipulation_type": manipulation_type,
        "manipulated_at": now_iso(),
    }
    if generator is not None:
        data["generator"] = generator
    if original_path is not None:
        data["original_path"] = original_path
    if mask is not None:
        data["mask"] = mask
    if source_image_size is not None:
        data["source_image_size"] = source_image_size
    if output_size is not None:
        data["output_size"] = output_size
    if label is not None:
        data["label"] = label
    if output_sha256 is not None:
        data["sha256"] = output_sha256
    if source is not None:
        data["source"] = source

    add_fair_metadata(data, schema_type="manipulation")
    extend_provenance(data, parent_metadata, parent_type, parent_file)
    return reorganize_for_fair(data, "manipulation")


def walk_provenance(metadata: dict) -> list[dict]:
    """Return the full lineage of a manipulation, oldest first.

    This is the embedded ``provenance_chain`` (all ancestors) followed by this
    artifact's own compact record — the complete history, reconstructable from
    this sidecar alone.

    Args:
        metadata: A manipulation sidecar dict.

    Returns:
        list[dict]: Lineage records from pristine root to this artifact.
    """
    chain = list(metadata.get("provenance_chain", []))
    chain.append(build_provenance_entry(metadata, "manipulation", None))
    return chain


def cumulative_fake_regions(metadata: dict) -> list[dict]:
    """Return every edited region's mask up the chain, including this operation.

    The union of these regions is the accumulated fake area — the spatial
    ground-truth for a manipulation detector across a chain of edits.

    Args:
        metadata: A manipulation sidecar dict.

    Returns:
        list[dict]: Mask objects (with geometry) for each manipulation hop.
    """
    regions: list[dict] = []
    for entry in metadata.get("provenance_chain", []):
        if entry.get("type") == "manipulation" and entry.get("mask"):
            regions.append(entry["mask"])
    if metadata.get("mask"):
        regions.append(metadata["mask"])
    return regions
