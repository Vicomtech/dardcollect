"""FAIR (Findable, Accessible, Interoperable, Reusable) compliance utilities.

Ensures all pipeline outputs follow FAIR data principles by injecting:
- UUIDs for global uniqueness
- Schema versions for data structure validation
- Parent links for provenance tracking
- Source attribution (archive.org identifiers and URLs)
- License information
- A Dublin Core JSON-LD ``@context`` so sidecars parse as linked data

Also provides JSON Schema loading and validation for all output types.
"""

import json
import uuid
from pathlib import Path

import jsonschema

# Schema versions for each data type
SCHEMA_VERSIONS = {
    "person_clip": "1.0",
    "face_crop": "1.0",
    "quality_annotation": "1.0",
    "transcription": "1.0",
    "document": "1.0",
    "image_detection": "1.0",
}

# Shared JSON-LD @context (Dublin Core Terms + schema.org). Injected into every
# sidecar by `add_fair_metadata` so the JSON files are also valid JSON-LD: a
# consumer can lift `source.title` → dct:title, `source.creator` →
# dct:creator, `source.license` → dct:license, `uuid` → dct:identifier and the
# `parent_*` links → prov:wasDerivedFrom without any transformation. Deliberately
# vocabulary-less (no `@vocab`): only real, resolvable namespaces (dct:, prov:)
# are mapped — keys not listed here stay plain JSON. Only pipeline-relevant
# terms are pinned.
JSONLD_CONTEXT = {
    "dct": "http://purl.org/dc/terms/",
    "prov": "http://www.w3.org/ns/prov#",
    "title": "dct:title",
    "creator": "dct:creator",
    "date": "dct:date",
    "license": "dct:license",
    "uuid": "dct:identifier",
    "schema_version": "dct:conformsTo",
    "source": "dct:source",
    "parent_clip": "prov:wasDerivedFrom",
    "parent_crop": "prov:wasDerivedFrom",
    "parent_audio": "prov:wasDerivedFrom",
}


def generate_uuid() -> str:
    """Generate a new UUID version 4 string.

    Returns:
        str: Standard 36-character UUID v4 string (e.g., '12345678-1234-...').
    """
    return str(uuid.uuid4())


# Which parent-link key each schema type uses in its sidecar.
_PARENT_KEY_BY_SCHEMA: dict[str, str] = {
    "face_crop": "parent_clip",
    "quality_annotation": "parent_crop",
    "transcription": "parent_clip",
    "person_clip": "parent_clip",
    "image_detection": "parent_clip",
}


def _add_parent_link(
    data: dict,
    schema_type: str,
    parent_uuid: str | None,
    parent_file: str | None,
) -> None:
    """Set the schema-appropriate parent link (``parent_clip``/``parent_crop``)."""
    if not (parent_uuid or parent_file):
        return
    key = _PARENT_KEY_BY_SCHEMA.get(schema_type)
    if key is not None:
        data[key] = {"uuid": parent_uuid, "file": parent_file}


def _add_source_attribution(
    data: dict,
    archive_org_id: str | None,
    archive_org_url: str | None,
) -> None:
    """Set the ``source`` block and its public-domain license when applicable."""
    if archive_org_id or archive_org_url:
        data.setdefault("source", {})
        if archive_org_id:
            data["source"]["archive_org_id"] = archive_org_id
        if archive_org_url:
            data["source"]["archive_org_url"] = archive_org_url
    if "source" in data and "license" not in data.get("source", {}):
        if archive_org_id or archive_org_url:
            data["source"]["license"] = "public-domain"


def add_fair_metadata(
    data: dict,
    schema_type: str,
    parent_uuid: str | None = None,
    parent_file: str | None = None,
    archive_org_id: str | None = None,
    archive_org_url: str | None = None,
    title: str | None = None,
    creator: str | None = None,
) -> dict:
    """Inject FAIR-compliant fields into a data dictionary in-place.

    Adds UUID, schema version, the shared JSON-LD ``@context``, parent
    provenance links, and source attribution. Mutates the input dict and
    returns it for convenience.

    Args:
        data: Dictionary to enrich with FAIR fields. Modified in-place.
        schema_type: Data type key for schema version lookup.
            One of: 'person_clip', 'face_crop', 'quality_annotation',
            'transcription', 'document'.
        parent_uuid: UUID of the upstream artifact (e.g., the person clip's UUID
            when schema_type is 'face_crop').
        parent_file: Filename of the upstream artifact.
        archive_org_id: archive.org identifier for public-domain source tracking.
        archive_org_url: archive.org item URL.
        title: Dublin Core title for the sidecar (dct:title via @context).
        creator: Dublin Core creator for the sidecar (dct:creator via @context).

    Returns:
        dict: The same dictionary, mutated in-place (returned for convenience).
    """
    if "uuid" not in data:
        data["uuid"] = generate_uuid()

    if "schema_version" not in data:
        data["schema_version"] = SCHEMA_VERSIONS.get(schema_type, "1.0")

    if "title" not in data and title:
        data["title"] = title
    if "creator" not in data and creator:
        data["creator"] = creator

    _add_parent_link(data, schema_type, parent_uuid, parent_file)

    # Shared JSON-LD context (Dublin Core Terms + PROV-O) — makes the sidecar
    # parse as linked data. Injected last among the FAIR identity fields so
    # reorganize_for_fair can place it right after schema_version.
    if "@context" not in data:
        data["@context"] = dict(JSONLD_CONTEXT)

    _add_source_attribution(data, archive_org_id, archive_org_url)

    return data


def reorganize_for_fair(data: dict) -> dict:
    """Reorder dict keys so FAIR fields appear first.

    Creates a new dictionary with the JSON-LD @context, UUID, schema version,
    Dublin Core terms, source, and parent links at the top. This makes sidecar
    JSON files human-readable without scrolling through large domain data to
    find identity fields.

    Call after `add_fair_metadata` so all FAIR fields are present.

    Args:
        data: Dictionary containing FAIR fields (will not be modified).

    Returns:
        dict: New dictionary with FAIR fields first, followed by all other keys.
    """
    ordered = {}

    if "@context" in data:
        ordered["@context"] = data.pop("@context")
    if "uuid" in data:
        ordered["uuid"] = data.pop("uuid")
    if "schema_version" in data:
        ordered["schema_version"] = data.pop("schema_version")
    if "title" in data:
        ordered["title"] = data.pop("title")
    if "creator" in data:
        ordered["creator"] = data.pop("creator")
    if "source" in data:
        ordered["source"] = data.pop("source")
    if "parent_clip" in data:
        ordered["parent_clip"] = data.pop("parent_clip")
    if "parent_audio" in data:
        ordered["parent_audio"] = data.pop("parent_audio")
    if "parent_crop" in data:
        ordered["parent_crop"] = data.pop("parent_crop")

    ordered.update(data)

    return ordered


# ── archive.org helpers (consolidated from download_media_from_archive) ────────


def _get_metadata_value(item, key: str, default: str = "") -> str:
    """Safely extract a metadata value from an archive.org item.

    Handles lists (joins with '; ') and None values.

    Args:
        item: archive.org item object with a .metadata dict.
        key: Metadata key to look up.
        default: Value to return if the key is missing or the value is empty.

    Returns:
        str: The metadata value as a string, or *default* if not found.
    """
    val = item.metadata.get(key, default)
    if isinstance(val, list):
        return "; ".join(str(v) for v in val if v)
    return str(val) if val else default


def _build_fair_metadata(identifier: str, item, filename: str, media_type: str) -> dict:
    """Build a FAIR metadata dict for a downloaded archive.org item.

    Pipeline-specific fields (UUID, identifier, filename, media_type, timestamp)
    come first. All remaining Archive.org metadata fields follow.
    The 'identifier' key from item.metadata is skipped — it is captured as
    archive_org_identifier instead.

    Args:
        identifier: archive.org item identifier.
        item: archive.org item object.
        filename: Name of the downloaded file.
        media_type: Type of media (video, audio, image, text).

    Returns:
        dict: Complete FAIR metadata dictionary with all archive.org fields.
    """
    from dardcollect.provenance import now_iso

    metadata = {
        "@context": dict(JSONLD_CONTEXT),
        "uuid": generate_uuid(),
        "title": _get_metadata_value(item, "title") or filename,
        "creator": _get_metadata_value(item, "creator"),
        "date": _get_metadata_value(item, "date"),
        "license": _get_metadata_value(item, "licenseurl"),
        "archive_org_identifier": identifier,
        "filename_downloaded": filename,
        "media_type": media_type,
        "downloaded_at": now_iso(),
    }
    for key, val in item.metadata.items():
        if key == "identifier":
            continue  # same value as archive_org_identifier
        if key in ("title", "creator", "date", "licenseurl"):
            continue  # captured above as Dublin Core terms
        if isinstance(val, list):
            metadata[key] = "; ".join(str(v) for v in val if v)
        else:
            metadata[key] = str(val) if val is not None else ""
    return metadata


def load_schema(schema_type: str) -> dict:
    """Load a JSON Schema from the schemas/ directory.

    Args:
        schema_type: Data type key (e.g., 'person_clip', 'face_crop').

    Returns:
        dict: Parsed JSON Schema as a Python dictionary.

    Raises:
        FileNotFoundError: If schemas/{schema_type}_schema.json does not exist.
    """
    schema_file = Path(__file__).parent.parent / "schemas" / f"{schema_type}_schema.json"
    if not schema_file.exists():
        raise FileNotFoundError(f"Schema not found: {schema_file}")

    with open(schema_file, encoding="utf-8") as f:
        return json.load(f)


def validate_against_schema(data: dict, schema_type: str) -> bool:
    """Validate a data dictionary against the JSON Schema for a given type.

    Args:
        data: Dictionary to validate.
        schema_type: Data type key for schema lookup.

    Returns:
        bool: True if validation passes.

    Raises:
        FileNotFoundError: If the schema file does not exist.
        jsonschema.ValidationError: If the data does not conform to the schema.
    """
    schema = load_schema(schema_type)
    jsonschema.validate(data, schema)
    return True
