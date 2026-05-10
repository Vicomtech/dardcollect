"""FAIR (Findable, Accessible, Interoperable, Reusable) compliance utilities.

Provides schema versioning, UUID generation, and metadata standardization
for all pipeline outputs to ensure data findability and reproducibility.
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
}


def generate_uuid() -> str:
    """Return a new UUID v4 string."""
    return str(uuid.uuid4())


def add_fair_metadata(
    data: dict,
    schema_type: str,
    parent_uuid: str | None = None,
    parent_file: str | None = None,
    archive_org_id: str | None = None,
    archive_org_url: str | None = None,
) -> dict:
    """Inject FAIR fields (uuid, schema_version, parent link, source) into *data* in-place.

    :param schema_type: One of 'person_clip', 'face_crop', 'quality_annotation',
        'transcription', 'document'.
    :param parent_uuid: UUID of the upstream artifact (e.g. the person clip's UUID
        when schema_type is 'face_crop').
    :param parent_file: Filename of the upstream artifact.
    :param archive_org_id: Archive.org identifier for public-domain source tracking.
    :param archive_org_url: Archive.org item URL.
    :return: The same dict, mutated in place (returned for convenience).
    """
    if "uuid" not in data:
        data["uuid"] = generate_uuid()

    if "schema_version" not in data:
        data["schema_version"] = SCHEMA_VERSIONS.get(schema_type, "1.0")

    if parent_uuid or parent_file:
        if schema_type == "face_crop":
            data["parent_clip"] = {
                "uuid": parent_uuid,
                "file": parent_file,
            }
        elif schema_type == "quality_annotation":
            data["parent_crop"] = {
                "uuid": parent_uuid,
                "file": parent_file,
            }

    if archive_org_id or archive_org_url:
        if "source" not in data:
            data["source"] = {}
        if archive_org_id:
            data["source"]["archive_org_id"] = archive_org_id
        if archive_org_url:
            data["source"]["archive_org_url"] = archive_org_url

    if "source" in data and "license" not in data.get("source", {}):
        if archive_org_id or archive_org_url:
            data["source"]["license"] = "public-domain"

    return data


def reorganize_for_fair(data: dict, schema_type: str) -> dict:
    """Return a new dict with FAIR fields (uuid, schema_version, source, parent_*) first.

    Call after add_fair_metadata so sidecars are readable without scrolling
    past domain data to find the identity fields.
    """
    ordered = {}

    if "uuid" in data:
        ordered["uuid"] = data.pop("uuid")
    if "schema_version" in data:
        ordered["schema_version"] = data.pop("schema_version")
    if "source" in data:
        ordered["source"] = data.pop("source")
    if "parent_clip" in data:
        ordered["parent_clip"] = data.pop("parent_clip")
    if "parent_crop" in data:
        ordered["parent_crop"] = data.pop("parent_crop")

    ordered.update(data)

    return ordered


# ── archive.org helpers (consolidated from download_media_from_archive) ────────


def _get_metadata_value(item, key: str, default: str = "") -> str:
    """Safely extract metadata value from an archive.org item, handling lists and None."""
    val = item.metadata.get(key, default)
    if isinstance(val, list):
        return "; ".join(str(v) for v in val if v)
    return str(val) if val else default


def _build_fair_metadata(identifier: str, item, filename: str, media_type: str) -> dict:
    """Build metadata dict for a downloaded archive.org item.

    Pipeline fields come first; all Archive.org item.metadata fields follow.
    Archive.org's own 'identifier' field is skipped — captured as archive_org_identifier.
    """
    from dardcollect.provenance import now_iso

    metadata = {
        "uuid": generate_uuid(),
        "archive_org_identifier": identifier,
        "filename_downloaded": filename,
        "media_type": media_type,
        "downloaded_at": now_iso(),
    }
    for key, val in item.metadata.items():
        if key == "identifier":
            continue  # same value as archive_org_identifier
        if isinstance(val, list):
            metadata[key] = "; ".join(str(v) for v in val if v)
        else:
            metadata[key] = str(val) if val is not None else ""
    return metadata


def load_schema(schema_type: str) -> dict:
    """Load the JSON Schema for *schema_type* from schemas/{schema_type}_schema.json.

    :raises FileNotFoundError: If the schema file does not exist.
    """
    schema_file = Path(__file__).parent.parent / "schemas" / f"{schema_type}_schema.json"
    if not schema_file.exists():
        raise FileNotFoundError(f"Schema not found: {schema_file}")

    with open(schema_file, encoding="utf-8") as f:
        return json.load(f)


def validate_against_schema(data: dict, schema_type: str) -> bool:
    """Validate *data* against the JSON Schema for *schema_type*.

    :return: True if valid.
    :raises jsonschema.ValidationError: If validation fails.
    :raises FileNotFoundError: If the schema file does not exist.
    """
    schema = load_schema(schema_type)
    jsonschema.validate(data, schema)
    return True
