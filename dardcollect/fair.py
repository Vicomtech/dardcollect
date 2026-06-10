"""FAIR (Findable, Accessible, Interoperable, Reusable) compliance utilities.

Ensures all pipeline outputs follow FAIR data principles by injecting:
- UUIDs for global uniqueness
- Schema versions for data structure validation
- Parent links for provenance tracking
- Source attribution (archive.org identifiers and URLs)
- License information

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
}


def generate_uuid() -> str:
    """Generate a new UUID version 4 string.

    Returns:
        str: Standard 36-character UUID v4 string (e.g., '12345678-1234-...').
    """
    return str(uuid.uuid4())


def add_fair_metadata(
    data: dict,
    schema_type: str,
    parent_uuid: str | None = None,
    parent_file: str | None = None,
    archive_org_id: str | None = None,
    archive_org_url: str | None = None,
) -> dict:
    """Inject FAIR-compliant fields into a data dictionary in-place.

    Adds UUID, schema version, parent provenance links, and source attribution.
    Mutates the input dict and returns it for convenience.

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

    Returns:
        dict: The same dictionary, mutated in-place (returned for convenience).
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
        elif schema_type == "transcription":
            # For video clip transcriptions
            data["parent_clip"] = {
                "uuid": parent_uuid,
                "file": parent_file,
            }
        elif schema_type in ("person_clip", "image_detection"):
            # For frames extracted from clips or image detection annotations
            data["parent_clip"] = {
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
    """Reorder dict keys so FAIR fields appear first.

    Creates a new dictionary with UUID, schema version, source, and parent
    links at the top. This makes sidecar JSON files human-readable without
    scrolling through large domain data to find identity fields.

    Call after `add_fair_metadata` so all FAIR fields are present.

    Args:
        data: Dictionary containing FAIR fields (will not be modified).
        schema_type: Unused — kept for API consistency with add_fair_metadata.

    Returns:
        dict: New dictionary with FAIR fields first, followed by all other keys.
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
