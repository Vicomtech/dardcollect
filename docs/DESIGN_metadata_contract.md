# DESIGN — Metadata contract (CSV ↔ JSON sidecar)

**Status:** Phases 1-5 implemented (2026-10-11/13) and the three known gaps closed;
the input-hash part of Phase 4 is deferred pending a cost decision. No artifact or
inference change, so no GPU reprocessing.

## Gap closure (2026-10-13)

The three remaining format gaps are closed:
- **Frame sidecars** now have their own `frame_schema.json` (they never fitted
  `person_clip`/`face_crop`, despite declaring them) and are validated at write
  (`frames.py`, `source_frames.py`); the golden `--validate` maps
  `extracted_frames/frame_*.json` to `frame` (the `frames_manifest.json` index
  stays schema-less).
- **`.magface.json`** now has `magface_schema.json`, validated at write by
  `filter_face_crops_by_quality.py` and mapped in the golden check. It is a derived
  working cache (no FAIR fields), not a FAIR-linked sidecar.
- **Model hashes**: `producer.models[]` records `{file, sha256}` for the ONNX
  weights a stage actually loaded (registered in `onnx_utils.create_ort_session`),
  hashed once per process.
- **Doc-drift gate**: `scripts/metadata_contract.json` (the Phase 0
  machine-readable contract) now exists as the single source for the CSV columns +
  schema inventory. `tests/test_metadata_contract.py` pins the code (each logger's
  `fieldnames`, the `schemas/` inventory); the `metadata contract` check in
  `scripts/validate_harness.py` pins the docs (columns verbatim in
  `docs/2-LINEAGE.md`, every schema named in `docs/3-ANNOTATIONS.md`).
  `docs/1-ARCHITECTURE.md` no longer enumerates columns — it points to
  `2-LINEAGE.md` — so it cannot drift.

## Implementation status (2026-10-11/13)

- **Phase 1 — done.** Every logger's row `uuid` is now the produced artifact's
  sidecar uuid (threaded at each write site); CSV link columns therefore carry the
  artifact uuid, the same identifier a sidecar's `parent_*.uuid` uses. The clips CSV
  links upstream by `download_uuid` (the manifest row's uuid) instead of
  `archive_org_identifier`. CSV timestamps are unified (`now_iso()`, `+00:00`).
- **Phase 2 — done.** The CSVs are the event log: identity + links + input/output
  paths + run-level values (detector/model, language, MagFace score/threshold).
  Measures the sidecar already owns were dropped: clips `fps/start_frame/end_frame/
  start_seconds/duration_seconds/max_persons_per_frame`; video crops `crop_id/face_bbox/
  confidence`; image crops `bbox_in_source/bbox_confidence`; image detections
  `source_image/num_persons`; frames `timestamp_seconds`; documents `text_length/word_count`.
  `print_summary` methods rewritten to the remaining columns.
- **Phase 3 — done.** `golden_snapshot._provenance_check` verifies each identity
  space against itself (defect-planting tests in `tests/test_provenance_spaces.py`).
  All six schemas are now closed (`additionalProperties: false`) and whitelist the
  FAIR/JSON-LD surface (`@context`, optional `title`/`creator`/`source`); the
  previously undeclared `source` (transcription) and `frames_scored`
  (quality annotation) are declared; a transcription now requires `parent_clip` or
  `parent_audio` (`anyOf`). `person_clip`/`image_detection` take no sidecar parent —
  their upstream is the download manifest, which has no sidecar (the CSV carries
  `download_uuid`).
- **Phase 4 — done (producer); input hash deferred.** `add_fair_metadata` injects a
  `producer` block into every sidecar: `pipeline_commit` (`DARD_PIPELINE_COMMIT`
  override, else the checkout's git HEAD) and `config_sha256` (the file named by
  `DARDCOLLECT_CONFIG`, which the orchestrator now exports to every stage
  subprocess). New `dardcollect/reproducibility.py`; declared in all schemas and
  mapped to `prov:wasGeneratedBy` in the JSON-LD context. The **input `sha256`** is
  deferred: hashing the 228 GB RAVDESSfake source is expensive and its scope (source
  vs produced artifact) is a user decision.
- **Phase 5 — done.** The versioning policy (semver on `schema_version`; additive ->
  minor, required/closed/semantics -> major) is recorded here and in
  `docs/3-ANNOTATIONS.md`. `schema_version` is enforced by each schema's `enum`; a
  test now pins `dardcollect.fair.SCHEMA_VERSIONS` to those enums so the code
  constant and the schemas cannot drift.
- **Not done:** Phase 4's input `sha256` (cost decision pending).


## Problem

The same artifact is described twice, by two independent code paths, and only one
half is checked. Evidence:

1. **Duplicated facts.** `clips_extraction.csv` (written by the logger,
   `dardcollect/extraction_logger.py:147`) and `person_clip_schema.json` (written by
   the clip writer) both record `source_video`, fps (`fps` vs `video_info.fps`),
   `start_frame`, `end_frame`, `start_seconds`, `duration_seconds`,
   persons-per-frame (`max_persons_per_frame` vs `max_persons`), the Archive.org id
   (`archive_org_identifier` vs `source.archive_org_id`) and the clip path
   (`output_path` vs `clip_path`) — eight facts, under different names. Face crops
   repeat the pattern (bbox, confidence, source path, timestamp, parent link).
   Two code paths, one validated (JSON) and one not (CSV) → they can diverge silently.
2. **Two UUID spaces, mixed.** A CSV `uuid` identifies an *event*; a sidecar `uuid`
   identifies an *artifact* (`docs/2-LINEAGE.md`, "How to read the records"). The only
   mechanical provenance check pools both spaces — plus CSV `parent_uuid`/`download_uuid`
   columns and sidecar `uuid`s — into a single set (`scripts/golden_snapshot.py:244-267`),
   so a parent link resolved against the *wrong* space passes.
3. **The join is the filename, not the UUID.** `_build_lookup`
   (`dardcollect/modality_loggers.py:25`) indexes by `Path(...).stem`/`.name`. The UUID is
   a record of the join, not the means of it; a rename breaks the chain while the UUIDs
   still "resolve".
4. **Inconsistent links.** `clips_extraction.csv` links upstream by
   `archive_org_identifier`; `image_person_detection.csv`, the audio and the video
   transcription CSVs link by `download_uuid`.
5. **Inconsistent timestamps.** Some loggers write `.isoformat()` (`+00:00`), others
   `.replace("+00:00", "Z")` — same field, two formats across stages.
6. **Open schemas, unenforced parent.** `add_fair_metadata` injects `@context` into every
   sidecar (`dardcollect/fair.py:155`), but only `document_schema.json` declares it (5/6
   schemas are open). `person_clip` and `image_detection` do not require a parent link,
   and `_build_lookup` returns `{}` on a missing CSV, so an unresolved link is dropped
   by `_add_parent_link` (`dardcollect/fair.py:74`) instead of failing.
7. **Unused version field.** `schema_version` is written into every sidecar
   (`dardcollect/fair.py:143`) and read by nothing but tests.

## Non-goals

- Not changing artifacts or inference outputs.
- Not touching `downloads.csv` as a **source manifest**: its Archive.org metadata exists
  in no sidecar, and it is the download-dedup ledger
  (`pipeline/download_media_from_archive.py:130`, `dardcollect/ingest.py:98`).
- Not claiming "perfect metadata". The achievable goal is a **frozen, mechanically
  enforced contract** — stable because it is closed and watched, not because fields
  were added.

## Ownership rule (contract)

- **Sidecar JSON** — owner of every *artifact* fact (what it is, how it looks, how it was
  measured, where it came from). Schema-validated.
- **CSV** — the *event log*: event identity, upstream link, `timestamp`, `output_path`,
  and the run's decision/threshold values (`magface_score`, `filter_threshold`,
  `detector_model`, `detector_confidence`). No artifact-descriptive fields.
- **Exception, documented** — `downloads.csv` / the custom source manifest, as in § Non-goals.

## Identity model (decision)

Recommended: every `*_uuid` **link** column holds the parent's **artifact (sidecar) UUID**.
The CSV row's own `uuid` identifies the event and is never a link target. `_build_lookup`
then reads the sidecar UUID (today it reads the CSV row UUID). Result: one join rule for
the whole provenance graph — source manifest → download → clip/crop → quality — in CSV and
JSON alike.

## Versioning policy (decision)

Semantic on `schema_version`: additive/optional → minor; new `required`, closed
`additionalProperties`, or a changed link semantics → major. **Do not bump until the policy
exists and at least one consumer branches on it** (none does today).

## Phased plan (dependency order)

| Phase | Work | Reprocess? |
|---|---|---|
| 0 | This note + a machine-readable ownership contract (`metadata_contract.json`). | none |
| 1 | Identity & links: unify links on the artifact UUID space; standardize `download_uuid`; `_build_lookup` reads the sidecar UUID; backfill existing CSVs. | metadata backfill only |
| 2 | Single ownership: drop artifact-descriptive columns from the CSVs; keep event id + parent link + `timestamp` + `output_path` + decision value; regenerate existing CSVs. | metadata backfill only |
| 3 | Enforcement: declare `@context` (and `source`/`title`/`creator`) in **all** schemas before closing them; require the parent link in `person_clip`/`image_detection`; split the provenance check per UUID space; validate CSV structure against the contract; defect-planting tests (crossed link, duplicated column). | none |
| 4 | Reproducibility (additive): `producer` (commit, config hash, model hash/version), input `sha256`, uniform `generated_at`; unify CSV timestamps. | sidecar rewrite only |
| 5 | Versioning + migration route. | none |

## Cost over an existing dataset (e.g. RAVDESSfake)

No phase changes artifacts or inference data. Phases 1, 2 and 4 are **metadata backfill /
regeneration** (I/O over the output tree); 3 and 5 are checks/docs. **Zero GPU reprocessing.**
CSV regeneration is required because loggers only write a header when the file is empty, so
an old header persists otherwise.

## Gates and evidence

Design note + `metadata_contract.json` (Phase 0) before any code. Each phase closes with
`scripts/objective_gate.py` **fresh** (never `--no-wipe`), `golden_snapshot.py --validate`,
defect-planting tests, `scripts/validate_harness.py`, docs synced, and Windows + (WSL|Linux).
Evidence = gate outputs and golden diff, never "it works".

## Open decisions (recommendation = default)

1. **Identity** — recommended: links in the artifact-UUID space (single join rule).
   Alternative: keep two spaces and verify each separately.
2. **CSV** — recommended: hand-written event log + a *checked* contract (not a generated
   projection, which would change runtime).
3. **Versioning** — recommended: semver on the schema.
4. **Scope** — recommended: approve Phase 0 on paper first.

← [Back to README](../README.md)
