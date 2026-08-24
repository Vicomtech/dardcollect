# Design: Manipulation provenance (edited / generatively-inpainted artifacts)

Status: implemented (schema + library helpers + tests). No producer pipeline
stage yet — this lands the FAIR contract so a future manipulation stage (or an
external ingestion tool) writes traceable sidecars from day one.

## Problem statement (what + why)

DARDcollect must build a *labelled* dataset for a manipulation detector. A
manipulated image (generative inpainting / editing) needs a sidecar that records
**what was done, where, and from what** — and that traceability must survive
**manipulation-over-manipulation** and the **loss of intermediate artifacts**.

The existing FAIR model links each artifact to its parent by `uuid`
(`add_fair_metadata`). Pure link-chasing breaks the moment an intermediate
sidecar/file is deleted (dangling UUID), exactly like deleting a source video
would — except there is no `downloads.csv` backstop for edits. We need the
provenance to be **self-contained**.

## Architecture (which stages, new vs existing)

New, all in the **library layer** (no `pipeline/` dependency, import-linter
clean):

- `schemas/manipulation_schema.json` — new artifact type, same house style as
  the other six schemas; validated at write via `fair.validate_against_schema`.
- `dardcollect/manipulation.py` — helpers that assemble a schema-ready sidecar:
  `build_manipulation_metadata` (public entry point), `extend_provenance`
  (parent link + root/depth + inherited chain), `build_provenance_entry`,
  `walk_provenance`, `cumulative_fake_regions`.
- `dardcollect/fair.py` — `SCHEMA_VERSIONS["manipulation"]="1.0"` and FAIR-field
  ordering for `parent` / `root_uuid` / `manipulation_depth` / `provenance_chain`.

**Hybrid traceability** (the core decision):

| Mechanism | Purpose | Survives ancestor deletion? |
| :-- | :-- | :-- |
| `parent {uuid,file,type}` | fast live traversal | no (link only) |
| `provenance_chain[]` | self-contained lineage copy | **yes** |
| `root_uuid`, `manipulation_depth` | O(1) shortcuts to root / edit depth | yes |

Each manipulation **inherits its parent's `provenance_chain`** and appends the
parent's own compact record. Each record embeds the mask **geometry**
(`bbox`/`quad`, not just the PNG path), the generator params (prompt/seed/model)
and a `sha256`, so "which region was faked, how" is recoverable from the final
sidecar alone.

### Manipulation-over-manipulation

`parent.type == "manipulation"` ⇒ `root_uuid = parent.root_uuid`,
`manipulation_depth = parent.depth + 1`, `chain = parent.chain + [parent]`.
Otherwise (parent is a real `face_crop`/`image_detection`) ⇒ `root_uuid =
parent.uuid`, `depth = 0`, `chain = [parent]`. Pristine `source` attribution is
inherited down the chain (keeps public-domain provenance intact at any depth).

### External / unknown-history artifacts

`parent_metadata=None` ⇒ `parent=null`, `root_uuid=null`, `depth=0`,
`provenance="external"`, empty chain. We record only what we know; we do not
invent ancestors.

## FAIR impact

- New sidecar type (JSON), validated at write. No new CSV.
- Provenance chain **Archive.org → … → face_crop → manipulation → manipulation
  …** stays resolvable both live (links) and offline (embedded chain).
- New AI system row in the README AI Systems table (EU AI Act Annex IV): the
  generative model is documented per artifact via the `generator` block
  (name/version/provider + prompt/seed/steps/guidance/scheduler).

## Resumability strategy

Producer-side (future stage): a manipulated output + its `.json` sidecar are the
unit; a `.done`/existing-sidecar check makes re-runs skip completed items, same
as other stages. The library helpers here are pure and deterministic (only
`uuid` and `manipulated_at` vary), so re-generating metadata is idempotent apart
from those identity/time fields.

## Test plan

`tests/test_manipulation_provenance.py` (CPU, no GPU):
- first edit on a real image → depth 0, root = parent, chain len 1, source inherited;
- edit-of-edit → depth 1, root preserved, chain len 2;
- 3-deep chain → self-contained (ancestor mask geometry + generator embedded);
- `cumulative_fake_regions` unions all edit masks;
- external → fresh chain, `provenance="external"`;
- FAIR fields ordered first; invalid `parent_type` rejected.
Every generated sidecar is validated against the schema.

Fixture/objective gate: unaffected (no producer stage yet); the golden harness
picks up manipulation sidecars automatically once a stage writes them.
