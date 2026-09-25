---
name: originality-guard
description: Local-only license and IPR check for code and dependencies (EEE, no-reuse). Remote SaaS scanners blocked by default.
---

# originality-guard (local-only, code path)

Policy gate for license/IPR risk in code. Posture: **local-only, zero
egress**. Remote scanners (SCANOSS osskb, FOSSA/Snyk/Sonar cloud) are blocked
by default — an API key is an egress path. EU project: processing stays
inside the EEE; the checking tool must not reuse or license introduced
content. Model-training provenance cannot be certified here; it is mitigated
and logged, never assumed clean.

## Code path (local)

1. Run `uv run python scripts/license_scan.py` (advisory): unpinned deps,
   copyleft watch, duplicate-block watch. Stdlib only, no egress.
2. Release audit (manual, before publishing): ScanCode Toolkit, FOSSology,
   or ORT run locally. Snippet origin is NOT covered by license detection —
   accept the gap as documented or provision an on-prem KB.
3. Clone detection: the scan's 40-line watch is the cheap gate; a full
   `jscpd` run is the release tool.
4. Snippet > 15 lines or a full function requires reimplementation or a
   license-compatible source; generated blocks carry a provenance header
   (prompt, date, destination license).
5. Allowlist: MIT, Apache-2.0, BSD, PSF, ISC by default. Copyleft only with
   explicit approval recorded in the session log.

## Remote exception (gated)

Allowed only with all of: EU region, signed DPA, written no-reuse and
no-training, and a log entry stating what was sent, where it was processed,
and under which agreement. Free tiers without these papers never qualify.

## Done criteria

Evidence in the session close: tool run, warnings triaged, approvals
recorded, or explicit `none`. Findings state what changed, what was
verified, what failed. The dataset side (public-domain inputs) is covered
separately: `licenseurl:*publicdomain*` filter in `configs/` + the
`source.license` provenance block (`dardcollect/fair.py`).
