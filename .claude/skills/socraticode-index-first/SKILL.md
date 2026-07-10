---
name: socraticode-index-first
description: Index the repo with SocratiCode before navigating code, then use codebase_search/symbol/flow/impact/graph to understand it. Run at the start of any code work in this repo, before grep/glob fan-outs.
---

# socraticode-index-first

Before navigating the codebase in this repo, ensure it's indexed with SocratiCode, then prefer SocratiCode tools over ad-hoc `grep`/`glob` for structural questions.

## Steps
1. `codebase_status` — check index state. If not indexed or stale, `codebase_index` (runs in background) and poll `codebase_status` until 100%. Do NOT search until indexing completes.
2. Optionally `codebase_graph_build` for a dependency graph (poll `codebase_graph_status`); `codebase_graph_visualize` if a diagram helps.
3. Navigate with:
   - `codebase_search` — semantic search for "how does X work / where is X".
   - `codebase_symbol` / `codebase_symbols` — 360° view of a symbol (definition, callers, callees).
   - `codebase_flow` — trace execution flow from an entry point (auto-detected if no arg).
   - `codebase_impact` — blast radius before refactoring/renaming/deleting.
   - `codebase_graph_query` / `codebase_graph_circular` — imports, dependents, circular deps.
4. `codebase_context_search` for non-code artifacts (configs, schemas, specs) declared in `.socraticodecontextartifacts.json`. In this repo the `schemas/` JSON sidecar schemas and `config.yaml` are good candidates to register as context artifacts.

## When to prefer SocratiCode vs Grep/Glob
- **SocratiCode:** structural questions — "who calls X", "what breaks if I change Y", "where is the entry point", "how does this stage flow across files". Especially valuable here when splitting the god-files (`dardcollect/tracker.py`, `dardcollect/quality.py`, `dardcollect/pipeline_loggers.py`): `codebase_impact` gives the blast radius before you split or rename a function, and `codebase_symbol` shows callers/callees so you don't leave dangling references. `codebase_graph_circular` is the runnable layer-boundary gate (must stay at 0 circular deps).
- **Grep/Glob:** simple literal searches ("find every `TODO`", "list `*.yaml` files", locate a one-off script).

## Duplicate-config warning
If BOTH `mcp__plugin_socraticode_socraticode__*` and `mcp__socraticode__*` tools appear, the user has a duplicate MCP config — advise `claude mcp remove socraticode` (the plugin already provides the server).