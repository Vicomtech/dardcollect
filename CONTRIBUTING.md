# Contributing to DETECTOR Archive Data Collector

Thank you for your interest in contributing. This document covers how to set up a development environment, the code style rules, and what is expected in a pull request.

---

## Development Setup

```bash
# 1. Create and activate the environment
uv venv --python 3.12
source .venv/bin/activate        # Linux/Mac
.venv\Scripts\activate           # Windows

# 2. Install the package with dev dependencies
uv pip install -e ".[dev]"

# 3. Set up pre-commit hooks (optional but recommended)
pip install pre-commit
pre-commit install
```

---

## Code Style

This project uses [Ruff](https://docs.astral.sh/ruff/) for linting and formatting, plus [ty](https://docs.astral.sh/ty/) for type checking.

**Check for issues:**
```bash
ruff check .
ty check .
```

**Auto-fix and format:**
```bash
ruff check --fix .
ruff format .
ty check .  # ty does not auto-fix; review type issues manually
```

### Pre-commit Hooks

Once you've run `pre-commit install`, Ruff and `ty` will automatically run before each commit. This ensures all commits are properly formatted and type-safe—no manual intervention needed.

To run the checks manually without committing:
```bash
pre-commit run --all-files
```

All pull requests must pass `ruff check .` and `ty check .` with no errors. The rules are configured in `pyproject.toml` under `[tool.ruff]` and `[tool.ty]`.

---

## Pull Request Guidelines

- **Branch naming**: `feature/short-description`, `fix/short-description`
- **PR title**: short, imperative ("Add X", "Fix Y", "Remove Z") — not "I added X" or "WIP"
- **Scope**: one concern per PR; avoid mixing unrelated changes
- **Tests**: if you add or change pipeline logic, verify on a real video with `dry_run: true` before submitting
- **Documentation**: update `config.yaml` comments and the relevant README sections if you change behaviour or add parameters

---

## Adding a New Pipeline Component

Any new automated component — whether it uses a learned model or a rule-based algorithm — must be documented as an AI system in accordance with EU AI Act Annex IV:

1. Create a system card or model card in `persondet/models/README_<component>.md`. Use the existing cards as a template.
2. Add a row to the **AI Systems** table in `README.md`.
3. Cross-link the new card from the related existing cards where appropriate.

This applies to detectors, trackers, pose estimators, segmentation algorithms, and any other automated decision-making component.

---

## Bundled Model Weights

The ONNX and PyTorch model files in `persondet/models/` are **not covered by the Apache 2.0 license**. Do not add new model weight files without first verifying their license permits redistribution. See [NOTICE](NOTICE) for the existing weight licenses.
