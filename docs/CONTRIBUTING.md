# Contributing to DARDcollect

Thank you for your interest in contributing. This document covers code style rules and expectations for pull requests.

---

## Before You Start: Feature Requests & Planning

**If you are proposing a new feature (not a bug fix):**

1. **Open an issue** using the [Feature Request template](../.github/ISSUE_TEMPLATE/feature_request.md)
2. **Include a 1-page design doc** answering:
   - Which pipeline stage(s) will it affect?
   - Will it create new CSVs or extend existing sidecars?
   - Is it resumable? (critical for long-running jobs)
   - Does it align with the [project objective](../CLAUDE.md#objective)?

3. **Wait for review** before starting implementation to avoid wasted effort

**For developers using Claude/Copilot:** See [`.claude/FEATURE_WORKFLOW.md`](../.claude/FEATURE_WORKFLOW.md) for the complete protocol (gates, acceptance criteria, platform testing).

---

## Development Setup

**Installation & environment setup:** See [docs/4-DEVELOPMENT.md](4-DEVELOPMENT.md#development-workflow) for complete instructions.

**Pre-commit hooks** (installed automatically via `uv sync --extra dev`):
```bash
uv sync --extra dev   # installs ruff, ty, pytest, pre-commit, import-linter
pre-commit install    # registers the hooks in your local .git/
```
After `pre-commit install`, the following run automatically on every `git commit`:
- `pre-commit-hooks` hygiene: trailing whitespace, end-of-file fixer, `check-yaml`/`check-toml`, **`check-added-large-files` (10 MB — guards against committing fixture media / dataset blobs)**, merge-conflict, debug-statements.
- **Ruff** (check + format) and **ty** (type check).
- **import-linter** — hard-enforces the library/pipeline layer DAG (see "Library vs. Pipeline Scripts" below): the `dardcollect/` library must not import the `pipeline/` stage scripts.

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

All pull requests must pass `ruff check .` and `ty check .` with no errors. The rules are configured in `pyproject.toml` under `[tool.ruff]` and `[tool.ty]`.

---

## Docstring Style Guide

All functions, methods, and classes must include docstrings following **Google style** format. This ensures consistency across the codebase and improves IDE integration and automatic documentation generation.

### Module-Level Docstrings

Every Python module must start with a docstring describing its purpose:

```python
"""Brief module description.

Optional: Longer description of what this module provides, including
key classes or functions.

Provides:
    - `ClassName`: Brief description of the class.
    - `function_name()`: Brief description of the function.
"""
```

### Function/Method Docstrings

Use Google-style docstrings with the following sections:

```python
def calculate_face_crop_corners(
    keypoints: np.ndarray,
    kpt_scores: np.ndarray,
    mode: str = "ofiq",
) -> np.ndarray | None:
    """Brief one-line summary of what the function does.

    Optional: Longer description explaining the algorithm, edge cases,
    or important context. Can span multiple paragraphs.

    Args:
        keypoints: (K, 2) array of keypoint coordinates in image space.
        kpt_scores: (K,) array of keypoint confidence scores [0, 1].
        mode: "ofiq" (616×616), "arcface" (112×112), or "unaligned".

    Returns:
        (4, 2) float32 array of crop corners [TL, TR, BR, BL] in image space,
        or None if corners cannot be computed (e.g., insufficient keypoints).

    Raises:
        ValueError: If kpt_scores has invalid shape or all scores are below threshold.
    """
```

### Code References

Use **backticks** (not Sphinx-style `:func:` or `:class:`) for code references:

✅ **Correct:**
```python
"""Uses `PersonDetector` and calls `get_preferred_providers()` internally."""
```

❌ **Incorrect:**
```python
"""Uses :class:`PersonDetector` and calls :func:`get_preferred_providers()`."""
```

### Inline Comments

- Use single `#` for explanatory comments
- Explain **why**, not **what** — the code shows what it does
- Avoid over-commenting obvious code
- Section headers use decorative separators:
  ```python
  # ── Signal processing step ──────────────────────────────────────────
  ```

### Example: Complete Function

```python
def extract_ofiq_crop(
    image: np.ndarray,
    corners: np.ndarray,
    output_size: int = 616,
) -> np.ndarray:
    """Extract a normalized OFIQ face crop using affine transformation.

    Applies an affine warp to extract a square face region from the image.
    The output is always RGB, float32 [0, 1], and ready for quality scoring.

    Args:
        image: Input image (H, W, 3) in BGR uint8 format.
        corners: (4, 2) array of crop corners [TL, TR, BR, BL] in image space.
        output_size: Side length of the square output crop (default: 616 for OFIQ).

    Returns:
        (output_size, output_size, 3) float32 array in RGB format, normalized to [0, 1].

    Raises:
        ValueError: If corners array has invalid shape or is degenerate.
    """
    # ── Validate and prepare corners ─────────────────────────────────────
    if corners.shape != (4, 2):
        raise ValueError(f"Expected corners shape (4, 2), got {corners.shape}")

    src = corners[:3].astype(np.float32)  # Use first 3 corners for affine
    dst = np.array(
        [[0, 0], [output_size, 0], [output_size, output_size]],
        dtype=np.float32,
    )

    # Compute affine transformation matrix and warp image
    M = cv2.getAffineTransform(src, dst)
    warped = cv2.warpAffine(image, M, (output_size, output_size))

    # Convert BGR → RGB and normalize to [0, 1]
    rgb = cv2.cvtColor(warped, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return rgb
```

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

1. Create a system card or model card in `dardcollect/models/README_<component>.md`. Use the existing cards as a template.
2. Add a row to the **AI Systems** table in [../README.md](../README.md).
3. Cross-link the new card from the related existing cards where appropriate.

This applies to detectors, trackers, pose estimators, segmentation algorithms, and any other automated decision-making component.

---

## Library vs. Pipeline Scripts

DARDcollect has two distinct layers:

> **Layer rule (hard-enforced):** the library (`dardcollect/`) must NOT import the pipeline stage scripts (`pipeline/`). This keeps the library usable standalone as a modular library (per the README). The boundary is hard-enforced by `import-linter` (`[tool.importlinter]` in `pyproject.toml`; runs as a pre-commit hook and a CPU gate — `lint-imports --config pyproject.toml`). The pipeline may import the library; the library may never reach back into the stage scripts.

Reusable components exposed in `dardcollect/__init__.py`:
- Classes: `PersonDetector`, `PoseEstimator`, `AudioTranscriber`, `DocumentExtractor`, etc.
- Functions: `process_image()`, `process_video()`, `extract_frames()`, `download_item()`, etc.
- Utilities: `add_fair_metadata()`, `check_face_visibility()`, `check_frontal_face()`, etc.

**Guidelines:**
- Code should be **modular and stateless** where possible
- Accept explicit parameters (avoid relying on global config)
- Include comprehensive docstrings (Google style)
- Raise informative exceptions on errors
- Update `dardcollect/__init__.py` `__all__` when adding public APIs
- Write unit tests for new components

### **Pipeline Scripts (`pipeline/` folder)**

Entry points that orchestrate library components:
- `extract_person_clips_from_videos.py` — Uses `PersonDetector`, `PersonTracker`, `PoseEstimator`
- `annotate_face_quality.py` — Uses `load_models()`, `score_video()`
- `download_media_from_archive.py` — Uses `download_item()`
- etc.

**Guidelines:**
- Scripts handle **CLI orchestration, config loading, logging, and progress bars**
- Delegate core logic to library components
- Use `_TqdmHandler` for logging that doesn't break progress bars
- Handle resumability (skip already-processed files)
- Write to CSV logs for traceability

### **When adding new functionality**

1. **Is it reusable in other contexts?** → Add to `dardcollect/`, export in `__all__`
2. **Is it specific to a pipeline stage?** → Keep in `pipeline/`, import from library
3. **Is it a validation helper?** → Add to `dardcollect/pipeline_utils.py` and export

Example:
```python
# NEW: Reusable face detection + validation
# Location: dardcollect/face_validation.py (NEW)
def validate_face_crop(crop_path, quality_threshold=70.0):
    """Check if a face crop meets quality standards."""
    # ... logic here
    return passes, score

# THEN: Use in pipeline scripts
# Location: pipeline/annotate_face_quality.py
from dardcollect import validate_face_crop
for crop in crops:
    if validate_face_crop(crop):
        # Process further
```

---

## Bundled Model Weights

The ONNX and PyTorch model files in `dardcollect/models/` are **not covered by the Apache 2.0 license**. Do not add new model weight files without first verifying their license permits redistribution. See [../NOTICE](../NOTICE) for the existing weight licenses.

---

← [Back to README](../README.md)
