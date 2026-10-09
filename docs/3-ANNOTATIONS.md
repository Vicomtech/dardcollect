# Sidecar JSON annotations

This guide explains the JSON sidecars written beside pipeline artifacts. The
JSON Schemas in [`schemas/`](../schemas/) are the source of truth for required
fields and value constraints. CSV logs are covered in
[2-LINEAGE.md](2-LINEAGE.md).

## Sidecar map

| Artifact | Sidecar | Schema |
|---|---|---|
| Person clip (`extracted_person_clips/<clip>.mp4`) | `<clip>.json` | `person_clip_schema.json` |
| Image detection (`extracted_image_detections/<image>.json`) | `<image>.json` | `image_detection_schema.json` |
| Video or image face crop | `<crop>.json` | `face_crop_schema.json` |
| Face quality | `<crop>.ofiq_attr.json`, `<crop>.magface.json` | `quality_annotation_schema.json` (OFIQ; MagFace has no separate schema) |
| Video transcription | `<clip>.transcription.json` | `transcription_schema.json` |
| Audio transcription | `<audio>.transcription.json` | `transcription_schema.json` |
| Document text | `<document>.annotation.json` | `document_schema.json` |

Every sidecar has its own `uuid` and `schema_version` (`"1.0"`). Parent links
point to the parent sidecar's UUID. For example, a face crop's `parent_clip.uuid`
is the parent person clip's sidecar UUID.

## Video and image variants

Video face crops and image face crops share one schema with two variants:

- **Video crop** — `source_video`, `track_id`, `duration_seconds` and a
  `frame_data` object.
- **Image crop** — `image_path`, `person_idx`, the source bounding box and
  keypoints.

Crop geometry is shared by pixels and annotations. Video stabilization uses
confidence-weighted eye-contour landmarks to estimate eye centers, line angle,
and interocular scale. It filters the midpoint once and crop pose twice; it
never freezes translation. The final filtered eye anchor maps to the canonical
OFIQ midpoint. Each output frame stores:

- `source_frame_index`: the source frame whose pixels are used;
- `render_quad_source`: the exact source-space OFIQ quad `[TL, TR, BR, BL]`
  used to render that frame.

Keypoints and bounding boxes in `frame_data` are already expressed in
output-crop coordinates through that same quad. Do not estimate a second
alignment from the sidecar. Gap frames repeat the previous source index, quad
and annotations.

```json
{
  "uuid": "…",
  "schema_version": "1.0",
  "parent_clip": { "uuid": "…", "file": "VideoTitle.mp4" },
  "source_video": "…/VideoTitle.mp4",
  "track_id": 0,
  "crop_format": "ofiq",
  "output_size": 616,
  "stabilized": true,
  "stabilization_window_seconds": 0.8,
  "stabilization_anchor_tolerance_px": 2.5,
  "source_frame_overshoot_px": 64.5,
  "frame_data": {
    "0": [
      {
        "track_id": 0,
        "source_frame_index": 1200,
        "render_quad_source": [[160, 180], [340, 180], [340, 560], [160, 560]],
        "bbox": [50, 60, 560, 570],
        "keypoints": [[60, 75]],
        "keypoint_scores": [0.98]
      }
    ]
  }
}
```

The example is abbreviated; the schema defines the complete structure.

`source_frame_overshoot_px` is the largest distance by which the rendered quad
extends beyond the source. Pixels outside the source are black, not replicated.
A value of `0` means the crop lies fully inside the source.

## Transcription and document sidecars

Transcriptions and documents use the same pattern: a UUID, a parent link and
the extracted content. Their schemas are `transcription_schema.json` and
`document_schema.json`.

```json
{
  "uuid": "…",
  "schema_version": "1.0",
  "parent_clip": { "uuid": "…", "file": "VideoTitle_00m12s-00m15s.mp4" },
  "transcriber": { "method": "openai_whisper", "model_size": "small" },
  "transcribed_at": "2026-05-06T10:28:54+00:00",
  "language": "en",
  "duration_seconds": 3.0,
  "transcription": "Well, hello there! How are you today?",
  "segments": [
    { "start": 0.0, "end": 2.5, "text": "Well, hello there!" },
    { "start": 2.5, "end": 3.0, "text": "How are you today?" }
  ]
}
```

A transcription of an audio file has `parent_audio.filename` instead of
`parent_clip`. A document annotation (`<document>.annotation.json`) records how
the text was obtained:

```json
{
  "uuid": "…",
  "schema_version": "1.0",
  "source_file": "1955_10_28_Green_Mountain_Rifleman.pdf",
  "extraction_method": "text_layer",
  "page_count": 12,
  "word_count": 2140,
  "char_count": 11873,
  "text_file": "1955_10_28_Green_Mountain_Rifleman.text.txt",
  "processed_at": "2026-05-07T09:12:00+00:00"
}
```

`extraction_method` is `text_layer` (embedded text), `ocr_paddleocr` (OCR fallback)
or `native` (plain `.txt`).

## Quality sidecars

Quality data sits beside each face crop, not in the person-clip sidecar.

- `.magface.json` is written by the quality filter. It holds the MagFace
  `unified_score` aggregate that the filter thresholds.
- `.ofiq_attr.json` is written by the quality annotation, from the pipeline stage
  or from `score_video`; both produce the same file. It holds, per measure, the
  aggregate statistics (`max`, `mean`, `p10`, `p50`, `p90`) over the sampled
  frames, `frames_scored`, and `frame_data` with the single-frame values the viewer
  reads. For crops with `crop_format: "ofiq"` it also holds `unified_score`: the
  MagFace score computed per frame during annotation.

Both files link to their parent sidecar. `annotator` names the writer:
`pipeline/annotate_face_quality.py` or `dardcollect/quality.py`.

| Measure | Model or method | Range |
|---|---|---|
| `unified_score` | MagFace IResNet50 | 0–100, higher is better |
| `sharpness` | Random forest | 0–100, higher is better |
| `compression_artifacts` | SSIM CNN | 0–100, higher is better |
| `expression_neutrality` | HSEmotion + AdaBoost | 0–100, higher is better |
| `no_head_coverings` | BiSeNet | 0–100, higher is better |
| `face_occlusion_prevention` | BiSeNet | 0–100, higher is better |
| `head_pose` | MobileNetV1 3DDFAV2 | yaw/pitch/roll angles plus quality scores |

The model cards in [`dardcollect/models/`](../dardcollect/models/) document the
models and their intended use.

## Viewer

`viewer/` indexes the folders with `viewer/index_data.py`, pairing each artifact
with its sidecars by file stem: `.magface.json`, `.ofiq_attr.json` and
`.transcription.json`. Those files are shown alongside the media and are not listed
as artifacts themselves. Per-frame quality values for face-crop videos come from
`frame_data` in `.ofiq_attr.json`.

## Implementation

Sidecars are written by the stage that produces the artifact. Validation happens
at write time against the schema listed above; an invalid sidecar is a
data-integrity failure, not a warning. The writers are in
`dardcollect/face_crop_writers.py`, `dardcollect/quality.py`,
`pipeline/annotate_face_quality.py`, and the transcription and document
pipeline scripts.

← [Back to README](../README.md)
