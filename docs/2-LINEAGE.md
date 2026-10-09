# CSV lineage and traceability

This guide covers the pipeline CSV logs: where they are written, what their
columns mean, and how to follow a file upstream. JSON sidecars are described in
[3-ANNOTATIONS.md](3-ANNOTATIONS.md); their JSON Schemas are the source of truth
for sidecar fields.

## How to read the records

- A CSV row records one download, extraction or processing event. `uuid`
  identifies that **row**.
- `parent_uuid`, `clip_uuid`, `crop_uuid`, `download_uuid` and similar columns
  link a row to an upstream CSV row when that stage has one.
- A JSON sidecar has its own UUID. A JSON parent link points to that sidecar
  UUID, not to the corresponding CSV row UUID. Do not mix the two identifiers.
- `output_path` identifies the artifact produced by a row. Some joins are
  resolved by filename/path rather than UUID; the relevant key is noted below.

The usual chain is:

```text
source manifest → downloaded media → person clip / image detection → face crop → quality annotation
```

For custom datasets, `register_source_files()` creates the source manifest in
place of Archive.org's `downloads.csv`; the rest of the chain is the same.

## CSV files

Paths below are relative to the configured DARD output root. Headers are shown
in canonical order. The logger code owns each fixed header; `downloads.csv` is
the exception because it appends newly encountered Archive.org metadata
columns.

### Source manifests

**Archive.org — `archive_org_public_domain/downloads.csv`**

The fixed columns are followed by Archive.org metadata fields, which vary by
item:

```text
uuid,title,creator,date,license,archive_org_identifier,filename_downloaded,media_type,downloaded_at,download_stage_script,download_stage_timestamp,[Archive.org metadata…]
```

`uuid` identifies the manifest row; `archive_org_identifier` identifies the
Archive.org item. Empty metadata cells are normal because fields vary between
items and media types.

**Custom dataset — configured manifest path**

Created by `register_source_files()`; columns after the fixed values may also
include caller-provided metadata:

```text
uuid,title,creator,date,license,archive_org_identifier,filename_downloaded,media_type,registered_at,source_path,[extra columns…]
```

`archive_org_identifier` is empty for custom sources. The row UUID still anchors
downstream traceability.

### Video and image processing

**Person clips — `extracted_person_clips/clips_extraction.csv`**

```text
uuid,archive_org_identifier,timestamp,source_video,fps,start_frame,end_frame,start_seconds,duration_seconds,max_persons_per_frame,detector_model,detector_confidence,output_path
```

The source download is identified by `archive_org_identifier`; `source_video`
names the original video. The clip sidecar has its own UUID.

**Extracted frames — `extracted_frames/frames_extraction.csv`**

```text
uuid,clip_uuid,timestamp,source_clip_path,frame_number,timestamp_seconds,output_path
```

`clip_uuid` refers to the person-clip extraction row.

**Video face crops — `video_face_crops/video_face_crops_extraction.csv`**

```text
uuid,parent_uuid,timestamp,crop_id,source_type,source_path,face_bbox,confidence,output_path
```

`source_type` is `person_clip` or `image`. For a person clip,
`parent_uuid` refers to its clip-extraction row; for an image, it refers to the
source download row. `crop_id` is the output filename stem and is used by the
filter log to find this row.

**Image person detections — `extracted_image_detections/image_person_detection.csv`**

```text
uuid,download_uuid,timestamp,source_image,source_image_path,num_persons,detector_model,detector_confidence,output_path
```

`download_uuid` points to the source manifest row.

**Image face crops — `image_face_crops/image_face_crops_extraction.csv`**

```text
uuid,detection_uuid,timestamp,source_image_path,bbox_in_source,bbox_confidence,output_path
```

`detection_uuid` points to the image-detection row.

### Transcriptions, filtering and documents

**Video transcriptions — `extracted_person_clips/transcriptions_extraction.csv`**

```text
uuid,clip_uuid,timestamp,source_clip_path,language_detected,word_count,model_version,output_path
```

`clip_uuid` points to the person-clip row. The transcription text and segments
are in the JSON sidecar, not in this CSV.

**Audio transcriptions — `audio_transcriptions/audio_transcriptions_extraction.csv`**

```text
uuid,download_uuid,timestamp,source_audio_path,language_detected,model_version,output_path
```

`download_uuid` points to the source manifest row.

**Quality-filtered crops — `filtered_video_face_crops/video_filtered_face_crops.csv`**
(and `filtered_image_face_crops/image_filtered_face_crops.csv`)

```text
uuid,crop_uuid,timestamp,source_crop_path,magface_score,filter_threshold,output_path
```

`crop_uuid` points to the corresponding crop-extraction row. The CSV stores the
score used for the filtering decision; quality sidecars contain the richer
measurements (see [3-ANNOTATIONS.md](3-ANNOTATIONS.md)).

**Document text — `preprocessed_documents/document_text_extraction.csv`**

```text
uuid,download_uuid,timestamp,source_document_path,text_length,word_count,model_version,output_annotation_path,output_text_path
```

`download_uuid` points to the source manifest row. Extracted text and document
metadata are stored in the output text file and JSON annotation.

**Colour classification — `archive_org_public_domain/videos/color_classification.csv`**
(standalone stage)

```text
uuid,timestamp,video_path,video_name,classification,mean_saturation,frames_sampled,moved,moved_to
```

`classification` is `color`, `black_and_white` or `unreadable`. `moved` and
`moved_to` are populated only when the optional move operation is used.

## Trace a face crop to its source

For a video face crop:

1. Find its row in `video_face_crops_extraction.csv` by `output_path` or
   `crop_id`.
2. Read `parent_uuid` and find that UUID in `clips_extraction.csv`.
3. Read `archive_org_identifier` in the clip row and find the source item in
   `downloads.csv`.

For an image face crop, `detection_uuid` leads to
`image_person_detection.csv`; that row's `download_uuid` leads to the source
manifest.

The face-crop JSON sidecar has a separate `parent_clip.uuid` link to the
parent's JSON sidecar. See [3-ANNOTATIONS.md](3-ANNOTATIONS.md) for that JSON
lineage and the other annotation formats.

## Implementation

CSV headers and row construction are defined by the loggers in
`dardcollect/extraction_logger.py`, `dardcollect/pipeline_loggers.py`,
`dardcollect/modality_loggers.py` and `pipeline/filter_videos_by_color.py`.
CSV files are not validated with JSON Schema. Their field lists and join
behavior are implemented by those writers; sidecar validation is documented in
[3-ANNOTATIONS.md](3-ANNOTATIONS.md).

← [Back to README](../README.md)
