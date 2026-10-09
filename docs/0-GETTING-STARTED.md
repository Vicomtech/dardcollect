# Getting started

This guide takes you from a fresh clone to a first run. For configuration keys
see [7-CONFIG.md](7-CONFIG.md); for GPU drivers, tests and tuning see
[4-DEVELOPMENT.md](4-DEVELOPMENT.md).

## 1. Install

Requirements: **Python 3.12** and [`uv`](https://docs.astral.sh/uv/getting-started/installation/).
An NVIDIA GPU is optional; without one the pipeline runs on CPU. Driver versions
are listed in [4-DEVELOPMENT.md](4-DEVELOPMENT.md#driver-requirements-cuda-121).

```bash
git clone https://github.com/Vicomtech/dardcollect.git
cd dardcollect
uv sync
```

`uv sync` creates `.venv` and installs all dependencies. Run commands with
`uv run python …`, or activate `.venv` first.

## 2. Choose your data

**A. Archive.org.** Set the search query and media types in
`configs/config.archive_all.yaml`, then download with the first stage:

```bash
uv run python scripts/run_pipeline.py
```

**B. Your own dataset.** Put files in the folders the config points to, skip the
download stage and reuse the processing stages:

```yaml
# configs/config.mydata.yaml
root: "/data/mydata"              # every {root} path resolves from here
run_pipeline:
  skip_download: true
person_extraction:
  input_dir: "{root}/videos"
```

```bash
uv run python scripts/run_pipeline.py --config configs/config.mydata.yaml
```

To keep provenance for files that did not come from Archive.org, register them
with a source manifest. See
[5-LIBRARY-API.md](5-LIBRARY-API.md#11-use-dardcollect-with-your-own-data-custom-source).

## 3. Run stages

`scripts/run_pipeline.py` runs the stages in order, each as soon as its inputs
exist. The table below lists them in run order.

To run a single stage, call its script directly. It reads
`configs/config.archive_all.yaml`, or the file in `DARDCOLLECT_CONFIG`:

```bash
uv run python pipeline/extract_person_clips_from_videos.py
DARDCOLLECT_CONFIG=configs/config.mydata.yaml uv run python pipeline/extract_persons_from_images.py
```

Stages are resumable: rerunning skips finished outputs. Set the stage's
`overwrite: true` (see [7-CONFIG.md](7-CONFIG.md)) to recompute them.

### Stages

| Alias | Script | Writes (under `root`) |
|---|---|---|
| `download` | `pipeline/download_media_from_archive.py` | `archive_org_public_domain/<type>/`, `downloads.csv` |
| `clips` | `extract_person_clips_from_videos.py` | `extracted_person_clips/*.mp4` + `.json`, `clips_extraction.csv` |
| `audio_clips` | `extract_audio_from_clips.py` | `extracted_person_clips/*.wav` |
| `images` | `extract_persons_from_images.py` | `extracted_image_detections/*.json`, `image_person_detection.csv` |
| `face_crops_video` | `extract_face_crops_from_videos.py` | `video_face_crops/*.mp4` + `.json`, `video_face_crops_extraction.csv` |
| `face_crops_image` | `extract_face_crops_from_images.py` | `image_face_crops/*.jpg` + `.json`, `image_face_crops_extraction.csv` |
| `transcribe_video` | `transcribe_video_clips.py` | `*.transcription.json` beside clips, `transcriptions_extraction.csv` |
| `transcribe_audio` | `transcribe_audio_files.py` | `audio_transcriptions/`, `audio_transcriptions_extraction.csv` |
| `docs` | `extract_text_from_doc.py` | `preprocessed_documents/*.text.txt` + `.annotation.json`, `document_text_extraction.csv` |
| `quality` | `annotate_face_quality.py` | `*.ofiq_attr.json` and `*.magface.json` beside crops |
| `filter` | `filter_face_crops_by_quality.py` | `filtered_video_face_crops/` (or `filtered_image_face_crops/`), `*_filtered_face_crops.csv` |
| `frames` | `extract_frames_from_videos.py` | `extracted_frames/` (PNG + sidecars), `frames_extraction.csv` |
| `masks` | `generate_face_masks.py` | `*_mask.png` beside crops and frames |

Stages depend on each other: `frames` needs `filter`, and `masks` needs crops
and frames. The aliases can be listed in `run_pipeline.skip_stages`; a skipped
stage also skips every stage that depends on it. The color filter
(`pipeline/filter_videos_by_color.py`) is standalone and not in the table.

## 4. Check the outputs

Each stage appends one row per output to its CSV log, next to the files it
wrote:

```bash
tail -5 DARD/archive_org_public_domain/downloads.csv
tail -5 DARD/extracted_person_clips/clips_extraction.csv
tail -5 DARD/video_face_crops/video_face_crops_extraction.csv
tail -5 DARD/filtered_video_face_crops/video_filtered_face_crops.csv
```

What each CSV contains, and how the rows link to one another, is in
[2-LINEAGE.md](2-LINEAGE.md). The JSON sidecars are in
[3-ANNOTATIONS.md](3-ANNOTATIONS.md).

## Troubleshooting

- **No GPU found, or CUDA unavailable.** The pipeline falls back to CPU
  automatically. Check GPU setup in [4-DEVELOPMENT.md](4-DEVELOPMENT.md#gpu-configuration-optional).
- **Config error.** The message names the failing key. Check it against
  [7-CONFIG.md](7-CONFIG.md); some keys have no default and must be set.
- **`ModuleNotFoundError` for `dardcollect`.** Run through `uv run …` after
  `uv sync`, or activate `.venv`.
- **Disk fills up during a run.** See the disk-budget notes in
  [4-DEVELOPMENT.md](4-DEVELOPMENT.md#storage).

← [Back to README](../README.md)
