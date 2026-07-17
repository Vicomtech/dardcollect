#!/usr/bin/env python3
"""Benchmark individual pipeline components to identify bottlenecks.

Usage::

    # Full benchmark (GPU + real video inputs)
    uv run python scripts/benchmark_pipeline.py --config configs/config.custom_videos.yaml

    # Model-load timings only (fast, no video needed)
    uv run python scripts/benchmark_pipeline.py --config configs/config.custom_videos.yaml \
        --only-models

    # Network/local I/O throughput only (no GPU)
    uv run python scripts/benchmark_pipeline.py --config configs/config.custom_videos.yaml \
        --only-io

Output: timing table to stdout + JSON summary to benchmark_results.json (gitignored).

What this measures (and why each matters for the optimization plan):

  [IO]   network_read_fps     – cv2 frames/s over the source path (network vs local)
  [IO]   local_copy_speed_mbs – MB/s achieved by shutil.copy2 to local_cache_dir
  [GPU]  detection_fps        – YOLOX inference fps (pure inference, no decode)
  [GPU]  pose_fps             – CIGPose inference fps (per-crop, single track)
  [GPU]  magface_fps          – MagFace inference fps on OFIQ 616x616 frames
  [CLIP] clip_extract_s       – moviepy/ffmpeg encode time per second of source video
  [CLIP] parallel_speedup     – ratio of serial vs parallel clip extraction (3 clips)
  [MODEL]model_load_s         – cold ONNX model load time for det+pose, magface, quality

Reading the results:

  If network_read_fps << local_read_fps  → preload_source_to_local is the right lever.
  If detection_fps * frame_count >> total_clips_time → moviepy is the bottleneck, not GPU.
  If model_load_s is large relative to stage run time → DEFER_UNTIL_DEPS_DONE is critical.
  If parallel_speedup ≈ N clips → parallel_clip_extraction delivers full benefit.
  If detection_fps is low even on local SSD → GPU is the bottleneck, consider batch>1.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

# ── helpers ───────────────────────────────────────────────────────────────────

_RESULTS: dict[str, float | str] = {}


def _ms(seconds: float) -> str:
    return f"{seconds * 1000:.1f} ms"


def _fps(frames: int, seconds: float) -> str:
    return f"{frames / seconds:.1f} fps" if seconds > 0 else "n/a"


def _mbs(bytes_: int, seconds: float) -> str:
    return f"{bytes_ / 1e6 / seconds:.1f} MB/s" if seconds > 0 else "n/a"


def _header(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print("=" * 60)


def _row(label: str, value: str, note: str = "") -> None:
    note_str = f"  # {note}" if note else ""
    print(f"  {label:<35} {value}{note_str}")
    _RESULTS[label] = value


# ── benchmarks ────────────────────────────────────────────────────────────────

def bench_model_load(config_path: Path, gpu_id: int) -> None:
    """Measure cold load time for each ONNX model group."""
    _header("MODEL LOAD TIMES (cold, first launch)")

    from dardcollect.gpu_setup import setup_gpu_paths
    setup_gpu_paths(str(config_path))

    models_dir = REPO_ROOT / "dardcollect" / "models"

    # Detection + pose (used by clips stage)
    t0 = time.perf_counter()
    from dardcollect import PersonDetector, PoseEstimator
    from dardcollect.config import DetectorConfig
    det_config = DetectorConfig.from_yaml(str(config_path))
    det_model = models_dir / "yolox_tiny_8xb8-300e_humanart-6f3252f9.onnx"
    pose_model = models_dir / "cigpose-m_coco-wholebody_256x192.onnx"
    if det_model.exists():
        PersonDetector(det_config, model_path=str(det_model))
    if pose_model.exists():
        PoseEstimator(det_config, model_path=str(pose_model))
    det_load = time.perf_counter() - t0
    _row("det+pose model load", f"{det_load:.2f}s",
         "paid once per clips run; DEFER avoids re-paying per rerun")

    # MagFace (used by filter stage)
    t0 = time.perf_counter()
    from dardcollect.magface import load_magface
    magface_model = models_dir / "magface_iresnet50_norm.onnx"
    if magface_model.exists():
        load_magface(str(magface_model), gpu_id)
    magface_load = time.perf_counter() - t0
    _row("magface model load", f"{magface_load:.2f}s",
         "paid once per filter run; DEFER avoids re-paying per rerun")

    # OFIQ quality stack (used by quality stage)
    t0 = time.perf_counter()
    from dardcollect.quality import load_models
    quality_models = load_models(models_dir, gpu_id)
    quality_load = time.perf_counter() - t0
    _row("OFIQ quality stack load", f"{quality_load:.2f}s",
         "paid once per quality run; DEFER avoids re-paying per rerun")
    del quality_models


def bench_io(config_path: Path) -> None:
    """Measure frame read throughput on network vs local path."""
    _header("I/O THROUGHPUT (cv2 frame decode)")
    import cv2

    from dardcollect.config import ClipExtractionConfig
    clip_config = ClipExtractionConfig.from_yaml(str(config_path))

    input_dir = Path(clip_config.input_dir)
    video_files = []
    for ext in ("*.mp4", "*.avi", "*.mkv", "*.mov", "*.webm"):
        video_files.extend(input_dir.rglob(ext))
    if not video_files:
        print("  [skip] No video files found in input_dir")
        return

    # Pick smallest file to keep this benchmark fast
    src = min(video_files, key=lambda p: p.stat().st_size)
    src_size_mb = src.stat().st_size / 1e6
    print(f"  Source: {src.name}  ({src_size_mb:.1f} MB)")

    # Network read rate
    BENCH_FRAMES = 300
    cap = cv2.VideoCapture(str(src))
    frames = 0
    t0 = time.perf_counter()
    while cap.isOpened() and frames < BENCH_FRAMES:
        ret, _ = cap.read()
        if not ret:
            break
        frames += 1
    network_s = time.perf_counter() - t0
    cap.release()
    _row("network read rate", _fps(frames, network_s),
         f"{frames} frames from {src.parent.parent.name}/.../{src.name[:30]}")

    # Local copy + local read rate
    local_dir = (
        Path(clip_config.local_cache_dir)
        if clip_config.local_cache_dir
        else Path(tempfile.gettempdir())
    )
    local_dir.mkdir(parents=True, exist_ok=True)
    dst = local_dir / src.name

    import shutil
    t0 = time.perf_counter()
    shutil.copy2(src, dst)
    copy_s = time.perf_counter() - t0
    _row("local copy speed", _mbs(src.stat().st_size, copy_s),
         "shutil.copy2 → local_cache_dir")

    cap = cv2.VideoCapture(str(dst))
    frames = 0
    t0 = time.perf_counter()
    while cap.isOpened() and frames < BENCH_FRAMES:
        ret, _ = cap.read()
        if not ret:
            break
        frames += 1
    local_s = time.perf_counter() - t0
    cap.release()
    _row("local read rate", _fps(frames, local_s),
         "same file from local SSD (after copy)")

    speedup = (network_s / local_s) if local_s > 0 else float("inf")
    _row("I/O speedup (local vs network)", f"{speedup:.1f}x",
         "> 2x → preload_source_to_local is worth it")

    try:
        dst.unlink()
    except OSError:
        pass


def bench_detection(config_path: Path) -> None:
    """Measure detection + pose inference fps on synthetic frames."""
    _header("GPU INFERENCE (synthetic 1080p frames)")

    from dardcollect.gpu_setup import setup_gpu_paths
    setup_gpu_paths(str(config_path))

    from dardcollect import PersonDetector, PoseEstimator
    from dardcollect.config import DetectorConfig

    models_dir = REPO_ROOT / "dardcollect" / "models"
    det_config = DetectorConfig.from_yaml(str(config_path))
    det_model = models_dir / "yolox_tiny_8xb8-300e_humanart-6f3252f9.onnx"
    pose_model = models_dir / "cigpose-m_coco-wholebody_256x192.onnx"

    if not det_model.exists() or not pose_model.exists():
        print("  [skip] Model files not found")
        return

    detector = PersonDetector(det_config, model_path=str(det_model))
    poser = PoseEstimator(det_config, model_path=str(pose_model))

    # Warmup
    frame_1080p = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
    for _ in range(3):
        detector.get_detections(frame_1080p)

    N = 50
    t0 = time.perf_counter()
    for _ in range(N):
        detector.get_detections(frame_1080p)
    det_s = time.perf_counter() - t0
    _row("detection fps (1080p, batch=1)", _fps(N, det_s),
         f"{_ms(det_s / N)} per frame — bottleneck if << network_read_rate")

    # Pose on a synthetic crop-sized array
    crop_128 = np.random.randint(0, 255, (256, 192, 3), dtype=np.uint8)
    for _ in range(3):
        poser.get_keypoints(crop_128, [0, 0, 192, 256])

    t0 = time.perf_counter()
    for _ in range(N):
        poser.get_keypoints(crop_128, [0, 0, 192, 256])
    pose_s = time.perf_counter() - t0
    _row("pose fps (192x256 crop, batch=1)", _fps(N, pose_s),
         f"{_ms(pose_s / N)} per crop")

    total_inference = det_s + pose_s
    _row("combined det+pose fps (1 person)", _fps(N, total_inference),
         "real throughput with 1 detection per frame")


def bench_magface(config_path: Path, gpu_id: int) -> None:
    """Measure MagFace scoring throughput on synthetic OFIQ frames."""
    _header("MAGFACE THROUGHPUT (synthetic 616x616 OFIQ frames)")

    from dardcollect.gpu_setup import setup_gpu_paths
    setup_gpu_paths(str(config_path))
    from dardcollect.magface import load_magface

    models_dir = REPO_ROOT / "dardcollect" / "models"
    magface_model = models_dir / "magface_iresnet50_norm.onnx"
    if not magface_model.exists():
        print("  [skip] magface_iresnet50_norm.onnx not found")
        return

    session = load_magface(str(magface_model), gpu_id)

    # OFIQ 616x616 frame → ArcFace crop is 112x112
    frame_ofiq = np.random.randint(0, 255, (616, 616, 3), dtype=np.uint8)

    # Warmup
    for _ in range(3):
        from dardcollect.face_geometry import arcface_from_ofiq_frame
        crop = arcface_from_ofiq_frame(frame_ofiq)
        input_name = session.get_inputs()[0].name
        inp = crop.astype(np.float32).transpose(2, 0, 1)[None] / 255.0
        session.run(None, {input_name: inp})

    N = 100
    t0 = time.perf_counter()
    for _ in range(N):
        crop = arcface_from_ofiq_frame(frame_ofiq)
        inp = crop.astype(np.float32).transpose(2, 0, 1)[None] / 255.0
        session.run(None, {input_name: inp})
    elapsed = time.perf_counter() - t0
    _row("MagFace fps (112x112 crop, batch=1)", _fps(N, elapsed),
         f"{_ms(elapsed / N)} per frame")


def bench_clip_extraction(config_path: Path) -> None:
    """Compare serial vs parallel clip extraction on a real source video."""
    _header("CLIP EXTRACTION (moviepy/ffmpeg encode time)")
    import cv2

    from dardcollect.config import ClipExtractionConfig
    from dardcollect.pipeline_utils import extract_clip

    clip_config = ClipExtractionConfig.from_yaml(str(config_path))
    input_dir = Path(clip_config.input_dir)
    video_files = sorted(
        [p for ext in ("*.mp4", "*.avi", "*.mkv", "*.mov", "*.webm")
         for p in input_dir.rglob(ext)],
        key=lambda p: p.stat().st_size,
    )
    if not video_files:
        print("  [skip] No video files found")
        return

    src = video_files[0]
    cap = cv2.VideoCapture(str(src))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    print(f"  Source: {src.name}  ({src.stat().st_size / 1e6:.1f} MB, {total/fps:.1f}s)")

    # Extract a single 10-second segment
    seg_frames = min(int(10 * fps), total - 1)
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "bench_clip.mp4"
        t0 = time.perf_counter()
        success = extract_clip(src, out, 0, seg_frames, fps)
        clip_s = time.perf_counter() - t0
        seg_dur = seg_frames / fps
        if success:
            _row("clip encode time / source-second", f"{clip_s / seg_dur:.2f}s/s",
                 f"extracted {seg_dur:.1f}s segment in {clip_s:.2f}s")
        else:
            print("  [skip] extract_clip failed")

    # Parallel extraction: 3 independent segments
    if total >= 90 and fps > 0:
        from concurrent.futures import ThreadPoolExecutor
        segments = [
            (0, int(10 * fps)),
            (int(10 * fps), int(20 * fps)),
            (int(20 * fps), int(30 * fps)),
        ]
        with tempfile.TemporaryDirectory() as td:
            paths = [Path(td) / f"bench_{i}.mp4" for i in range(3)]
            # Serial
            t0 = time.perf_counter()
            for (s, e), p in zip(segments, paths):
                extract_clip(src, p, s, e, fps)
            serial_s = time.perf_counter() - t0

            # Parallel
            paths2 = [Path(td) / f"bench_p{i}.mp4" for i in range(3)]
            t0 = time.perf_counter()
            with ThreadPoolExecutor(max_workers=3) as ex:
                futs = [ex.submit(extract_clip, src, p, s, e, fps)
                        for (s, e), p in zip(segments, paths2)]
                for f in futs:
                    f.result()
            parallel_s = time.perf_counter() - t0

            speedup = serial_s / parallel_s if parallel_s > 0 else float("inf")
            _row("clip extract serial (3 clips)", f"{serial_s:.2f}s",
                 "moviepy serial baseline")
            _row("clip extract parallel (3 clips)", f"{parallel_s:.2f}s",
                 "ThreadPoolExecutor(3)")
            _row("parallel speedup", f"{speedup:.2f}x",
                 "≈ 3x ideal; > 1.5x → parallel_clip_extraction is worth it")


# ── summary + plan ────────────────────────────────────────────────────────────

_OPTIMIZATION_DECISION_TREE = """
┌─────────────────────────────────────────────────────────────────────────┐
│  DARDcollect — optimization decision tree (based on benchmark results)  │
└─────────────────────────────────────────────────────────────────────────┘

1. I/O speedup > 2x?
   YES → preload_source_to_local: true  (already available in config)
    └─ also set local_cache_dir to a fast local SSD path outside input_dir

2. After preload: detection_fps > network_read_fps * 1.5?
   YES → GPU has headroom; bottleneck is CPU decode
    └─ readahead_decode: true  (already available in config)

3. parallel speedup > 1.5x?
   YES → parallel_clip_extraction: true  (already available)
    └─ tune max_extraction_workers to match your CPU core count

4. model_load_s > stage run time / 5  (model load > 20% of total)?
    # YES -> DEFER_UNTIL_DEPS_DONE prevents the N-reload problem
    └─ already set for quality, filter, transcribe_video
    └─ do NOT add face_crops_video or masks here — they benefit from overlap

5. detection_fps still bottleneck after all above?
   → Batch inference (batch > 1): requires TRT re-export or dynamic profiles
   → NOT pre-approved; assess golden-baseline risk before implementing
   → open issue: "batched TRT inference" with measured fps at batch=1 first

6. quality/filter wall time bottleneck at scale?
   → Persistent GPU worker + semaphore (one model set resident, loop over inputs)
   → NOT pre-approved; implement only with measured wall-time justification
"""


def print_plan(results: dict) -> None:
    print("\n" + _OPTIMIZATION_DECISION_TREE)
    print("\n── Levers already available in config (no code change needed) ──")
    levers = [
        ("preload_source_to_local: true", "person_extraction"),
        ("local_cache_dir: {output_root}/.source_cache", "person_extraction"),
        ("readahead_decode: true", "person_extraction"),
        ("readahead_queue_frames: 32", "person_extraction"),
        ("parallel_clip_extraction: true", "person_extraction"),
        ("max_extraction_workers: 3", "person_extraction"),
        ("rerun_interval_seconds: 20", "run_pipeline"),
        ("heartbeat_interval_seconds: 10", "run_pipeline"),
    ]
    for key, section in levers:
        print(f"  {section}.{key}")

    print("\n── Levers requiring code change (deferred, measure first) ──")
    print("  batch > 1 TRT inference: re-export ONNX with dynamic batch axis")
    print("  persistent worker with GPU semaphore: complex lifecycle, VRAM risk")


# ── entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="configs/config.custom_videos.yaml")
    parser.add_argument("--only-models", action="store_true",
                        help="Only run model load timings (fast)")
    parser.add_argument("--only-io", action="store_true",
                        help="Only run I/O throughput (no GPU)")
    parser.add_argument("--output", default="benchmark_results.json",
                        help="JSON output path (default: benchmark_results.json)")
    args = parser.parse_args()

    config_path = (REPO_ROOT / args.config).resolve()
    if not config_path.exists():
        print(f"Config not found: {config_path}")
        sys.exit(1)

    # Read gpu_id from config
    try:
        import yaml
        with open(config_path, encoding="utf-8") as f:
            _cfg = yaml.safe_load(f) or {}
        gpu_id = int(_cfg.get("gpu_id", 0))
    except Exception:
        gpu_id = 0

    print(f"[benchmark] config: {args.config}")
    print(f"[benchmark] gpu_id: {gpu_id}")

    if args.only_models:
        bench_model_load(config_path, gpu_id)
    elif args.only_io:
        bench_io(config_path)
    else:
        bench_model_load(config_path, gpu_id)
        bench_io(config_path)
        bench_detection(config_path)
        bench_magface(config_path, gpu_id)
        bench_clip_extraction(config_path)

    print_plan(_RESULTS)

    out = REPO_ROOT / args.output
    with open(out, "w", encoding="utf-8") as f:
        json.dump(_RESULTS, f, indent=2)
    print(f"\n[benchmark] Results saved → {out}")


if __name__ == "__main__":
    main()
