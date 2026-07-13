"""YOLO-seg person segmentation with simple frame-to-frame tracking.

Turns one raw walking video into a sequence of clean binary silhouette PNGs,
following DATASET_DEVELOPMENT_PIPELINE.md section 5's segmentation rules:
keep the main walking subject (matching the previous frame's track when more
than one person is detected), fill small holes, remove small blobs, save as
PNG (never JPG).
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from gait.config import ExperimentConfig
from gait.preprocessing import clean_and_align

DEFAULT_WEIGHTS = "yolo26m-seg.pt"
_MODEL_CACHE: dict[str, object] = {}


def load_model(weights_path: str = DEFAULT_WEIGHTS):
    if weights_path not in _MODEL_CACHE:
        from ultralytics import YOLO

        _MODEL_CACHE[weights_path] = YOLO(weights_path)
    return _MODEL_CACHE[weights_path]


def _mask_centroid(mask: np.ndarray) -> tuple[float, float] | None:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    return float(ys.mean()), float(xs.mean())


def select_tracked_mask(
    masks: list[np.ndarray],
    previous_centroid: tuple[float, float] | None,
) -> tuple[np.ndarray | None, tuple[float, float] | None]:
    """Pick which detected person-mask belongs to the tracked walking subject.

    First tracked frame (no previous centroid): pick the largest-area mask,
    since the primary subject is assumed dominant at the start of a
    single-subject walking clip. Later frames: pick whichever mask's
    centroid is closest to the previous kept centroid, so a momentarily
    occluded/mis-detected frame doesn't reset the track.
    """
    if not masks:
        return None, previous_centroid

    if previous_centroid is None:
        best = max(masks, key=lambda mask: int(mask.sum()))
        return best, _mask_centroid(best)

    candidates = [(mask, _mask_centroid(mask)) for mask in masks]
    candidates = [(mask, centroid) for mask, centroid in candidates if centroid is not None]
    if not candidates:
        return None, previous_centroid

    def distance(centroid: tuple[float, float]) -> float:
        return (centroid[0] - previous_centroid[0]) ** 2 + (centroid[1] - previous_centroid[1]) ** 2

    best_mask, best_centroid = min(candidates, key=lambda item: distance(item[1]))
    return best_mask, best_centroid


@dataclass(slots=True)
class SegmentationResult:
    source_video: str
    frames_read: int
    frames_kept: int
    frames_dropped: int
    multi_person_frames: int
    elapsed_seconds: float
    output_dir: str
    source_fps: float
    sample_fps: float | None
    stride: int


def _frame_masks(result, frame_shape: tuple[int, int]) -> list[np.ndarray]:
    if result.masks is None:
        return []
    masks: list[np.ndarray] = []
    for mask in result.masks.data:
        mask_np = mask.cpu().numpy()
        resized = cv2.resize(mask_np, (frame_shape[1], frame_shape[0]), interpolation=cv2.INTER_LINEAR)
        masks.append((resized > 0.5).astype(np.uint8))
    return masks


def segment_video(
    video_path: Path,
    output_dir: Path,
    config: ExperimentConfig,
    subject: str,
    condition: str,
    view: str,
    weights_path: str = DEFAULT_WEIGHTS,
    sample_fps: float | None = None,
) -> SegmentationResult:
    """Segment one video into binary silhouette PNGs.

    If sample_fps is set (e.g. 5), only ~sample_fps frames per second of
    source video are decoded and run through YOLO; the rest are skipped via
    cv2's cheap grab() (no full decode), so a lower sample rate is also a
    real speedup, not just fewer output files.
    """
    model = load_model(weights_path)
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    source_fps = capture.get(cv2.CAP_PROP_FPS) or 0.0
    stride = 1
    if sample_fps and source_fps > 0:
        stride = max(1, round(source_fps / sample_fps))

    output_dir.mkdir(parents=True, exist_ok=True)
    start = time.time()
    source_frame_index = 0
    frames_processed = 0
    frames_kept = 0
    multi_person_frames = 0
    previous_centroid: tuple[float, float] | None = None

    while True:
        if source_frame_index % stride != 0:
            if not capture.grab():
                break
            source_frame_index += 1
            continue

        success, frame = capture.read()
        if not success:
            break
        source_frame_index += 1
        frames_processed += 1

        results = model(frame, classes=[0], verbose=False)
        masks = _frame_masks(results[0], frame.shape[:2])
        if len(masks) > 1:
            multi_person_frames += 1

        chosen, previous_centroid = select_tracked_mask(masks, previous_centroid)
        if chosen is None:
            continue

        silhouette = clean_and_align(chosen.astype(bool), config)
        filename = f"{subject}-{condition}-{view}-{frames_kept + 1:06d}.png"
        cv2.imwrite(str(output_dir / filename), silhouette * 255, [cv2.IMWRITE_PNG_COMPRESSION, 3])
        frames_kept += 1

    capture.release()
    elapsed = time.time() - start
    result = SegmentationResult(
        source_video=str(video_path),
        frames_read=frames_processed,
        frames_kept=frames_kept,
        frames_dropped=frames_processed - frames_kept,
        multi_person_frames=multi_person_frames,
        elapsed_seconds=elapsed,
        output_dir=str(output_dir),
        source_fps=source_fps,
        sample_fps=sample_fps,
        stride=stride,
    )

    marker = {
        "source_video": result.source_video,
        "frames_read": result.frames_read,
        "frames_kept": result.frames_kept,
        "frames_dropped": result.frames_dropped,
        "multi_person_frames": result.multi_person_frames,
        "elapsed_seconds": result.elapsed_seconds,
        "source_fps": result.source_fps,
        "sample_fps": result.sample_fps,
        "stride": result.stride,
    }
    (output_dir / "_SEGMENT_COMPLETE.json").write_text(json.dumps(marker, indent=2))
    return result
