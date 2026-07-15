"""YOLO-seg person segmentation with simple frame-to-frame tracking.

Turns one raw walking video into a sequence of clean binary silhouette PNGs,
following DATASET_DEVELOPMENT_PIPELINE.md section 5's segmentation rules:
keep the main walking subject (matching the previous frame's track when more
than one person is detected), fill small holes, remove small blobs, save as
PNG (never JPG).

Stores silhouettes with a natural margin around the subject (like CASIA-B's
raw GaitDatasetB-silh storage: person occupies a modest fraction of the
frame, not cropped tight to their bounding box) rather than baking in the
tight fill-canvas normalization -- that step (gait.preprocessing.clean_and_align)
already runs at V6 training-cache-build time, exactly as it does for real
CASIA-B. This only changes how the stored dataset looks on disk, not what
V6 actually trains on.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from gait.config import ExperimentConfig
from gait.preprocessing import clean_silhouette_mask

DEFAULT_WEIGHTS = "yolo26m-seg.pt"
DEFAULT_IMGSZ = 1280
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


def crop_with_margin(
    mask: np.ndarray,
    height: int,
    width: int,
    fill_ratio: float = 0.45,
) -> np.ndarray:
    """Crop to the subject's bbox with generous margin and store at a fixed canvas.

    Unlike gait.preprocessing.crop_to_canvas (tight fill, margin=2px, used at
    V6 training time), this scales the subject to occupy only fill_ratio of
    the canvas height (default 0.45), leaving natural headroom/footroom --
    matching how CASIA-B's raw GaitDatasetB-silh silhouettes look before any
    training-time alignment. Always uses area-average downscaling (proper
    anti-aliasing for the large full-HD-to-canvas reduction), not NEAREST.
    """
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return np.zeros((height, width), dtype=np.uint8)
    cropped = mask[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1].astype(np.uint8)
    scale = (height * fill_ratio) / cropped.shape[0]
    new_h = max(1, round(cropped.shape[0] * scale))
    new_w = max(1, round(cropped.shape[1] * scale))
    if new_w > width:
        scale = width / cropped.shape[1]
        new_h = max(1, round(cropped.shape[0] * scale))
        new_w = width
    soft = cv2.resize(cropped * 255, (new_w, new_h), interpolation=cv2.INTER_AREA)
    resized = (soft > 127).astype(np.uint8)
    canvas = np.zeros((height, width), dtype=np.uint8)
    y0 = (height - new_h) // 2
    x0 = (width - new_w) // 2
    canvas[y0 : y0 + new_h, x0 : x0 + new_w] = resized
    return canvas


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
    imgsz: int
    fill_ratio: float


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
    progress_position: int = 2,
    imgsz: int = DEFAULT_IMGSZ,
    fill_ratio: float = 0.45,
) -> SegmentationResult:
    """Segment one video into binary silhouette PNGs.

    If sample_fps is set (e.g. 5), only ~sample_fps frames per second of
    source video are decoded and run through YOLO; the rest are skipped via
    cv2's cheap grab() (no full decode), so a lower sample rate is also a
    real speedup, not just fewer output files.

    imgsz controls YOLO's inference resolution -- Ultralytics' returned masks
    are already hard-binarized at a resolution proportional to imgsz, so the
    default 640 produces a coarse ~384x640 mask for our 1920x1080 source that
    aliases into blocky limb edges. Raising it to 1280 (default here) gives
    visibly cleaner mask detail at ~3.3x the per-frame YOLO cost.
    """
    model = load_model(weights_path)
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    source_fps = capture.get(cv2.CAP_PROP_FPS) or 0.0
    stride = 1
    if sample_fps and source_fps > 0:
        stride = max(1, round(source_fps / sample_fps))

    total_source_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    expected_sampled_frames = (total_source_frames + stride - 1) // stride if total_source_frames else None

    output_dir.mkdir(parents=True, exist_ok=True)
    start = time.time()
    source_frame_index = 0
    frames_processed = 0
    frames_kept = 0
    multi_person_frames = 0
    previous_centroid: tuple[float, float] | None = None

    frame_bar = tqdm(
        total=expected_sampled_frames,
        desc=f"    {subject}/{condition}/{view}",
        unit="frame",
        position=progress_position,
        leave=False,
    )
    try:
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
            frame_bar.update(1)

            results = model(frame, classes=[0], verbose=False, imgsz=imgsz)
            masks = _frame_masks(results[0], frame.shape[:2])
            if len(masks) > 1:
                multi_person_frames += 1

            chosen, previous_centroid = select_tracked_mask(masks, previous_centroid)
            if chosen is None:
                continue

            cleaned = clean_silhouette_mask(chosen.astype(bool), config)
            silhouette = crop_with_margin(cleaned, config.height, config.width, fill_ratio=fill_ratio)
            filename = f"{subject}-{condition}-{view}-{frames_kept + 1:06d}.png"
            cv2.imwrite(str(output_dir / filename), silhouette * 255, [cv2.IMWRITE_PNG_COMPRESSION, 3])
            frames_kept += 1
            frame_bar.set_postfix(kept=frames_kept, multi=multi_person_frames)
    finally:
        frame_bar.close()

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
        imgsz=imgsz,
        fill_ratio=fill_ratio,
    )

    marker = {
        "source_video": result.source_video,
        "frames_read": result.frames_read,
        "frames_kept": result.frames_kept,
        "frames_dropped": result.frames_dropped,
        "multi_person_frames": result.multi_person_frames,
        "elapsed_seconds": result.elapsed_seconds,
        "source_fps": result.source_fps,
        "imgsz": result.imgsz,
        "fill_ratio": result.fill_ratio,
        "sample_fps": result.sample_fps,
        "stride": result.stride,
    }
    (output_dir / "_SEGMENT_COMPLETE.json").write_text(json.dumps(marker, indent=2))
    return result
