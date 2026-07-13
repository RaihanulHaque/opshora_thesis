"""Hamilton skeleton generation from already-segmented silhouette PNGs.

Decoupled from segment.py on purpose: it only reads silhouette PNGs already
written to disk, so it can be re-run independently (e.g. after tuning the
thinning step) without re-running the much slower YOLO segmentation pass.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from gait.preprocessing import topology_preserving_thinning


@dataclass(slots=True)
class SkeletonResult:
    silhouette_dir: str
    skeleton_dir: str
    frames_processed: int
    empty_skeleton_frames: int
    elapsed_seconds: float


def generate_skeleton_for_sequence(silhouette_dir: Path, skeleton_dir: Path) -> SkeletonResult:
    silhouette_paths = sorted(path for path in silhouette_dir.glob("*.png"))
    skeleton_dir.mkdir(parents=True, exist_ok=True)

    start = time.time()
    frames_processed = 0
    empty_skeleton_frames = 0

    for silhouette_path in silhouette_paths:
        image = cv2.imread(str(silhouette_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"OpenCV could not decode silhouette frame: {silhouette_path}")
        mask = image > 0
        skeleton = topology_preserving_thinning(mask)
        if not skeleton.any():
            empty_skeleton_frames += 1

        skeleton_path = skeleton_dir / f"{silhouette_path.stem}_skeleton.png"
        cv2.imwrite(str(skeleton_path), (skeleton.astype(np.uint8) * 255), [cv2.IMWRITE_PNG_COMPRESSION, 3])
        frames_processed += 1

    elapsed = time.time() - start
    result = SkeletonResult(
        silhouette_dir=str(silhouette_dir),
        skeleton_dir=str(skeleton_dir),
        frames_processed=frames_processed,
        empty_skeleton_frames=empty_skeleton_frames,
        elapsed_seconds=elapsed,
    )

    marker = {
        "silhouette_dir": result.silhouette_dir,
        "frames_processed": result.frames_processed,
        "empty_skeleton_frames": result.empty_skeleton_frames,
        "elapsed_seconds": result.elapsed_seconds,
    }
    (skeleton_dir / "_SKELETON_COMPLETE.json").write_text(json.dumps(marker, indent=2))
    return result
