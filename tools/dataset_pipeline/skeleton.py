"""Hamilton-Jacobi skeleton generation from already-segmented silhouette PNGs.

Decoupled from segment.py on purpose: it only reads silhouette PNGs already
written to disk, so it can be re-run independently (e.g. after tuning the
skeleton extraction step) without re-running the much slower YOLO
segmentation pass.

Uses opshora_archive/phase1_hamilton.py's run_hamilton_pipeline() -- inward
Fast Marching distance field -> average outward flux -> homotopy-preserving
thinning (Siddiqi et al., "Hamilton-Jacobi Skeleton") -- so the branch
structure matches datasets/CASIA_B_Hamilton_Skeleton's style. This is a
genuine Hamilton-Jacobi skeleton, not the simpler Zhang-Suen-style
topology_preserving_thinning() in gait/preprocessing.py (that function is
used elsewhere for an on-the-fly topology heatmap, not for CLoP-Gait).
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from opshora_archive.phase1_hamilton import run_hamilton_pipeline  # noqa: E402


@dataclass(slots=True)
class SkeletonResult:
    silhouette_dir: str
    skeleton_dir: str
    frames_processed: int
    empty_skeleton_frames: int
    elapsed_seconds: float


def generate_skeleton_for_sequence(
    silhouette_dir: Path,
    skeleton_dir: Path,
    progress_desc: str | None = None,
    progress_position: int = 2,
) -> SkeletonResult:
    silhouette_paths = sorted(path for path in silhouette_dir.glob("*.png"))
    skeleton_dir.mkdir(parents=True, exist_ok=True)

    start = time.time()
    frames_processed = 0
    empty_skeleton_frames = 0

    for silhouette_path in tqdm(
        silhouette_paths,
        desc=f"    {progress_desc or skeleton_dir.name}",
        unit="frame",
        position=progress_position,
        leave=False,
    ):
        image = cv2.imread(str(silhouette_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"OpenCV could not decode silhouette frame: {silhouette_path}")
        skeleton = run_hamilton_pipeline(image)["hamilton_skeleton"]
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
