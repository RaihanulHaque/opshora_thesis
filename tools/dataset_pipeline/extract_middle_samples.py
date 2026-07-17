"""One-off script: extract middle-of-sequence sample triplets (original video
frame + matching silhouette + matching skeleton, each at its own native
resolution) for a handful of CLoP-Gait subject/domain/view combinations,
for thesis figure use.

Re-runs the exact same YOLO detection + tracking loop as
`tools/dataset_pipeline/segment.py::segment_video` (same weights, same
`select_tracked_mask` logic, same stride) so that the Nth "kept" frame this
script finds lines up exactly with the existing Nth silhouette/skeleton PNG
already on disk -- this is required because the on-disk filename counter is
a *kept-frame* index, not the raw video frame index (frames where person
detection failed are skipped, so the two indices diverge whenever
`frames_dropped > 0`, e.g. the outdoor-night clip below).
"""

from __future__ import annotations

import shutil
from pathlib import Path

import cv2

from tools.dataset_pipeline.segment import DEFAULT_IMGSZ, _frame_masks, load_model, select_tracked_mask

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_ROOT = REPO_ROOT / "papers" / "samples" / "clop_gait_samples"

TARGETS = [
    {
        "label": "sub04_outdoor_day_090",
        "subject": "004",
        "sil_dir": REPO_ROOT / "datasets/CLoPGaitSilhouettes/004/nm-01-od/090",
        "skel_dir": REPO_ROOT / "datasets/CLoPGaitHamiltonSkeleton/004/nm-01-od/090",
        "video": REPO_ROOT / "datasets/CLoP-Gait/OUTDOOR/Outdoor_Day/Sub04_OD/Nm1_04_OD/Nm1_04_OD_90/Nm1_04_OD_90.mp4",
    },
    {
        "label": "sub04_indoor_180",
        "subject": "004",
        "sil_dir": REPO_ROOT / "datasets/CLoPGaitSilhouettes/004/nm-01-id/180",
        "skel_dir": REPO_ROOT / "datasets/CLoPGaitHamiltonSkeleton/004/nm-01-id/180",
        "video": REPO_ROOT / "datasets/CLoP-Gait/INDOOR/Sub04_ID/Nm1_04_ID/Nm1_04_ID_180/Nm1_04_ID_180.mp4",
    },
    {
        "label": "sub02_outdoor_day_180",
        "subject": "002",
        "sil_dir": REPO_ROOT / "datasets/CLoPGaitSilhouettes/002/nm-01-od/180",
        "skel_dir": REPO_ROOT / "datasets/CLoPGaitHamiltonSkeleton/002/nm-01-od/180",
        "video": REPO_ROOT / "datasets/CLoP-Gait/OUTDOOR/Outdoor_Day/Sub02_OD/Nm1_02_OD/Nm1_02_OD_180/Nm1_02_OD_180.mp4",
    },
    {
        "label": "sub02_indoor_000",
        "subject": "002",
        "sil_dir": REPO_ROOT / "datasets/CLoPGaitSilhouettes/002/nm-01-id/000",
        "skel_dir": REPO_ROOT / "datasets/CLoPGaitHamiltonSkeleton/002/nm-01-id/000",
        "video": REPO_ROOT / "datasets/CLoP-Gait/INDOOR/Sub02_ID/Nm1_02_ID/Nm1_02_ID_0/Nm1_02_ID_0.mp4",
    },
    {
        "label": "sub01_outdoor_night_090",
        "subject": "001",
        "sil_dir": REPO_ROOT / "datasets/CLoPGaitSilhouettes/001/nm-01-on/090",
        "skel_dir": REPO_ROOT / "datasets/CLoPGaitHamiltonSkeleton/001/nm-01-on/090",
        "video": REPO_ROOT / "datasets/CLoP-Gait/OUTDOOR/Outdoor_Night/Sub01_ON/Nm1_01_ON/Nm1_01_ON_90/Nm1_01_ON_90.mp4",
    },
]

SAMPLE_FPS = 10.0
MAX_IMAGES = 10


def middle_window(total_kept: int, count: int) -> list[int]:
    count = min(count, total_kept)
    center = round(total_kept / 2)
    start = max(1, center - count // 2)
    end = min(total_kept, start + count - 1)
    start = max(1, end - count + 1)
    return list(range(start, end + 1))


def find_kept_source_indices(video_path: Path, wanted_kept_indices: set[int]) -> dict[int, int]:
    """Replays segment_video's exact detection/tracking loop and records,
    for each wanted kept-frame index, the raw source video frame index."""
    model = load_model()
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    source_fps = capture.get(cv2.CAP_PROP_FPS) or 0.0
    stride = max(1, round(source_fps / SAMPLE_FPS)) if source_fps > 0 else 1

    source_frame_index = 0
    frames_kept = 0
    previous_centroid = None
    found: dict[int, int] = {}
    last_wanted = max(wanted_kept_indices)

    while True:
        if source_frame_index % stride != 0:
            if not capture.grab():
                break
            source_frame_index += 1
            continue

        success, frame = capture.read()
        if not success:
            break
        this_source_index = source_frame_index
        source_frame_index += 1

        results = model(frame, classes=[0], verbose=False, imgsz=DEFAULT_IMGSZ)
        masks = _frame_masks(results[0], frame.shape[:2])
        chosen, previous_centroid = select_tracked_mask(masks, previous_centroid)
        if chosen is None:
            continue

        frames_kept += 1
        if frames_kept in wanted_kept_indices:
            found[frames_kept] = this_source_index
        if frames_kept >= last_wanted:
            break

    capture.release()
    return found


def main() -> None:
    if OUT_ROOT.exists():
        shutil.rmtree(OUT_ROOT)
    OUT_ROOT.mkdir(parents=True)

    for target in TARGETS:
        sil_files = sorted(target["sil_dir"].glob("*.png"))
        total_kept = len(sil_files)
        wanted = middle_window(total_kept, MAX_IMAGES)
        print(f"=== {target['label']}: total_kept={total_kept}, window={wanted} ===")

        kept_to_source = find_kept_source_indices(target["video"], set(wanted))
        missing = [k for k in wanted if k not in kept_to_source]
        if missing:
            print(f"  WARNING: could not locate source frames for kept indices {missing}")

        out_dir = OUT_ROOT / target["label"]
        out_dir.mkdir(parents=True, exist_ok=True)

        capture = cv2.VideoCapture(str(target["video"]))
        for kept_idx in wanted:
            sil_path = target["sil_dir"] / f"{target['subject']}-{target['sil_dir'].parent.name}-{target['sil_dir'].name}-{kept_idx:06d}.png"
            if not sil_path.exists():
                # filename condition segment may include cl/nm variants; fall back to positional match
                sil_path = sil_files[kept_idx - 1]
            skel_name = sil_path.name.replace(".png", "_skeleton.png")
            skel_path = target["skel_dir"] / skel_name

            tag = f"frame_{kept_idx:03d}"
            if sil_path.exists():
                shutil.copy(sil_path, out_dir / f"{tag}_silhouette.png")
            if skel_path.exists():
                shutil.copy(skel_path, out_dir / f"{tag}_skeleton.png")

            source_idx = kept_to_source.get(kept_idx)
            if source_idx is not None:
                capture.set(cv2.CAP_PROP_POS_FRAMES, source_idx)
                ok, frame = capture.read()
                if ok:
                    cv2.imwrite(str(out_dir / f"{tag}_original.png"), frame, [cv2.IMWRITE_PNG_COMPRESSION, 3])
                else:
                    print(f"  WARNING: could not read source frame {source_idx} for kept {kept_idx}")
        capture.release()
        print(f"  wrote {len(list(out_dir.glob('*.png')))} files to {out_dir}")


if __name__ == "__main__":
    main()
