"""Orchestration CLI: raw CLoP-Gait videos -> silhouette PNGs -> skeleton PNGs.

Example (single-video sanity check):

    python -m tools.dataset_pipeline.pipeline \\
      --subjects 001 --conditions nm-01-id --views 000 --stage all

Resumable: each (subject, condition, view) sequence is skipped if its
_SEGMENT_COMPLETE.json / _SKELETON_COMPLETE.json marker already exists,
unless --force is passed. Re-running after adding new raw footage later only
processes the new sequences -- nothing already done is touched.

Output layout for both DEFAULT_SILHOUETTE_OUT and DEFAULT_SKELETON_OUT is
<root>/<subject>/<condition>/<view>/*.png -- the same plain-directory-of-PNGs
shape gait/preprocessing.py's iter_archive_sequences()/_sequences_from_directory()
already reads for CASIA_B_Hamilton_Skeleton and GaitDatasetB-silh. So to
retrain V6 on this dataset, point an ExperimentConfig at:
  dataset_format: "skeleton_silhouette_fusion"
  dataset_path: "datasets/CLoPGaitHamiltonSkeleton"
  silhouette_dataset_path: "datasets/CLoPGaitSilhouettes"
No conversion/zipping step needed -- both are read directly as directories.

Stored silhouettes/skeletons keep a natural margin around the subject
(--fill-ratio, default 0.45 of canvas height) rather than a tight fill-canvas
crop -- matching CASIA-B's own raw storage convention. The tight crop
(gait.preprocessing.clean_and_align) still runs at V6 training-cache-build
time exactly as it always has, so this only changes how the stored dataset
looks when you inspect it, not what V6 actually trains on.
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gait.config import ExperimentConfig  # noqa: E402
from tools.dataset_pipeline.manifest import (  # noqa: E402
    RAW_ROOT,
    RawVideoRecord,
    discover_raw_videos,
    write_manifest_json,
)
from tools.dataset_pipeline.segment import DEFAULT_IMGSZ, segment_video  # noqa: E402
from tools.dataset_pipeline.skeleton import generate_skeleton_for_sequence  # noqa: E402

DEFAULT_SILHOUETTE_OUT = Path("datasets/CLoPGaitSilhouettes")
DEFAULT_SKELETON_OUT = Path("datasets/CLoPGaitHamiltonSkeleton")
DEFAULT_SAMPLE_FPS = 10.0
# Storage canvas: portrait, generous margin (fill_ratio below) -- like CASIA-B's
# raw GaitDatasetB-silh storage convention, not the tight training-time crop.
DEFAULT_STORAGE_HEIGHT = 320
DEFAULT_STORAGE_WIDTH = 240
DEFAULT_FILL_RATIO = 0.45


def _parse_csv(value: str | None) -> set[str] | None:
    if not value:
        return None
    return {item.strip() for item in value.split(",") if item.strip()}


def _filter_records(
    records: list[RawVideoRecord],
    subjects: set[str] | None,
    conditions: set[str] | None,
    views: set[str] | None,
) -> list[RawVideoRecord]:
    selected = []
    for record in records:
        if subjects is not None and record.subject not in subjects:
            continue
        if conditions is not None and record.condition not in conditions:
            continue
        if views is not None and record.view not in views:
            continue
        selected.append(record)
    return selected


def _leaf_dir(root: Path, record: RawVideoRecord) -> Path:
    return root / record.subject / record.condition / record.view


def _group_by_subject_and_condition(
    records: list[RawVideoRecord],
) -> dict[str, dict[str, list[RawVideoRecord]]]:
    grouped: dict[str, dict[str, list[RawVideoRecord]]] = defaultdict(lambda: defaultdict(list))
    for record in records:
        grouped[record.subject][record.condition].append(record)
    return grouped


def run(args: argparse.Namespace) -> None:
    config = ExperimentConfig(height=args.height, width=args.width)

    records = discover_raw_videos(args.raw_root)
    write_manifest_json(records, Path("tools/dataset_pipeline/output/raw_manifest.json"))

    selected = _filter_records(
        records,
        _parse_csv(args.subjects),
        _parse_csv(args.conditions),
        _parse_csv(args.views),
    )
    usable = [record for record in selected if record.status == "ok"]
    skipped = [record for record in selected if record.status != "ok"]

    print(f"Selected {len(selected)} sequences ({len(usable)} usable, {len(skipped)} skipped: not ok).")
    for record in skipped:
        print(f"  SKIP ({record.status}): {record.subject}/{record.condition}/{record.view}")
    print(f"Sample rate: {args.sample_fps} fps (source is 30 fps)" if args.sample_fps else "Sample rate: every frame")

    if args.dry_run:
        for record in usable:
            print(f"  WOULD PROCESS: {record.subject}/{record.condition}/{record.view} <- {record.video_path}")
        return

    grouped = _group_by_subject_and_condition(usable)
    subject_bar = tqdm(sorted(grouped), desc="Subjects", unit="subj", position=0)
    for subject in subject_bar:
        subject_bar.set_postfix_str(subject)
        conditions = grouped[subject]
        condition_bar = tqdm(sorted(conditions), desc=f"  {subject} conditions", unit="cond", position=1, leave=False)
        for condition in condition_bar:
            condition_bar.set_postfix_str(condition)
            for record in sorted(conditions[condition], key=lambda item: item.view):
                silhouette_dir = _leaf_dir(args.silhouette_out, record)
                skeleton_dir = _leaf_dir(args.skeleton_out, record)
                key = f"{record.subject}/{record.condition}/{record.view}"

                if args.stage in ("segment", "all"):
                    marker = silhouette_dir / "_SEGMENT_COMPLETE.json"
                    if marker.exists() and not args.force:
                        tqdm.write(f"segment: skip (already done) {key}")
                    else:
                        result = segment_video(
                            Path(record.video_path),
                            silhouette_dir,
                            config,
                            record.subject,
                            record.condition,
                            record.view,
                            weights_path=args.weights,
                            sample_fps=args.sample_fps,
                            imgsz=args.imgsz,
                            fill_ratio=args.fill_ratio,
                        )
                        tqdm.write(
                            f"segment: {key} -- frames_read={result.frames_read} frames_kept={result.frames_kept} "
                            f"frames_dropped={result.frames_dropped} multi_person_frames={result.multi_person_frames} "
                            f"source_fps={result.source_fps:.1f} sample_fps={result.sample_fps} stride={result.stride} "
                            f"elapsed={result.elapsed_seconds:.1f}s"
                        )

                if args.stage in ("skeleton", "all"):
                    marker = skeleton_dir / "_SKELETON_COMPLETE.json"
                    if marker.exists() and not args.force:
                        tqdm.write(f"skeleton: skip (already done) {key}")
                        continue
                    if not silhouette_dir.exists() or not any(silhouette_dir.glob("*.png")):
                        tqdm.write(f"skeleton: SKIP, no silhouettes found at {silhouette_dir}")
                        continue
                    result = generate_skeleton_for_sequence(silhouette_dir, skeleton_dir, progress_desc=key)
                    tqdm.write(
                        f"skeleton: {key} -- frames_processed={result.frames_processed} "
                        f"empty_skeleton_frames={result.empty_skeleton_frames} elapsed={result.elapsed_seconds:.1f}s"
                    )
        condition_bar.close()
    subject_bar.close()

    print("Done.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build CLoP-Gait silhouette + Hamilton skeleton datasets.")
    parser.add_argument("--raw-root", type=Path, default=RAW_ROOT)
    parser.add_argument("--silhouette-out", type=Path, default=DEFAULT_SILHOUETTE_OUT)
    parser.add_argument("--skeleton-out", type=Path, default=DEFAULT_SKELETON_OUT)
    parser.add_argument("--subjects", default=None, help="Comma-separated 3-digit subject ids, e.g. 001,002")
    parser.add_argument("--conditions", default=None, help="Comma-separated conditions, e.g. nm-01-id,cl-01-id")
    parser.add_argument("--views", default=None, help="Comma-separated views, e.g. 000,090")
    parser.add_argument("--stage", choices=["segment", "skeleton", "all"], default="all")
    parser.add_argument("--force", action="store_true", help="Reprocess even if already marked complete")
    parser.add_argument("--dry-run", action="store_true", help="Only print what would be processed")
    parser.add_argument(
        "--height",
        type=int,
        default=DEFAULT_STORAGE_HEIGHT,
        help=f"Stored silhouette/skeleton canvas height (default {DEFAULT_STORAGE_HEIGHT}, portrait, with natural "
        "margin per --fill-ratio -- not the tight training-time crop, which V6 still applies at cache-build time).",
    )
    parser.add_argument("--width", type=int, default=DEFAULT_STORAGE_WIDTH)
    parser.add_argument(
        "--fill-ratio",
        type=float,
        default=DEFAULT_FILL_RATIO,
        help=f"Fraction of canvas height the subject occupies in stored silhouettes (default {DEFAULT_FILL_RATIO}, "
        "matching CASIA-B's raw storage convention). Lower = more margin.",
    )
    parser.add_argument("--weights", default="yolo26m-seg.pt")
    parser.add_argument(
        "--imgsz",
        type=int,
        default=DEFAULT_IMGSZ,
        help=f"YOLO inference resolution (default {DEFAULT_IMGSZ}). Higher gives finer mask detail on 1080p "
        "source (less blocky limb edges) at a real per-frame speed cost -- see segment.py's segment_video docstring.",
    )
    parser.add_argument(
        "--sample-fps",
        type=float,
        default=DEFAULT_SAMPLE_FPS,
        help="Fixed frames-per-second to decode/segment from the 30 fps source video (e.g. 6-10), skipping the "
        "rest via a cheap grab() instead of a full decode -- also a real speedup, not just fewer output files. "
        f"Default: {DEFAULT_SAMPLE_FPS}. Pass 0 to process every frame instead.",
    )
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
