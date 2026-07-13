"""Orchestration CLI: raw CLoP-Gait videos -> silhouette PNGs -> skeleton PNGs.

Example (single-video sanity check):

    python -m tools.dataset_pipeline.pipeline \\
      --subjects 001 --conditions nm-01-id --views 000 --stage all

Resumable: each (subject, condition, view) sequence is skipped if its
_SEGMENT_COMPLETE.json / _SKELETON_COMPLETE.json marker already exists,
unless --force is passed. Re-running after adding Outdoor_Night later only
processes the new sequences -- nothing already done is touched.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

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
from tools.dataset_pipeline.segment import segment_video  # noqa: E402
from tools.dataset_pipeline.skeleton import generate_skeleton_for_sequence  # noqa: E402

DEFAULT_SILHOUETTE_OUT = Path("datasets/CLoPGaitSilhouettes")
DEFAULT_SKELETON_OUT = Path("datasets/CLoPGaitHamiltonSkeleton")


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

    if args.dry_run:
        for record in usable:
            print(f"  WOULD PROCESS: {record.subject}/{record.condition}/{record.view} <- {record.video_path}")
        return

    for index, record in enumerate(usable, start=1):
        silhouette_dir = _leaf_dir(args.silhouette_out, record)
        skeleton_dir = _leaf_dir(args.skeleton_out, record)

        if args.stage in ("segment", "all"):
            marker = silhouette_dir / "_SEGMENT_COMPLETE.json"
            if marker.exists() and not args.force:
                print(f"[{index}/{len(usable)}] segment: skip (already done) {record.subject}/{record.condition}/{record.view}")
            else:
                print(f"[{index}/{len(usable)}] segment: {record.subject}/{record.condition}/{record.view}")
                result = segment_video(
                    Path(record.video_path),
                    silhouette_dir,
                    config,
                    record.subject,
                    record.condition,
                    record.view,
                    weights_path=args.weights,
                    sample_fps=args.sample_fps,
                )
                print(
                    f"    frames_read={result.frames_read} frames_kept={result.frames_kept} "
                    f"frames_dropped={result.frames_dropped} multi_person_frames={result.multi_person_frames} "
                    f"source_fps={result.source_fps:.1f} sample_fps={result.sample_fps} stride={result.stride} "
                    f"elapsed={result.elapsed_seconds:.1f}s"
                )

        if args.stage in ("skeleton", "all"):
            marker = skeleton_dir / "_SKELETON_COMPLETE.json"
            if marker.exists() and not args.force:
                print(f"[{index}/{len(usable)}] skeleton: skip (already done) {record.subject}/{record.condition}/{record.view}")
                continue
            if not silhouette_dir.exists() or not any(silhouette_dir.glob("*.png")):
                print(f"[{index}/{len(usable)}] skeleton: SKIP, no silhouettes found at {silhouette_dir}")
                continue
            print(f"[{index}/{len(usable)}] skeleton: {record.subject}/{record.condition}/{record.view}")
            result = generate_skeleton_for_sequence(silhouette_dir, skeleton_dir)
            print(
                f"    frames_processed={result.frames_processed} "
                f"empty_skeleton_frames={result.empty_skeleton_frames} elapsed={result.elapsed_seconds:.1f}s"
            )

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
    parser.add_argument("--height", type=int, default=128)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--weights", default="yolo26m-seg.pt")
    parser.add_argument(
        "--sample-fps",
        type=float,
        default=None,
        help="Only decode/segment ~N frames per second of source video (e.g. 5), skipping the rest via a cheap "
        "grab() instead of a full decode. Default: process every frame.",
    )
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
