"""Read-only validation/QC report over whatever the pipeline has produced so far.

Safe to run at any point, including mid-batch: it only reads the raw
manifest plus whatever silhouette/skeleton folders and _COMPLETE markers
already exist, and clearly separates known/expected gaps from newly
discovered, unexpected failures.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.dataset_pipeline.manifest import RawVideoRecord, discover_raw_videos  # noqa: E402
from tools.dataset_pipeline.pipeline import DEFAULT_SILHOUETTE_OUT, DEFAULT_SKELETON_OUT, _leaf_dir  # noqa: E402

MIN_FRAMES = 30


def _read_marker(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text())


def build_report(
    raw_root: Path,
    silhouette_root: Path,
    skeleton_root: Path,
    min_frames: int = MIN_FRAMES,
) -> dict:
    records = discover_raw_videos(raw_root)
    known_gaps = [record for record in records if record.status != "ok"]
    usable = [record for record in records if record.status == "ok"]

    not_yet_processed: list[dict] = []
    low_frame_count: list[dict] = []
    unexpected_failures: list[dict] = []
    missing_pairs: list[dict] = []
    processed: list[dict] = []
    multi_person_frames_total = 0

    for record in usable:
        key = f"{record.subject}/{record.condition}/{record.view}"
        silhouette_dir = _leaf_dir(silhouette_root, record)
        skeleton_dir = _leaf_dir(skeleton_root, record)
        segment_marker = _read_marker(silhouette_dir / "_SEGMENT_COMPLETE.json")
        skeleton_marker = _read_marker(skeleton_dir / "_SKELETON_COMPLETE.json")

        if segment_marker is None and skeleton_marker is None:
            not_yet_processed.append({"key": key})
            continue

        if segment_marker is not None:
            multi_person_frames_total += segment_marker.get("multi_person_frames", 0)
            frames_kept = segment_marker.get("frames_kept", 0)
            if frames_kept == 0:
                unexpected_failures.append({"key": key, "reason": "0 frames kept during segmentation"})
            elif frames_kept < min_frames:
                low_frame_count.append({"key": key, "frames_kept": frames_kept})

        if (segment_marker is None) != (skeleton_marker is None):
            missing_pairs.append(
                {
                    "key": key,
                    "has_silhouette": segment_marker is not None,
                    "has_skeleton": skeleton_marker is not None,
                }
            )

        if segment_marker is not None and skeleton_marker is not None:
            processed.append(
                {
                    "key": key,
                    "frames_kept": segment_marker.get("frames_kept"),
                    "frames_dropped": segment_marker.get("frames_dropped"),
                    "skeleton_frames": skeleton_marker.get("frames_processed"),
                    "empty_skeleton_frames": skeleton_marker.get("empty_skeleton_frames"),
                }
            )

    coverage = {
        "total_raw_sequences": len(records),
        "known_gaps": len(known_gaps),
        "usable_sequences": len(usable),
        "processed": len(processed),
        "not_yet_processed": len(not_yet_processed),
    }

    return {
        "coverage": coverage,
        "known_gaps": [
            {"key": f"{r.subject}/{r.condition}/{r.view}", "status": r.status, "raw_leaf": r.raw_leaf}
            for r in known_gaps
        ],
        "unexpected_failures": unexpected_failures,
        "low_frame_count": low_frame_count,
        "missing_pairs": missing_pairs,
        "not_yet_processed": not_yet_processed,
        "multi_person_frames_total": multi_person_frames_total,
        "processed_sequences": processed,
    }


def write_markdown_report(report: dict, path: Path) -> None:
    lines = [
        "# CLoP-Gait Dataset QC Report",
        "",
        "## Coverage",
        "",
    ]
    for key, value in report["coverage"].items():
        lines.append(f"- {key}: `{value}`")

    lines += ["", "## Known/Confirmed Gaps", "", "Pre-existing raw-collection gaps, not fabricated or backfilled.", ""]
    if report["known_gaps"]:
        for gap in report["known_gaps"]:
            lines.append(f"- `{gap['key']}` ({gap['status']}) <- `{gap['raw_leaf']}`")
    else:
        lines.append("- none")

    lines += ["", "## Unexpected Failures", "", "Sequences that were attempted but produced nothing usable.", ""]
    if report["unexpected_failures"]:
        for failure in report["unexpected_failures"]:
            lines.append(f"- `{failure['key']}`: {failure['reason']}")
    else:
        lines.append("- none")

    lines += ["", f"## Low Frame Count (< {MIN_FRAMES})", ""]
    if report["low_frame_count"]:
        for item in report["low_frame_count"]:
            lines.append(f"- `{item['key']}`: {item['frames_kept']} frames")
    else:
        lines.append("- none")

    lines += ["", "## Missing Pairs (silhouette/skeleton mismatch)", ""]
    if report["missing_pairs"]:
        for item in report["missing_pairs"]:
            lines.append(f"- `{item['key']}`: silhouette={item['has_silhouette']} skeleton={item['has_skeleton']}")
    else:
        lines.append("- none")

    lines += [
        "",
        "## Diagnostics",
        "",
        f"- Multi-person-detected frames across all processed sequences: "
        f"`{report['multi_person_frames_total']}` (the tracker picks the closest-centroid match; "
        "an unexpectedly high count may mean background pedestrians in outdoor footage).",
        f"- Not yet processed (no markers found): `{len(report['not_yet_processed'])}`",
        "",
    ]
    path.write_text("\n".join(lines) + "\n")


def sample_visual_grid(
    records: list[RawVideoRecord],
    silhouette_root: Path,
    skeleton_root: Path,
    output_path: Path,
    max_sequences: int = 8,
) -> None:
    panel_height = 176
    rows = []
    shown = 0
    for record in records:
        if shown >= max_sequences:
            break
        silhouette_dir = _leaf_dir(silhouette_root, record)
        skeleton_dir = _leaf_dir(skeleton_root, record)
        silhouette_frames = sorted(silhouette_dir.glob("*.png")) if silhouette_dir.exists() else []
        if not silhouette_frames:
            continue
        mid_silhouette = silhouette_frames[len(silhouette_frames) // 2]
        mid_skeleton = skeleton_dir / f"{mid_silhouette.stem}_skeleton.png"

        raw_frame = None
        if record.video_path:
            capture = cv2.VideoCapture(record.video_path)
            capture.set(cv2.CAP_PROP_POS_FRAMES, capture.get(cv2.CAP_PROP_FRAME_COUNT) // 2)
            success, frame = capture.read()
            capture.release()
            if success:
                raw_frame = frame

        panels = []
        for label, image in (
            ("raw", raw_frame),
            ("silhouette", cv2.imread(str(mid_silhouette))),
            ("skeleton", cv2.imread(str(mid_skeleton)) if mid_skeleton.exists() else None),
        ):
            if image is None:
                image = np.zeros((panel_height, panel_height, 3), dtype=np.uint8)
            else:
                aspect = image.shape[1] / image.shape[0]
                image = cv2.resize(image, (max(1, round(panel_height * aspect)), panel_height))
                if image.ndim == 2:
                    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            cv2.putText(image, label, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1, cv2.LINE_AA)
            panels.append(image)

        max_width = max(panel.shape[1] for panel in panels)
        padded = [
            cv2.copyMakeBorder(panel, 0, 0, 0, max_width - panel.shape[1], cv2.BORDER_CONSTANT, value=(0, 0, 0))
            for panel in panels
        ]
        row = np.concatenate(padded, axis=1)
        label_row = np.zeros((24, row.shape[1], 3), dtype=np.uint8)
        cv2.putText(
            label_row,
            f"{record.subject}/{record.condition}/{record.view}",
            (6, 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        rows.append(np.concatenate([label_row, row], axis=0))
        shown += 1

    if not rows:
        return
    max_width = max(row.shape[1] for row in rows)
    padded_rows = [
        cv2.copyMakeBorder(row, 0, 0, 0, max_width - row.shape[1], cv2.BORDER_CONSTANT, value=(0, 0, 0))
        for row in rows
    ]
    grid = np.concatenate(padded_rows, axis=0)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), grid)


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate the CLoP-Gait silhouette/skeleton datasets built so far.")
    parser.add_argument("--raw-root", type=Path, default=Path("datasets/CLoP-Gait"))
    parser.add_argument("--silhouette-root", type=Path, default=DEFAULT_SILHOUETTE_OUT)
    parser.add_argument("--skeleton-root", type=Path, default=DEFAULT_SKELETON_OUT)
    parser.add_argument("--output-dir", type=Path, default=Path("tools/dataset_pipeline/output/qc"))
    parser.add_argument("--min-frames", type=int, default=MIN_FRAMES)
    parser.add_argument("--max-grid-sequences", type=int, default=8)
    args = parser.parse_args()

    report = build_report(args.raw_root, args.silhouette_root, args.skeleton_root, args.min_frames)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "qc_report.json").write_text(json.dumps(report, indent=2))
    write_markdown_report(report, args.output_dir / "qc_report.md")

    records = discover_raw_videos(args.raw_root)
    usable = [record for record in records if record.status == "ok"]
    sample_visual_grid(
        usable,
        args.silhouette_root,
        args.skeleton_root,
        args.output_dir / "sample_grid.png",
        max_sequences=args.max_grid_sequences,
    )

    print(json.dumps(report["coverage"], indent=2))
    print(f"Report written to {args.output_dir}")


if __name__ == "__main__":
    main()
