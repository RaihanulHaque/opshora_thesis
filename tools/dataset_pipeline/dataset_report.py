"""Thesis-facing dataset-composition report for CLoP-Gait.

Read-only, like qc.py, but where qc.py is process-QC-focused (did
segmentation/skeleton generation succeed?), this is descriptive
dataset-statistics for the thesis: subject/domain/condition/view coverage,
frame-count distributions, and the domain-generalization train/test split
sizes (indoor+outdoor-night = train, outdoor-day = test, per
gait.dataset.GaitSequenceDataset's split_mode="domain").
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.dataset_pipeline.manifest import RawVideoRecord, discover_raw_videos  # noqa: E402
from tools.dataset_pipeline.pipeline import DEFAULT_SILHOUETTE_OUT, DEFAULT_SKELETON_OUT, _leaf_dir  # noqa: E402

DOMAIN_NAMES = {"id": "indoor", "od": "outdoor-day", "on": "outdoor-night"}
TEST_DOMAIN_SUFFIX = "od"


def _read_marker(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text())


def _domain_of(record: RawVideoRecord) -> str:
    return record.condition.rsplit("-", 1)[-1]


def _distribution(values: list[int]) -> dict[str, float]:
    if not values:
        return {"count": 0, "mean": 0.0, "median": 0.0, "min": 0, "max": 0, "stdev": 0.0}
    return {
        "count": len(values),
        "mean": round(statistics.mean(values), 1),
        "median": round(statistics.median(values), 1),
        "min": min(values),
        "max": max(values),
        "stdev": round(statistics.pstdev(values), 1) if len(values) > 1 else 0.0,
    }


def build_report(
    raw_root: Path,
    silhouette_root: Path,
    skeleton_root: Path,
    test_domain_suffix: str = TEST_DOMAIN_SUFFIX,
) -> dict:
    records = discover_raw_videos(raw_root)
    ok = [record for record in records if record.status == "ok"]
    subjects = sorted({record.subject for record in records})

    # Coverage: subjects x domain x condition x view
    coverage_by_domain: dict[str, dict] = {}
    for domain_suffix, domain_name in DOMAIN_NAMES.items():
        domain_records = [record for record in records if _domain_of(record) == domain_suffix]
        if not domain_records:
            continue
        domain_ok = [record for record in domain_records if record.status == "ok"]
        coverage_by_domain[domain_name] = {
            "subjects": sorted({record.subject for record in domain_records}),
            "total_sequences": len(domain_records),
            "ok_sequences": len(domain_ok),
            "missing_sequences": len(domain_records) - len(domain_ok),
        }

    # Frame-count distributions, from already-written _COMPLETE.json markers.
    silhouette_frames: list[int] = []
    skeleton_frames: list[int] = []
    silhouette_frames_by_domain: dict[str, list[int]] = defaultdict(list)
    for record in ok:
        segment_marker = _read_marker(_leaf_dir(silhouette_root, record) / "_SEGMENT_COMPLETE.json")
        if segment_marker is not None:
            silhouette_frames.append(segment_marker["frames_kept"])
            silhouette_frames_by_domain[DOMAIN_NAMES[_domain_of(record)]].append(segment_marker["frames_kept"])
        skeleton_marker = _read_marker(_leaf_dir(skeleton_root, record) / "_SKELETON_COMPLETE.json")
        if skeleton_marker is not None:
            skeleton_frames.append(skeleton_marker["frames_processed"])

    # Domain-generalization split sizes (matches GaitSequenceDataset split_mode="domain").
    per_subject_split: dict[str, dict[str, int]] = {}
    for subject in subjects:
        subject_ok = [record for record in ok if record.subject == subject]
        train_count = sum(1 for record in subject_ok if _domain_of(record) != test_domain_suffix)
        test_count = sum(1 for record in subject_ok if _domain_of(record) == test_domain_suffix)
        per_subject_split[subject] = {"train": train_count, "test": test_count}
    total_train = sum(item["train"] for item in per_subject_split.values())
    total_test = sum(item["test"] for item in per_subject_split.values())

    return {
        "total_subjects": len(subjects),
        "total_raw_sequences": len(records),
        "total_ok_sequences": len(ok),
        "coverage_by_domain": coverage_by_domain,
        "silhouette_frame_distribution": _distribution(silhouette_frames),
        "silhouette_frame_distribution_by_domain": {
            domain: _distribution(values) for domain, values in silhouette_frames_by_domain.items()
        },
        "skeleton_frame_distribution": _distribution(skeleton_frames),
        "domain_generalization_split": {
            "test_domain_suffix": test_domain_suffix,
            "per_subject": per_subject_split,
            "total_train_sequences": total_train,
            "total_test_sequences": total_test,
            "subjects_with_zero_test_sequences": sorted(
                subject for subject, item in per_subject_split.items() if item["test"] == 0
            ),
        },
    }


def write_markdown_report(report: dict, path: Path) -> None:
    lines = [
        "# CLoP-Gait Dataset Composition Report",
        "",
        f"- Subjects: `{report['total_subjects']}`",
        f"- Raw sequences discovered: `{report['total_raw_sequences']}` "
        f"(`{report['total_ok_sequences']}` usable)",
        "",
        "## Coverage by domain",
        "",
        "| Domain | Subjects | Sequences (ok/total) |",
        "|---|---:|---:|",
    ]
    for domain_name, item in report["coverage_by_domain"].items():
        lines.append(
            f"| {domain_name} | {len(item['subjects'])} | {item['ok_sequences']}/{item['total_sequences']} |"
        )

    lines += ["", "## Silhouette frame-count distribution", ""]
    dist = report["silhouette_frame_distribution"]
    lines.append(
        f"- Overall: n=`{dist['count']}`, mean=`{dist['mean']}`, median=`{dist['median']}`, "
        f"min=`{dist['min']}`, max=`{dist['max']}`, stdev=`{dist['stdev']}`"
    )
    for domain_name, domain_dist in report["silhouette_frame_distribution_by_domain"].items():
        lines.append(
            f"- {domain_name}: n=`{domain_dist['count']}`, mean=`{domain_dist['mean']}`, "
            f"min=`{domain_dist['min']}`, max=`{domain_dist['max']}`"
        )

    skel_dist = report["skeleton_frame_distribution"]
    lines += [
        "",
        "## Skeleton frame-count distribution",
        "",
        f"- Overall: n=`{skel_dist['count']}`, mean=`{skel_dist['mean']}`, median=`{skel_dist['median']}`, "
        f"min=`{skel_dist['min']}`, max=`{skel_dist['max']}`, stdev=`{skel_dist['stdev']}`",
    ]

    split = report["domain_generalization_split"]
    lines += [
        "",
        "## Domain-generalization train/test split",
        "",
        f"Train = every domain except `{split['test_domain_suffix']}` (indoor + outdoor-night); "
        f"test = only `{split['test_domain_suffix']}` (outdoor-day) -- same subjects in both, per "
        "`gait.dataset.GaitSequenceDataset(split_mode=\"domain\")`.",
        "",
        f"- Total train sequences: `{split['total_train_sequences']}`",
        f"- Total test sequences: `{split['total_test_sequences']}`",
        "",
        "| Subject | Train | Test |",
        "|---|---:|---:|",
    ]
    for subject, item in sorted(split["per_subject"].items()):
        lines.append(f"| {subject} | {item['train']} | {item['test']} |")
    if split["subjects_with_zero_test_sequences"]:
        lines += [
            "",
            f"**Note:** subject(s) `{', '.join(split['subjects_with_zero_test_sequences'])}` have zero "
            "outdoor-day footage, so they contribute no probe sequences to the test split (still used in "
            "training via indoor/outdoor-night).",
        ]
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the CLoP-Gait thesis dataset-composition report.")
    parser.add_argument("--raw-root", type=Path, default=Path("datasets/CLoP-Gait"))
    parser.add_argument("--silhouette-root", type=Path, default=DEFAULT_SILHOUETTE_OUT)
    parser.add_argument("--skeleton-root", type=Path, default=DEFAULT_SKELETON_OUT)
    parser.add_argument("--test-domain-suffix", default=TEST_DOMAIN_SUFFIX)
    parser.add_argument("--output-dir", type=Path, default=Path("tools/dataset_pipeline/output/dataset_report"))
    args = parser.parse_args()

    report = build_report(args.raw_root, args.silhouette_root, args.skeleton_root, args.test_domain_suffix)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "dataset_report.json").write_text(json.dumps(report, indent=2))
    write_markdown_report(report, args.output_dir / "dataset_report.md")
    print(f"Report written to {args.output_dir}")


if __name__ == "__main__":
    main()
