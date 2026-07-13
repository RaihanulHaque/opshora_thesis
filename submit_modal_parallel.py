from __future__ import annotations

import argparse
import json
from pathlib import Path

import modal

from modal_app import APP_NAME
from submit_modal import load_design_config

DEFAULT_JOBS = [
    "skeleton_silhouette_lstm_v1:lstm_baseline_001",
    "skeleton_silhouette_tcn_v1:tcn_baseline_001",
    "skeleton_silhouette_transformer_v1:transformer_baseline_001",
]


def parse_jobs(raw_jobs: list[str]) -> list[tuple[str, str]]:
    jobs = []
    for item in raw_jobs:
        if ":" not in item:
            raise SystemExit(f"Expected design:run_name, got {item!r}")
        design, run_name = item.split(":", 1)
        jobs.append((design, run_name))
    return jobs


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Submit several Modal training runs at once. Each spawn() call gets "
            "its own container because modal_app.py sets single_use_containers=True "
            "on the `train` function, so this genuinely runs the jobs in parallel."
        )
    )
    parser.add_argument(
        "jobs",
        nargs="*",
        default=DEFAULT_JOBS,
        help=(
            "One or more 'design:run_name' pairs, space separated. "
            f"Defaults to the three baseline comparison designs: {DEFAULT_JOBS}"
        ),
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="Optional JSON with overrides applied on top of every job's own config.json",
    )
    args = parser.parse_args()

    jobs = parse_jobs(args.jobs)
    overrides = json.loads(args.config.read_text()) if args.config else {}
    function = modal.Function.from_name(APP_NAME, "train")

    print(
        "Note: all of these designs share skeleton_silhouette_fusion_v6's cache_dir. "
        "If that cache has not been built yet, submit one job first (or run "
        "`python submit_modal.py preprocess --design skeleton_silhouette_fusion_v6`) "
        "so the others don't race to build it concurrently.\n"
    )

    submitted = []
    for design, run_name in jobs:
        values = load_design_config(design)
        values.update(overrides)
        values.update({"design_name": design, "run_name": run_name})
        call = function.spawn(values)
        submitted.append((design, run_name, call.object_id))
        print(f"Submitted {design}/{run_name}: {call.object_id}")

    print(f"\n{len(submitted)} runs submitted in parallel, each in its own container.")
    print("Monitor them with: modal app logs hj-topogait-training")


if __name__ == "__main__":
    main()
