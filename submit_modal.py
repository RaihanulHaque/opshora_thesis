from __future__ import annotations

import argparse
import json
from pathlib import Path

import modal

from modal_app import APP_NAME


def load_design_config(design: str) -> dict:
    path = Path("designs") / design / "config.json"
    if not path.exists():
        available = sorted(item.parent.name for item in Path("designs").glob("*/config.json"))
        raise SystemExit(f"Unknown design {design!r}. Available designs: {available}")
    return json.loads(path.read_text())


def main() -> None:
    parser = argparse.ArgumentParser(description="Submit one reproducible gait experiment to Modal.")
    parser.add_argument("job", choices=("run", "preprocess"), help="Use 'run' for normal experiments")
    parser.add_argument("--design", default="hj_topogait_v1", help="Folder name under designs/")
    parser.add_argument("--run", default="baseline_001", help="Unique run name; never reuse it for a different trial")
    parser.add_argument("--config", type=Path, help="Optional JSON with final per-run overrides")
    parser.add_argument("--force", action="store_true", help="Rebuild the preprocessing cache")
    args = parser.parse_args()
    values = load_design_config(args.design)
    if args.config:
        values.update(json.loads(args.config.read_text()))
    values.update({"design_name": args.design, "run_name": args.run})
    function_name = "preprocess" if args.job == "preprocess" else "train"
    function = modal.Function.from_name(APP_NAME, function_name)
    call = function.spawn(values, args.force) if function_name == "preprocess" else function.spawn(values)
    print(f"Submitted {args.design}/{args.run}: {call.object_id}")
    if args.job == "run":
        print("Preprocessing is automatic: an existing shared cache is reused, otherwise it is created before training.")
    print("The remote call is detached from this laptop. Monitor it in the Modal dashboard or with `modal app logs hj-topogait-training`.")


if __name__ == "__main__":
    main()
