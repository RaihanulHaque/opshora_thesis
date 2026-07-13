"""Raw CLoP-Gait video discovery and subject/condition/view normalization.

Walks datasets/CLoP-Gait/ and maps its raw folder names onto the
subject/condition/view convention documented in DATASET_DEVELOPMENT_PIPELINE.md,
so downstream stages (segment.py, skeleton.py) never need to know about the
raw dataset's own naming quirks.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

RAW_ROOT = Path("datasets/CLoP-Gait")

# Raw condition folder prefix -> normalized condition code (without domain suffix).
CONDITION_MAP = {
    "Nm1": "nm-01",
    "Nm2": "nm-02",
    "Nm3": "nm-03",
    "Lc1": "cl-01",
    "Lc2": "cl-02",
}

# Raw view folder suffix -> normalized three-digit view.
VIEW_MAP = {
    "0": "000",
    "90": "090",
    "180": "180",
}

# Raw domain subfolder (relative to RAW_ROOT) -> condition-string domain suffix.
# Walking is generic, so a new "OUTDOOR/Outdoor_Night" folder is picked up
# automatically the moment it exists on disk -- no code change needed.
DOMAIN_SUFFIX = {
    "INDOOR": "id",
    "OUTDOOR/Outdoor_Day": "od",
    "OUTDOOR/Outdoor_Night": "on",
}

_SUBJECT_RE = re.compile(r"^Sub(\d+)_(?:ID|OD|ON)$")
_LEAF_RE = re.compile(r"^([A-Za-z]+\d+)_(\d+)_(?:ID|OD|ON)_(\d+)$")

Status = Literal["ok", "missing", "multiple"]


@dataclass(slots=True)
class RawVideoRecord:
    subject: str  # normalized 3-digit id, e.g. "001"
    condition: str  # normalized condition + domain suffix, e.g. "nm-01-id"
    view: str  # normalized 3-digit view, e.g. "090"
    video_path: str | None  # absolute path to the single mp4, or None
    raw_leaf: str  # raw leaf folder path, for traceability
    status: Status

    def to_dict(self) -> dict:
        return asdict(self)


def _normalize_subject(name: str) -> str | None:
    match = _SUBJECT_RE.match(name)
    if not match:
        return None
    return f"{int(match.group(1)):03d}"


def _normalize_leaf(name: str) -> tuple[str, str] | None:
    """Parse a view-level leaf folder name like 'Nm1_01_ID_090' -> (condition, view)."""
    match = _LEAF_RE.match(name)
    if not match:
        return None
    condition_prefix, _subject_num, raw_view = match.groups()
    condition = CONDITION_MAP.get(condition_prefix)
    view = VIEW_MAP.get(raw_view)
    if condition is None or view is None:
        return None
    return condition, view


def discover_raw_videos(raw_root: Path = RAW_ROOT) -> list[RawVideoRecord]:
    records: list[RawVideoRecord] = []
    for domain_rel, suffix in DOMAIN_SUFFIX.items():
        domain_root = raw_root / domain_rel
        if not domain_root.is_dir():
            continue  # e.g. Outdoor_Night not collected yet -- skip silently, it just isn't there.
        for subject_dir in sorted(domain_root.iterdir()):
            if not subject_dir.is_dir():
                continue
            subject = _normalize_subject(subject_dir.name)
            if subject is None:
                continue
            for condition_dir in sorted(subject_dir.iterdir()):
                if not condition_dir.is_dir():
                    continue
                for leaf_dir in sorted(condition_dir.iterdir()):
                    if not leaf_dir.is_dir():
                        continue
                    parsed = _normalize_leaf(leaf_dir.name)
                    if parsed is None:
                        continue
                    condition, view = parsed
                    condition = f"{condition}-{suffix}"
                    videos = sorted(leaf_dir.glob("*.mp4"))
                    if len(videos) == 0:
                        status: Status = "missing"
                        video_path = None
                    elif len(videos) == 1:
                        status = "ok"
                        video_path = str(videos[0])
                    else:
                        status = "multiple"
                        video_path = str(videos[0])
                    records.append(
                        RawVideoRecord(
                            subject=subject,
                            condition=condition,
                            view=view,
                            video_path=video_path,
                            raw_leaf=str(leaf_dir),
                            status=status,
                        )
                    )
    records.sort(key=lambda record: (record.subject, record.condition, record.view))
    return records


def write_manifest_json(records: list[RawVideoRecord], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([record.to_dict() for record in records], indent=2))


def load_manifest_json(path: Path) -> list[RawVideoRecord]:
    payload = json.loads(path.read_text())
    return [RawVideoRecord(**item) for item in payload]


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Discover and normalize raw CLoP-Gait videos.")
    parser.add_argument("--raw-root", type=Path, default=RAW_ROOT)
    parser.add_argument("--output", type=Path, default=Path("tools/dataset_pipeline/output/raw_manifest.json"))
    args = parser.parse_args()

    records = discover_raw_videos(args.raw_root)
    write_manifest_json(records, args.output)

    ok = sum(1 for record in records if record.status == "ok")
    missing = [record for record in records if record.status == "missing"]
    multiple = [record for record in records if record.status == "multiple"]
    print(f"Discovered {len(records)} sequences: {ok} ok, {len(missing)} missing, {len(multiple)} multiple.")
    for record in missing:
        print(f"  MISSING: {record.subject}/{record.condition}/{record.view} <- {record.raw_leaf}")
    for record in multiple:
        print(f"  MULTIPLE mp4s: {record.subject}/{record.condition}/{record.view} <- {record.raw_leaf}")
    print(f"Manifest written to {args.output}")


if __name__ == "__main__":
    main()
