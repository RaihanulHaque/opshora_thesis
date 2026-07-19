from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Iterator

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler


def _condition_prefix(condition: str) -> str:
    return condition.split("-")[0].lower()


def _domain_suffix(condition: str) -> str | None:
    parts = condition.split("-")
    # CASIA-B conditions are "nm-01" (2 parts, no domain). CLoP-Gait conditions are
    # "nm-01-id" (3 parts: family, index, domain suffix).
    return parts[-1].lower() if len(parts) >= 3 else None


def _condition_family(condition: str) -> str:
    condition = condition.lower()
    if condition.startswith("fn") or condition.startswith("nm"):
        return "normal"
    if condition.startswith("fb") or condition.startswith("bg"):
        return "bag"
    if condition.startswith("cl"):
        return "clothing"
    if condition.startswith("fq"):
        return "fast"
    if condition.startswith("fs"):
        return "slow"
    return condition


class GaitSequenceDataset(Dataset):
    def __init__(
        self,
        cache_dir: str,
        split: str,
        train_subjects: int = 100,
        split_mode: str = "subject",
        test_domain_suffix: str = "od",
        train_condition_prefixes: list[str] | None = None,
        test_condition_prefixes: list[str] | None = None,
        train_domain_suffixes: list[str] | None = None,
        test_domain_suffixes: list[str] | None = None,
        validation_subjects: int = 0,
    ):
        self.cache_dir = Path(cache_dir)
        manifest_path = self.cache_dir / "manifest.jsonl"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Prepared manifest missing: {manifest_path}")
        records = [json.loads(line) for line in manifest_path.read_text().splitlines() if line]

        if split_mode == "domain":
            # Domain-generalization split: every subject appears in both splits.
            # Train = all sequences except the held-out test domain (e.g. indoor +
            # outdoor-night); test = only sequences from the held-out domain (e.g.
            # outdoor-day). This tests recognizing known identities in an unseen
            # environment, not generalizing to unseen identities.
            def is_test_domain(record: dict) -> bool:
                return record["condition"].rsplit("-", 1)[-1] == test_domain_suffix

            if split == "train":
                self.records = [record for record in records if not is_test_domain(record)]
            else:
                self.records = [record for record in records if is_test_domain(record)]
            if not self.records:
                raise ValueError(
                    f"Split {split!r} (domain mode, test_domain_suffix={test_domain_suffix!r}) has no records"
                )
            selected = {record["subject"] for record in self.records}
        elif split_mode == "condition":
            # Cross-condition generalization split: train on one condition family
            # (e.g. normal walk, optionally restricted to specific domains), test on a
            # disjoint condition family (e.g. clothing-change) — same subjects, same
            # views, in both splits. This tests robustness to an unseen appearance
            # variation, not unseen identities.
            def matches(record: dict, prefixes: list[str] | None, domains: list[str] | None) -> bool:
                prefix_ok = not prefixes or _condition_prefix(record["condition"]) in prefixes
                if not domains:
                    domain_ok = True
                else:
                    domain = _domain_suffix(record["condition"])
                    domain_ok = domain is not None and domain in domains
                return prefix_ok and domain_ok

            if split == "train":
                self.records = [
                    record for record in records if matches(record, train_condition_prefixes, train_domain_suffixes)
                ]
            else:
                self.records = [
                    record for record in records if matches(record, test_condition_prefixes, test_domain_suffixes)
                ]
            if not self.records:
                raise ValueError(
                    f"Split {split!r} (condition mode) has no records; "
                    f"train_condition_prefixes={train_condition_prefixes}, test_condition_prefixes={test_condition_prefixes}, "
                    f"train_domain_suffixes={train_domain_suffixes}, test_domain_suffixes={test_domain_suffixes}"
                )
            selected = {record["subject"] for record in self.records}
        else:
            subjects = sorted({record["subject"] for record in records})
            boundary = min(train_subjects, len(subjects))
            if validation_subjects > 0:
                # Subject-disjoint 3-way split: train | validation | test, so
                # early stopping can monitor validation instead of the test
                # split (see ExperimentConfig.validation_subjects).
                if validation_subjects >= boundary:
                    raise ValueError(
                        f"validation_subjects={validation_subjects} must be smaller than train_subjects={boundary}"
                    )
                val_boundary = boundary - validation_subjects
                if split == "train":
                    selected = set(subjects[:val_boundary])
                elif split == "validation":
                    selected = set(subjects[val_boundary:boundary])
                else:
                    selected = set(subjects[boundary:])
            else:
                if split == "validation":
                    raise ValueError(
                        "split='validation' requested but config.validation_subjects is 0; "
                        "set validation_subjects > 0 to carve out a validation split."
                    )
                selected = set(subjects[:boundary] if split == "train" else subjects[boundary:])
            self.records = [record for record in records if record["subject"] in selected]
            if not self.records:
                raise ValueError(f"Split {split!r} has no records; subjects={len(subjects)}, boundary={boundary}")

        self.subject_to_label = {subject: index for index, subject in enumerate(sorted(selected))}

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        record = self.records[index]
        with np.load(self.cache_dir / record["path"]) as payload:
            maps = payload["maps"].astype(np.float32) / 255.0
        return {
            "silhouette": torch.from_numpy(maps[:, :1]),
            "topology": torch.from_numpy(maps[:, 1:]),
            "label": torch.tensor(self.subject_to_label[record["subject"]], dtype=torch.long),
            "subject": record["subject"],
            "condition": record["condition"],
            "condition_family": _condition_family(record["condition"]),
            "view": record["view"],
        }


class PKBatchSampler(Sampler[list[int]]):
    """Sample P identities and K sequences per identity for SupCon/triplet loss."""

    def __init__(
        self,
        dataset: GaitSequenceDataset,
        p: int,
        k: int,
        seed: int = 42,
        condition_aware: bool = False,
    ):
        self.p, self.k, self.seed = p, k, seed
        self.condition_aware = condition_aware
        self.epoch = 0
        self.by_subject: dict[str, list[int]] = defaultdict(list)
        self.by_subject_condition: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
        for index, record in enumerate(dataset.records):
            self.by_subject[record["subject"]].append(index)
            self.by_subject_condition[record["subject"]][_condition_family(record["condition"])].append(index)
        if len(self.by_subject) < p:
            raise ValueError(f"Need at least {p} identities, found {len(self.by_subject)}")

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        total = sum(len(indices) for indices in self.by_subject.values())
        return max(1, total // (self.p * self.k))

    def __iter__(self) -> Iterator[list[int]]:
        rng = random.Random(self.seed + self.epoch)
        subjects = list(self.by_subject)
        for _ in range(len(self)):
            chosen_subjects = rng.sample(subjects, self.p)
            batch: list[int] = []
            for subject in chosen_subjects:
                candidates = self.by_subject[subject]
                if self.condition_aware:
                    groups = list(self.by_subject_condition[subject].values())
                    rng.shuffle(groups)
                    selected = [rng.choice(group) for group in groups[: self.k]]
                    remaining = [index for index in candidates if index not in selected]
                    needed = self.k - len(selected)
                    if needed > 0:
                        selected.extend(rng.sample(remaining, needed) if len(remaining) >= needed else rng.choices(candidates, k=needed))
                    batch.extend(selected)
                elif len(candidates) >= self.k:
                    batch.extend(rng.sample(candidates, self.k))
                else:
                    batch.extend(rng.choices(candidates, k=self.k))
            yield batch
