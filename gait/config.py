from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class ExperimentConfig:
    design_name: str = "hj_topogait_v1"
    run_name: str = "baseline_001"
    experiments_root: str = "/data/experiments"
    # Modal paths. The uploaded command places the archive at this exact path.
    dataset_path: str = "/data/silhouette-C.zip"
    silhouette_dataset_path: str = ""
    cache_dir: str = "/data/processed/casia_c_hj"
    dataset_name: str = "casia_c"
    dataset_format: str = "silhouette_hj"

    height: int = 64
    width: int = 44
    sequence_length: int = 30
    morphology_kernel: int = 3
    min_component_ratio: float = 0.015

    train_subjects: int = 100
    split_mode: str = "subject"
    test_domain_suffix: str = "od"
    identities_per_batch: int = 8
    sequences_per_identity: int = 4
    num_workers: int = 2
    epochs: int = 40
    generative_warmup_epochs: int = 4
    generative_step_interval: int = 2
    early_stopping_patience: int = 8
    early_stopping_min_delta: float = 0.002
    early_stopping_metric: str = "rank1"
    early_stopping_start_epoch: int = 0
    eval_gallery_per_subject: int = 1
    visual_every_n_epochs: int = 5
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    embedding_dim: int = 256
    projection_dim: int = 128
    hidden_dim: int = 128
    mask_ratio: float = 0.4
    temperature: float = 0.1
    triplet_margin: float = 0.2
    lambda_radius: float = 0.5
    lambda_contrastive: float = 1.0
    lambda_ce: float = 1.0
    lambda_triplet: float = 1.0
    label_smoothing: float = 0.0
    topology_positive_weight: float = 8.0
    lambda_dice: float = 1.0
    condition_aware_sampling: bool = False
    scheduler_name: str = "none"
    scheduler_min_lr: float = 1e-5
    seed: int = 42
    amp: bool = True

    @property
    def batch_size(self) -> int:
        return self.identities_per_batch * self.sequences_per_identity

    @property
    def output_dir(self) -> str:
        return str(Path(self.experiments_root) / self.design_name / self.run_name)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "ExperimentConfig":
        allowed = {field.name for field in fields(cls)}
        unknown = set(values) - allowed
        if unknown:
            raise ValueError(f"Unknown configuration keys: {sorted(unknown)}")
        return cls(**values)

    def ensure_output_dirs(self) -> None:
        for label, value in (("design_name", self.design_name), ("run_name", self.run_name)):
            if not value or value in {".", ".."} or "/" in value or "\\" in value:
                raise ValueError(f"{label} must be a simple directory name, got {value!r}")
        Path(self.cache_dir).mkdir(parents=True, exist_ok=True)
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
