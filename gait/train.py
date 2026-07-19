from __future__ import annotations

import json
import importlib
import os
import random
import shutil
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import cv2
from torch.nn import functional as F
from torch.utils.data import DataLoader

from .config import ExperimentConfig
from .dataset import GaitSequenceDataset, PKBatchSampler
from .losses import batch_hard_triplet_loss, masked_reconstruction_loss, supervised_contrastive_loss
from .preprocessing import prepare_dataset


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def move_batch(batch: dict, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        batch["silhouette"].to(device, non_blocking=True),
        batch["topology"].to(device, non_blocking=True),
        batch["label"].to(device, non_blocking=True),
    )


def save_input_preview(batch: dict, path: Path) -> None:
    """Save silhouette, topology, radius and flux for one representative frame."""
    frame = min(10, batch["silhouette"].shape[1] - 1)
    maps = [batch["silhouette"][0, frame, 0], *batch["topology"][0, frame]]
    labels = ["silhouette", "topology", "radius", "flux"]
    panels: list[np.ndarray] = []
    for label, tensor in zip(labels, maps):
        image = np.round(tensor.numpy() * 255).astype(np.uint8)
        image = cv2.resize(image, (176, 256), interpolation=cv2.INTER_NEAREST)
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        cv2.putText(image, label, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1, cv2.LINE_AA)
        panels.append(image)
    cv2.imwrite(str(path), np.concatenate(panels, axis=1))


def save_reconstruction_preview(
    silhouette: torch.Tensor,
    topology: torch.Tensor,
    prediction: torch.Tensor,
    mask: torch.Tensor,
    path: Path,
    labels: list[str] | None = None,
) -> None:
    masked_frames = torch.nonzero(mask[0], as_tuple=False).flatten().tolist()
    frame = masked_frames[0] if masked_frames else 0
    second_target = topology[0, frame, 1] if topology.shape[2] > 1 else topology[0, frame, 0]
    second_prediction = prediction[0, frame, 1] if prediction.shape[2] > 1 else prediction[0, frame, 0]
    tensors = [
        silhouette[0, frame, 0],
        topology[0, frame, 0],
        torch.sigmoid(prediction[0, frame, 0]),
        second_target,
        torch.sigmoid(second_prediction),
    ]
    if labels is None:
        labels = ["input map", "target channel 1", "reconstructed channel 1", "target channel 2", "reconstructed channel 2"]
    panels: list[np.ndarray] = []
    for label, tensor in zip(labels, tensors):
        image = np.round(tensor.detach().float().cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        image = cv2.resize(image, (176, 256), interpolation=cv2.INTER_NEAREST)
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        cv2.putText(image, label, (5, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 0), 1, cv2.LINE_AA)
        panels.append(image)
    cv2.imwrite(str(path), np.concatenate(panels, axis=1))


def save_training_curves(metrics_path: Path, path: Path) -> None:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = [json.loads(line) for line in metrics_path.read_text().splitlines() if line]
    if not rows:
        return
    epochs = [row["epoch"] + 1 for row in rows]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(
        epochs,
        [row.get("generative_avg", row["generative"] / max(row["steps"], 1)) for row in rows],
        label="generative",
    )
    axes[0].plot(
        epochs,
        [row.get("recognition_avg", row["recognition"] / max(row["steps"], 1)) for row in rows],
        label="recognition",
    )
    axes[0].set(title="Training losses", xlabel="Epoch", ylabel="Average loss")
    axes[0].legend()
    axes[1].plot(epochs, [row["rank1"] for row in rows], label="Rank-1")
    axes[1].plot(epochs, [row["rank5"] for row in rows], label="Rank-5")
    axes[1].set(title="Unseen-subject retrieval", xlabel="Epoch", ylabel="Accuracy", ylim=(0, 1))
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    gallery_per_subject: int = 1,
) -> dict[str, float]:
    model.eval()
    embeddings: list[torch.Tensor] = []
    subjects: list[str] = []
    conditions: list[str] = []
    for batch in loader:
        output = model(batch["silhouette"].to(device), batch["topology"].to(device))
        embeddings.append(F.normalize(output["embedding"], dim=1).cpu())
        subjects.extend(batch["subject"])
        conditions.extend(batch["condition_family"])
    matrix = torch.cat(embeddings)
    gallery_indices: list[int] = []
    gallery_index_groups: list[list[int]] = []
    probe_indices: list[int] = []
    for subject in sorted(set(subjects)):
        candidates = [index for index, value in enumerate(subjects) if value == subject]
        normal = [index for index in candidates if conditions[index] == "normal"]
        preferred = normal if normal else candidates
        gallery_count = max(1, min(gallery_per_subject, len(preferred)))
        galleries = preferred[:gallery_count]
        gallery_index_groups.append(galleries)
        gallery_indices.extend(galleries)
        gallery_set = set(galleries)
        probe_indices.extend(index for index in candidates if index not in gallery_set)
    if not probe_indices:
        return {"rank1": 0.0, "rank5": 0.0}
    flat_similarities = matrix[probe_indices] @ matrix[gallery_indices].T
    subject_similarities: list[torch.Tensor] = []
    offset = 0
    gallery_subject_names: list[str] = []
    for group in gallery_index_groups:
        group_size = len(group)
        subject_similarities.append(flat_similarities[:, offset : offset + group_size].max(dim=1).values)
        gallery_subject_names.append(subjects[group[0]])
        offset += group_size
    similarities = torch.stack(subject_similarities, dim=1)
    order = similarities.argsort(dim=1, descending=True)
    gallery_subjects = np.array(gallery_subject_names)
    probe_subjects = np.array([subjects[index] for index in probe_indices])
    ranked = gallery_subjects[order.numpy()]
    rank1 = float(np.mean(ranked[:, 0] == probe_subjects))
    rank5 = float(np.mean([subject in row[: min(5, len(row))] for subject, row in zip(probe_subjects, ranked)]))
    distance_matrix = (1.0 - (matrix @ matrix.T)).numpy()
    same_distances: list[float] = []
    different_distances: list[float] = []
    for left in range(len(subjects)):
        for right in range(left + 1, len(subjects)):
            if subjects[left] == subjects[right]:
                same_distances.append(float(distance_matrix[left, right]))
            else:
                different_distances.append(float(distance_matrix[left, right]))
    same = np.asarray(same_distances, dtype=np.float32)
    different = np.asarray(different_distances, dtype=np.float32)
    if len(same) and len(different):
        rng = np.random.default_rng(123)
        different_auc = different
        if len(different_auc) > 50000:
            different_auc = rng.choice(different_auc, size=50000, replace=False)
        same_auc = same
        if len(same_auc) > 50000:
            same_auc = rng.choice(same_auc, size=50000, replace=False)
        different_sorted = np.sort(different_auc)
        not_greater = np.searchsorted(different_sorted, same_auc, side="right")
        verification_auc = float(np.mean(1.0 - not_greater / max(len(different_sorted), 1)))
        threshold = float((same.mean() + different.mean()) / 2.0)
        same_correct = np.mean(same <= threshold)
        different_correct = np.mean(different > threshold)
        verification_accuracy = float((same_correct + different_correct) / 2.0)
        same_mean = float(same.mean())
        different_mean = float(different.mean())
    else:
        verification_auc = verification_accuracy = same_mean = different_mean = 0.0
    return {
        "rank1": rank1,
        "rank5": rank5,
        "same_distance": same_mean,
        "different_distance": different_mean,
        "distance_gap": different_mean - same_mean,
        "verification_auc": verification_auc,
        "verification_accuracy": verification_accuracy,
        "gallery_per_subject": float(gallery_per_subject),
    }


def train_experiment(
    config: ExperimentConfig,
    commit_callback: Callable[[], None] | None = None,
) -> dict[str, float | int | str]:
    seed_everything(config.seed)
    config.ensure_output_dirs()
    summary = prepare_dataset(config)
    if commit_callback:
        commit_callback()
    print(f"Prepared dataset: {summary}", flush=True)

    train_data = GaitSequenceDataset(
        config.cache_dir,
        "train",
        config.train_subjects,
        config.split_mode,
        config.test_domain_suffix,
        config.train_condition_prefixes,
        config.test_condition_prefixes,
        config.train_domain_suffixes,
        config.test_domain_suffixes,
        config.validation_subjects,
    )
    test_data = GaitSequenceDataset(
        config.cache_dir,
        "test",
        config.train_subjects,
        config.split_mode,
        config.test_domain_suffix,
        config.train_condition_prefixes,
        config.test_condition_prefixes,
        config.train_domain_suffixes,
        config.test_domain_suffixes,
        config.validation_subjects,
    )
    validation_data = None
    if config.validation_subjects > 0:
        if config.split_mode != "subject":
            raise ValueError(
                f"config.validation_subjects > 0 is only supported for split_mode='subject', got {config.split_mode!r}"
            )
        validation_data = GaitSequenceDataset(
            config.cache_dir,
            "validation",
            config.train_subjects,
            config.split_mode,
            config.test_domain_suffix,
            config.train_condition_prefixes,
            config.test_condition_prefixes,
            config.train_domain_suffixes,
            config.test_domain_suffixes,
            config.validation_subjects,
        )
    sampler = PKBatchSampler(
        train_data,
        config.identities_per_batch,
        config.sequences_per_identity,
        config.seed,
        condition_aware=config.condition_aware_sampling,
    )
    train_loader = DataLoader(
        train_data,
        batch_sampler=sampler,
        num_workers=config.num_workers,
        pin_memory=True,
        persistent_workers=config.num_workers > 0,
    )
    test_loader = DataLoader(test_data, batch_size=config.batch_size, num_workers=config.num_workers, pin_memory=True)
    validation_loader = None
    if validation_data is not None:
        validation_loader = DataLoader(
            validation_data, batch_size=config.batch_size, num_workers=config.num_workers, pin_memory=True
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    design_module = importlib.import_module(f"designs.{config.design_name}.model")
    if not hasattr(design_module, "build_model"):
        raise AttributeError(f"Design {config.design_name!r} must export build_model(config, num_classes)")
    model = design_module.build_model(config, len(train_data.subject_to_label)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = None
    if config.scheduler_name == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(1, config.epochs),
            eta_min=config.scheduler_min_lr,
        )
    elif config.scheduler_name != "none":
        raise ValueError(f"Unsupported scheduler_name: {config.scheduler_name!r}")
    scaler = torch.amp.GradScaler("cuda", enabled=config.amp and device.type == "cuda")
    checkpoint_path = Path(config.output_dir) / "latest.pt"
    metrics_path = Path(config.output_dir) / "metrics.jsonl"
    visuals_dir = Path(config.output_dir) / "visuals"
    visuals_dir.mkdir(exist_ok=True)
    preview_batch = next(iter(DataLoader(train_data, batch_size=1)))
    try:
        save_input_preview(preview_batch, visuals_dir / "preprocessing_preview.png")
    except Exception as error:
        print(f"Warning: preprocessing preview was skipped: {error}", flush=True)
    monitor_metric = config.early_stopping_metric
    best_metric, best_rank1, stale_epochs = -1.0, -1.0, 0
    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scaler.load_state_dict(checkpoint["scaler"])
        if scheduler is not None and checkpoint.get("scheduler") is not None:
            scheduler.load_state_dict(checkpoint["scheduler"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_metric = float(
            checkpoint.get(
                "best_metric",
                checkpoint.get(f"best_{monitor_metric}", checkpoint.get("best_rank1", 0.0)),
            )
        )
        best_rank1 = float(checkpoint.get("best_rank1", -1.0))
        stale_epochs = int(checkpoint.get("stale_epochs", 0))
        print(f"Resuming at epoch {start_epoch}", flush=True)
    else:
        start_epoch = 0

    Path(config.output_dir, "config.json").write_text(json.dumps(config.to_dict(), indent=2))
    design_source = getattr(design_module, "__file__", None)
    if design_source:
        shutil.copy2(design_source, Path(config.output_dir) / "model_snapshot.py")
    reconstruction_preview_labels = None
    if hasattr(design_module, "reconstruction_preview_labels"):
        reconstruction_preview_labels = list(design_module.reconstruction_preview_labels())
    completed_epochs = start_epoch
    stopped_early = False
    for epoch in range(start_epoch, config.epochs):
        sampler.set_epoch(epoch)
        model.train()
        running = {
            "generative": 0.0,
            "recognition": 0.0,
            "generative_steps": 0,
            "recognition_steps": 0,
            "steps": 0,
        }
        reconstruction_preview = None
        for step, batch in enumerate(train_loader):
            silhouette, topology, labels = move_batch(batch, device)
            generative_step = (
                epoch < config.generative_warmup_epochs
                or step % max(1, config.generative_step_interval) == 0
            )
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=config.amp and device.type == "cuda"):
                if generative_step:
                    output = model(silhouette, topology, reconstruct=True, mask_ratio=config.mask_ratio)
                    reconstruction_target = topology[:, :, :2]
                    if hasattr(design_module, "build_reconstruction_target"):
                        reconstruction_target = design_module.build_reconstruction_target(silhouette, topology)
                    if hasattr(design_module, "compute_reconstruction_loss"):
                        loss = design_module.compute_reconstruction_loss(
                            output["reconstruction"], reconstruction_target, output["mask"], config
                        )
                    else:
                        loss = masked_reconstruction_loss(
                            output["reconstruction"], reconstruction_target, output["mask"], config.lambda_radius
                        )
                    running["generative"] += float(loss.detach())
                    running["generative_steps"] += 1
                    if reconstruction_preview is None:
                        reconstruction_preview = (
                            silhouette.detach().cpu(),
                            topology.detach().cpu(),
                            output["reconstruction"].detach().cpu(),
                            output["mask"].detach().cpu(),
                        )
                else:
                    output = model(silhouette, topology)
                    contrast = supervised_contrastive_loss(output["projection"], labels, config.temperature)
                    triplet = batch_hard_triplet_loss(output["embedding"], labels, config.triplet_margin)
                    loss = config.lambda_contrastive * contrast + config.lambda_triplet * triplet
                    if config.lambda_ce > 0:
                        if "logits" not in output:
                            raise KeyError("config.lambda_ce > 0 requires the model output to include 'logits'")
                        identity = F.cross_entropy(output["logits"], labels, label_smoothing=config.label_smoothing)
                        loss = loss + config.lambda_ce * identity
                    running["recognition"] += float(loss.detach())
                    running["recognition_steps"] += 1
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()
            running["steps"] += 1

        evaluation = evaluate(model, test_loader, device, config.eval_gallery_per_subject)
        validation_evaluation = None
        if validation_loader is not None:
            validation_evaluation = evaluate(model, validation_loader, device, config.eval_gallery_per_subject)
        if scheduler is not None:
            scheduler.step()
        # Model-selection metrics: monitor the validation split when one exists
        # (subject-disjoint from both train and test) so early stopping never
        # looks at the test split; otherwise fall back to the pre-existing
        # behavior of monitoring the test split directly.
        monitor_source = validation_evaluation if validation_evaluation is not None else evaluation
        if monitor_metric not in monitor_source:
            raise KeyError(
                f"early_stopping_metric={monitor_metric!r} is not in evaluation metrics: {sorted(monitor_source)}"
            )
        current_metric = float(monitor_source[monitor_metric])
        current_rank1 = float(monitor_source["rank1"])
        monitoring_active = epoch >= config.early_stopping_start_epoch
        rank1_improved = monitoring_active and current_rank1 > best_rank1 + config.early_stopping_min_delta
        if rank1_improved:
            best_rank1 = current_rank1
            torch.save(model.state_dict(), Path(config.output_dir) / "best_rank1_model.pt")
        improved = monitoring_active and current_metric > best_metric + config.early_stopping_min_delta
        if improved:
            best_metric = current_metric
            stale_epochs = 0
            torch.save(model.state_dict(), Path(config.output_dir) / "best_model.pt")
        elif monitoring_active and epoch >= config.generative_warmup_epochs:
            stale_epochs += 1
        epoch_metrics = {
            "epoch": epoch,
            **running,
            "generative_avg": running["generative"] / max(running["generative_steps"], 1),
            "recognition_avg": running["recognition"] / max(running["recognition_steps"], 1),
            "learning_rate": optimizer.param_groups[0]["lr"],
            **evaluation,
            **({f"val_{key}": value for key, value in validation_evaluation.items()} if validation_evaluation else {}),
            "monitor_metric": monitor_metric,
            "monitor_split": "validation" if validation_evaluation is not None else "test",
            "monitoring_active": monitoring_active,
            f"best_{monitor_metric}": best_metric,
            "best_rank1": best_rank1,
        }
        with metrics_path.open("a") as handle:
            handle.write(json.dumps(epoch_metrics) + "\n")
        temporary = checkpoint_path.with_suffix(".tmp")
        torch.save(
            {
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict(),
                "scheduler": scheduler.state_dict() if scheduler is not None else None,
                "best_metric": best_metric,
                f"best_{monitor_metric}": best_metric,
                "best_rank1": best_rank1,
                "stale_epochs": stale_epochs,
                "config": config.to_dict(),
            },
            temporary,
        )
        os.replace(temporary, checkpoint_path)
        try:
            if reconstruction_preview and ((epoch + 1) % config.visual_every_n_epochs == 0 or epoch == start_epoch):
                save_reconstruction_preview(
                    *reconstruction_preview,
                    visuals_dir / f"reconstruction_epoch_{epoch + 1:03d}.png",
                    labels=reconstruction_preview_labels,
                )
            save_training_curves(metrics_path, visuals_dir / "training_curves.png")
        except Exception as error:
            # Figures are evidence artifacts, not part of optimization. A plotting
            # incompatibility must never waste a completed training epoch.
            print(f"Warning: epoch visuals were skipped: {error}", flush=True)
        if commit_callback:
            commit_callback()
        print(json.dumps(epoch_metrics), flush=True)
        completed_epochs = epoch + 1
        if config.early_stopping_patience > 0 and stale_epochs >= config.early_stopping_patience:
            stopped_early = True
            print(
                f"Early stopping after {completed_epochs} epochs; {monitor_metric} did not improve for {stale_epochs} epochs.",
                flush=True,
            )
            break

    result = {
        "status": "early_stopped" if stopped_early else "complete",
        "epochs_completed": completed_epochs,
        "monitor_metric": monitor_metric,
        f"best_{monitor_metric}": best_metric,
        "best_rank1": best_rank1,
        "output_dir": config.output_dir,
    }
    Path(config.output_dir, "result_summary.json").write_text(json.dumps(result, indent=2))
    if commit_callback:
        commit_callback()
    return result
