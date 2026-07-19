from __future__ import annotations

import argparse
import csv
import importlib
import json
import math
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def load_rows(metrics_path: Path) -> list[dict]:
    return [json.loads(line) for line in metrics_path.read_text().splitlines() if line.strip()]


def metric_values(rows: list[dict], key: str) -> list[float]:
    values: list[float] = []
    for row in rows:
        value = row.get(key)
        values.append(float(value) if value is not None else float("nan"))
    return values


def ensure_matplotlib():
    import os

    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def save_line_plot(path: Path, title: str, xlabel: str, ylabel: str, x: list[int], series: dict[str, list[float]], ylim=None) -> None:
    try:
        plt = ensure_matplotlib()
    except ModuleNotFoundError:
        save_svg_line_plot(path.with_suffix(".svg"), title, xlabel, ylabel, x, series, ylim)
        return
    fig, ax = plt.subplots(figsize=(9, 5))
    for label, values in series.items():
        ax.plot(x, values, label=label, linewidth=2)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _svg_escape(value: str) -> str:
    return value.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def save_svg_line_plot(
    path: Path,
    title: str,
    xlabel: str,
    ylabel: str,
    x: list[int],
    series: dict[str, list[float]],
    ylim=None,
) -> None:
    """Dependency-free line plot fallback for laptops without matplotlib."""
    width, height = 980, 560
    left, right, top, bottom = 82, 30, 48, 78
    plot_w = width - left - right
    plot_h = height - top - bottom
    clean_series = {
        label: [(xv, yv) for xv, yv in zip(x, values) if not math.isnan(float(yv))]
        for label, values in series.items()
    }
    all_points = [point for points in clean_series.values() for point in points]
    if not all_points:
        path.write_text("<svg xmlns='http://www.w3.org/2000/svg'></svg>\n")
        return
    x_min, x_max = min(point[0] for point in all_points), max(point[0] for point in all_points)
    if ylim is None:
        y_min, y_max = min(point[1] for point in all_points), max(point[1] for point in all_points)
        margin = max((y_max - y_min) * 0.08, 1e-6)
        y_min -= margin
        y_max += margin
    else:
        y_min, y_max = ylim
    if x_min == x_max:
        x_max += 1
    if y_min == y_max:
        y_max += 1

    def sx(value: float) -> float:
        return left + (value - x_min) / (x_max - x_min) * plot_w

    def sy(value: float) -> float:
        return top + (y_max - value) / (y_max - y_min) * plot_h

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
    chunks = [
        f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}' viewBox='0 0 {width} {height}'>",
        "<rect width='100%' height='100%' fill='white'/>",
        f"<text x='{width/2}' y='28' text-anchor='middle' font-size='20' font-family='Arial'>{_svg_escape(title)}</text>",
        f"<line x1='{left}' y1='{top}' x2='{left}' y2='{top + plot_h}' stroke='#333'/>",
        f"<line x1='{left}' y1='{top + plot_h}' x2='{left + plot_w}' y2='{top + plot_h}' stroke='#333'/>",
    ]
    for index in range(6):
        gx = left + index * plot_w / 5
        gy = top + index * plot_h / 5
        y_value = y_max - index * (y_max - y_min) / 5
        x_value = x_min + index * (x_max - x_min) / 5
        chunks.append(f"<line x1='{gx}' y1='{top}' x2='{gx}' y2='{top + plot_h}' stroke='#ddd'/>")
        chunks.append(f"<line x1='{left}' y1='{gy}' x2='{left + plot_w}' y2='{gy}' stroke='#ddd'/>")
        chunks.append(f"<text x='{gx}' y='{top + plot_h + 22}' text-anchor='middle' font-size='11' font-family='Arial'>{x_value:.0f}</text>")
        chunks.append(f"<text x='{left - 10}' y='{gy + 4}' text-anchor='end' font-size='11' font-family='Arial'>{y_value:.3g}</text>")
    chunks.append(f"<text x='{left + plot_w/2}' y='{height - 24}' text-anchor='middle' font-size='14' font-family='Arial'>{_svg_escape(xlabel)}</text>")
    chunks.append(
        f"<text transform='translate(22 {top + plot_h/2}) rotate(-90)' text-anchor='middle' font-size='14' font-family='Arial'>{_svg_escape(ylabel)}</text>"
    )
    legend_y = 58
    for index, (label, points) in enumerate(clean_series.items()):
        color = colors[index % len(colors)]
        if len(points) >= 2:
            path_data = " ".join(f"{sx(px):.2f},{sy(py):.2f}" for px, py in points)
            chunks.append(f"<polyline fill='none' stroke='{color}' stroke-width='2.5' points='{path_data}'/>")
        elif len(points) == 1:
            chunks.append(f"<circle cx='{sx(points[0][0]):.2f}' cy='{sy(points[0][1]):.2f}' r='3' fill='{color}'/>")
        lx = width - 260
        ly = legend_y + index * 20
        chunks.append(f"<line x1='{lx}' y1='{ly}' x2='{lx + 28}' y2='{ly}' stroke='{color}' stroke-width='3'/>")
        chunks.append(f"<text x='{lx + 36}' y='{ly + 4}' font-size='12' font-family='Arial'>{_svg_escape(label)}</text>")
    chunks.append("</svg>")
    path.write_text("\n".join(chunks) + "\n")


def save_training_plots(run_dir: Path, rows: list[dict], output_dir: Path) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    epochs = [int(row["epoch"]) + 1 for row in rows]
    generated: dict[str, str] = {}

    plot_specs = [
        (
            "01_losses.png",
            "Training losses",
            "Average loss",
            {
                "generative_avg": metric_values(rows, "generative_avg"),
                "recognition_avg": metric_values(rows, "recognition_avg"),
            },
            None,
        ),
        (
            "02_retrieval_rank.png",
            "Unseen-subject retrieval",
            "Accuracy",
            {
                "Rank-1": metric_values(rows, "rank1"),
                "Rank-5": metric_values(rows, "rank5"),
                "Best Rank-1": metric_values(rows, "best_rank1"),
            },
            (0, 1),
        ),
        (
            "03_verification.png",
            "Blind verification metrics",
            "Score",
            {
                "Verification AUC": metric_values(rows, "verification_auc"),
                "Verification accuracy": metric_values(rows, "verification_accuracy"),
            },
            (0, 1),
        ),
        (
            "04_distances.png",
            "Embedding distance separation",
            "Cosine distance",
            {
                "Same-person distance": metric_values(rows, "same_distance"),
                "Different-person distance": metric_values(rows, "different_distance"),
                "Distance gap": metric_values(rows, "distance_gap"),
            },
            None,
        ),
        (
            "05_learning_rate.png",
            "Learning-rate schedule",
            "Learning rate",
            {"learning_rate": metric_values(rows, "learning_rate")},
            None,
        ),
        (
            "06_training_steps.png",
            "Generative vs recognition training steps",
            "Steps per epoch",
            {
                "generative_steps": metric_values(rows, "generative_steps"),
                "recognition_steps": metric_values(rows, "recognition_steps"),
            },
            None,
        ),
    ]
    for filename, title, ylabel, series, ylim in plot_specs:
        path = output_dir / filename
        save_line_plot(path, title, "Epoch", ylabel, epochs, series, ylim)
        actual_path = path if path.exists() else path.with_suffix(".svg")
        generated[actual_path.name] = str(actual_path)

    original_curve = run_dir / "visuals" / "training_curves.png"
    if original_curve.exists():
        copied = output_dir / "00_original_training_curves.png"
        shutil.copy2(original_curve, copied)
        generated[copied.name] = str(copied)
    return generated


def best_row(rows: list[dict], key: str) -> dict:
    valid = [row for row in rows if key in row and row[key] is not None and not math.isnan(float(row[key]))]
    return max(valid, key=lambda row: float(row[key]))


def summarize_training(rows: list[dict], output_dir: Path) -> dict:
    summary = {
        "epochs_recorded": len(rows),
        "best_rank1": {
            "epoch": int(best_row(rows, "rank1")["epoch"]) + 1,
            "value": float(best_row(rows, "rank1")["rank1"]),
        },
        "best_rank5": {
            "epoch": int(best_row(rows, "rank5")["epoch"]) + 1,
            "value": float(best_row(rows, "rank5")["rank5"]),
        },
        "best_verification_auc": {
            "epoch": int(best_row(rows, "verification_auc")["epoch"]) + 1,
            "value": float(best_row(rows, "verification_auc")["verification_auc"]),
        },
        "best_verification_accuracy": {
            "epoch": int(best_row(rows, "verification_accuracy")["epoch"]) + 1,
            "value": float(best_row(rows, "verification_accuracy")["verification_accuracy"]),
        },
        "best_distance_gap": {
            "epoch": int(best_row(rows, "distance_gap")["epoch"]) + 1,
            "value": float(best_row(rows, "distance_gap")["distance_gap"]),
        },
        "final_epoch": rows[-1],
    }
    (output_dir / "training_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def load_config(run_dir: Path, args: argparse.Namespace) -> ExperimentConfig:
    from gait.config import ExperimentConfig

    values = json.loads((run_dir / "config.json").read_text())
    values.update(
        {
            "dataset_path": args.skeleton_dataset,
            "silhouette_dataset_path": args.silhouette_dataset,
            "cache_dir": args.cache_dir,
            "experiments_root": str(run_dir / "local_eval_experiments"),
            "run_name": "local_post_training_eval",
        }
    )
    if args.train_subjects is not None:
        values["train_subjects"] = args.train_subjects
    return ExperimentConfig.from_dict(values)


def extract_embeddings(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> dict[str, np.ndarray | list[str]]:
    import torch
    from torch.nn import functional as F

    model.eval()
    embeddings: list[torch.Tensor] = []
    subjects: list[str] = []
    conditions: list[str] = []
    views: list[str] = []
    with torch.no_grad():
        for batch in loader:
            output = model(batch["silhouette"].to(device), batch["topology"].to(device))
            embeddings.append(F.normalize(output["embedding"], dim=1).cpu())
            subjects.extend(batch["subject"])
            conditions.extend(batch["condition"])
            views.extend(batch["view"])
    return {
        "embeddings": torch.cat(embeddings).numpy(),
        "subjects": subjects,
        "conditions": conditions,
        "views": views,
    }


def condition_family(condition: str) -> str:
    condition = condition.lower()
    if condition.startswith(("nm", "fn")):
        return "normal"
    if condition.startswith(("bg", "fb")):
        return "bag"
    if condition.startswith("cl"):
        return "clothing"
    if condition.startswith("fq"):
        return "fast"
    if condition.startswith("fs"):
        return "slow"
    return condition


def build_gallery_probe(subjects: list[str], conditions: list[str], gallery_per_subject: int) -> tuple[list[int], list[list[int]], list[int], list[str]]:
    gallery_indices: list[int] = []
    gallery_groups: list[list[int]] = []
    probe_indices: list[int] = []
    gallery_subjects: list[str] = []
    for subject in sorted(set(subjects)):
        candidates = [index for index, value in enumerate(subjects) if value == subject]
        normal = [index for index in candidates if condition_family(conditions[index]) == "normal"]
        preferred = normal if normal else candidates
        galleries = preferred[: max(1, min(gallery_per_subject, len(preferred)))]
        gallery_groups.append(galleries)
        gallery_indices.extend(galleries)
        gallery_subjects.append(subject)
        gallery_set = set(galleries)
        probe_indices.extend(index for index in candidates if index not in gallery_set)
    return gallery_indices, gallery_groups, probe_indices, gallery_subjects


def retrieval_metrics(
    embeddings: np.ndarray,
    subjects: list[str],
    conditions: list[str],
    views: list[str],
    gallery_per_subject: int,
) -> tuple[dict, list[dict]]:
    gallery_indices, gallery_groups, probe_indices, gallery_subjects = build_gallery_probe(
        subjects, conditions, gallery_per_subject
    )
    gallery_matrix = embeddings[gallery_indices]
    probe_matrix = embeddings[probe_indices]
    flat_similarities = probe_matrix @ gallery_matrix.T
    subject_scores: list[np.ndarray] = []
    offset = 0
    for group in gallery_groups:
        group_size = len(group)
        subject_scores.append(flat_similarities[:, offset : offset + group_size].max(axis=1))
        offset += group_size
    similarities = np.stack(subject_scores, axis=1)
    order = np.argsort(-similarities, axis=1)
    gallery_subjects_array = np.asarray(gallery_subjects)
    ranked = gallery_subjects_array[order]
    probe_subjects = np.asarray([subjects[index] for index in probe_indices])

    ranks: list[int] = []
    rows: list[dict] = []
    for row_index, subject in enumerate(probe_subjects):
        matches = np.flatnonzero(ranked[row_index] == subject)
        rank = int(matches[0]) + 1 if len(matches) else len(gallery_subjects) + 1
        ranks.append(rank)
        probe_index = probe_indices[row_index]
        rows.append(
            {
                "probe_index": probe_index,
                "subject": subject,
                "condition": conditions[probe_index],
                "condition_family": condition_family(conditions[probe_index]),
                "view": views[probe_index],
                "rank": rank,
                "top1_subject": ranked[row_index, 0],
                "top1_score": float(similarities[row_index, order[row_index, 0]]),
            }
        )
    ranks_array = np.asarray(ranks)
    metrics = {
        "gallery_per_subject": gallery_per_subject,
        "probes": len(probe_indices),
        "gallery_subjects": len(gallery_subjects),
        "rank1": float(np.mean(ranks_array <= 1)),
        "rank3": float(np.mean(ranks_array <= 3)),
        "rank5": float(np.mean(ranks_array <= 5)),
        "rank10": float(np.mean(ranks_array <= 10)),
        "median_rank": float(np.median(ranks_array)),
        "mean_rank": float(np.mean(ranks_array)),
    }
    for family in sorted(set(row["condition_family"] for row in rows)):
        selected = [row for row in rows if row["condition_family"] == family]
        family_ranks = np.asarray([row["rank"] for row in selected])
        metrics[f"rank1_{family}"] = float(np.mean(family_ranks <= 1))
        metrics[f"rank5_{family}"] = float(np.mean(family_ranks <= 5))
        metrics[f"probes_{family}"] = len(selected)
    return metrics, rows


def pairwise_verification_metrics(embeddings: np.ndarray, subjects: list[str]) -> tuple[dict, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    similarity = embeddings @ embeddings.T
    distance = 1.0 - similarity
    same: list[float] = []
    different: list[float] = []
    y_true: list[int] = []
    scores: list[float] = []
    for left in range(len(subjects)):
        for right in range(left + 1, len(subjects)):
            is_same = subjects[left] == subjects[right]
            value = float(distance[left, right])
            if is_same:
                same.append(value)
                y_true.append(1)
            else:
                different.append(value)
                y_true.append(0)
            scores.append(-value)
    same_array = np.asarray(same, dtype=np.float32)
    different_array = np.asarray(different, dtype=np.float32)
    y_true_array = np.asarray(y_true, dtype=np.int32)
    scores_array = np.asarray(scores, dtype=np.float32)

    different_sorted = np.sort(different_array)
    not_greater = np.searchsorted(different_sorted, same_array, side="right")
    auc = float(np.mean(1.0 - not_greater / max(len(different_sorted), 1)))
    thresholds = np.unique(np.quantile(np.concatenate([same_array, different_array]), np.linspace(0, 1, 301)))
    best = {
        "threshold": 0.0,
        "accuracy": -1.0,
        "balanced_accuracy": -1.0,
        "precision": 0.0,
        "recall": 0.0,
        "specificity": 0.0,
        "f1": 0.0,
    }
    roc_rows: list[tuple[float, float, float]] = []
    eer = 1.0
    eer_gap = float("inf")
    for threshold in thresholds:
        predict_same = np.concatenate([same_array, different_array]) <= threshold
        truth = np.concatenate([np.ones_like(same_array, dtype=bool), np.zeros_like(different_array, dtype=bool)])
        tp = float(np.sum(predict_same & truth))
        fp = float(np.sum(predict_same & ~truth))
        fn = float(np.sum(~predict_same & truth))
        tn = float(np.sum(~predict_same & ~truth))
        accuracy = (tp + tn) / max(tp + fp + fn + tn, 1.0)
        precision = tp / max(tp + fp, 1.0)
        recall = tp / max(tp + fn, 1.0)
        specificity = tn / max(fp + tn, 1.0)
        balanced_accuracy = (recall + specificity) / 2.0
        f1 = 2 * precision * recall / max(precision + recall, 1e-8)
        fpr = fp / max(fp + tn, 1.0)
        tpr = recall
        fnr = fn / max(tp + fn, 1.0)
        gap = abs(fpr - fnr)
        if gap < eer_gap:
            eer_gap = gap
            eer = (fpr + fnr) / 2.0
        roc_rows.append((fpr, tpr, float(threshold)))
        if balanced_accuracy > best["balanced_accuracy"]:
            best = {
                "threshold": float(threshold),
                "accuracy": float(accuracy),
                "balanced_accuracy": float(balanced_accuracy),
                "precision": float(precision),
                "recall": float(recall),
                "specificity": float(specificity),
                "f1": float(f1),
            }
    metrics = {
        "same_distance": float(same_array.mean()),
        "different_distance": float(different_array.mean()),
        "distance_gap": float(different_array.mean() - same_array.mean()),
        "verification_auc": auc,
        "eer_estimate": float(eer),
        "best_threshold": best["threshold"],
        "best_threshold_accuracy": best["accuracy"],
        "best_threshold_balanced_accuracy": best["balanced_accuracy"],
        "best_threshold_precision": best["precision"],
        "best_threshold_recall": best["recall"],
        "best_threshold_specificity": best["specificity"],
        "best_threshold_f1": best["f1"],
        "same_pairs": int(len(same_array)),
        "different_pairs": int(len(different_array)),
    }
    roc = np.asarray(sorted(roc_rows), dtype=np.float32)
    return metrics, same_array, different_array, y_true_array, scores_array, roc


def save_eval_plots(output_dir: Path, same: np.ndarray, different: np.ndarray, roc: np.ndarray, cmc: dict[int, float], pca_payload: dict) -> None:
    plt = ensure_matplotlib()

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(same, bins=50, alpha=0.65, label="same person", density=True)
    ax.hist(different, bins=50, alpha=0.65, label="different person", density=True)
    ax.set_title("Verification distance distributions")
    ax.set_xlabel("Cosine distance")
    ax.set_ylabel("Density")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "07_distance_histogram.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(roc[:, 0], roc[:, 1], linewidth=2)
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
    ax.set_title("Verification ROC curve")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "08_roc_curve.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    xs = sorted(cmc)
    ax.plot(xs, [cmc[x] for x in xs], marker="o", linewidth=2)
    ax.set_title("CMC retrieval curve")
    ax.set_xlabel("Rank k")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "09_cmc_curve.png", dpi=180)
    plt.close(fig)

    coords = pca_payload["coords"]
    labels = pca_payload["labels"]
    fig, ax = plt.subplots(figsize=(8, 6))
    unique = sorted(set(labels))[:20]
    for label in unique:
        selected = np.asarray([item == label for item in labels])
        ax.scatter(coords[selected, 0], coords[selected, 1], s=12, alpha=0.75, label=label)
    ax.set_title("Embedding PCA scatter\nfirst 20 test subjects")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.grid(alpha=0.2)
    ax.legend(ncol=2, fontsize=7)
    fig.tight_layout()
    fig.savefig(output_dir / "10_embedding_pca_subjects.png", dpi=180)
    plt.close(fig)


def pca_2d(embeddings: np.ndarray) -> np.ndarray:
    centered = embeddings - embeddings.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    return centered @ vt[:2].T


def save_csv(path: Path, rows: Iterable[dict]) -> None:
    rows = list(rows)
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def run_model_evaluation(run_dir: Path, args: argparse.Namespace, output_dir: Path) -> dict:
    import torch
    from torch.utils.data import DataLoader

    from gait.dataset import GaitSequenceDataset
    from gait.preprocessing import prepare_dataset

    config = load_config(run_dir, args)
    summary = prepare_dataset(config, force=args.force_preprocess)
    try:
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
    except ValueError as error:
        raise ValueError(
            f"{error}\n\n"
            "Local model evaluation needs enough paired skeleton+silhouette subjects to create an unseen-subject test split. "
            f"The prepared cache summary is: {summary}. "
            "If you just replaced the local dataset, rerun with --force-preprocess or use a new --cache-dir."
        ) from error
    loader = DataLoader(test_data, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=False)

    design_module = importlib.import_module(f"designs.{config.design_name}.model")
    model = design_module.build_model(config, len(train_data.subject_to_label))
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state_dict = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    if args.cpu:
        device = torch.device("cpu")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    model.to(device)

    payload = extract_embeddings(model, loader, device)
    embeddings = payload["embeddings"]
    subjects = list(payload["subjects"])
    conditions = list(payload["conditions"])
    views = list(payload["views"])

    retrieval, probe_rows = retrieval_metrics(embeddings, subjects, conditions, views, args.gallery_per_subject)
    verification, same, different, y_true, scores, roc = pairwise_verification_metrics(embeddings, subjects)
    cmc = {k: retrieval_at_k(probe_rows, k) for k in [1, 2, 3, 5, 10, 20, 30, 50]}
    condition_rows = condition_breakdown(probe_rows)

    save_csv(output_dir / "probe_retrieval_rows.csv", probe_rows)
    save_csv(output_dir / "condition_breakdown.csv", condition_rows)
    np.savez_compressed(
        output_dir / "embeddings_and_pair_scores.npz",
        embeddings=embeddings,
        subjects=np.asarray(subjects),
        conditions=np.asarray(conditions),
        views=np.asarray(views),
        y_true=y_true,
        pair_scores=scores,
    )
    coords = pca_2d(embeddings)
    save_eval_plots(output_dir, same, different, roc, cmc, {"coords": coords, "labels": subjects})
    result = {
        "dataset_summary": summary,
        "checkpoint": str(args.checkpoint),
        "device": str(device),
        "test_sequences": len(test_data),
        "test_subjects": len(set(subjects)),
        "retrieval": retrieval,
        "verification": verification,
        "cmc": {str(key): value for key, value in cmc.items()},
        "condition_breakdown": condition_rows,
    }
    (output_dir / "local_evaluation_summary.json").write_text(json.dumps(result, indent=2))
    return result


def retrieval_at_k(rows: list[dict], k: int) -> float:
    if not rows:
        return 0.0
    return float(np.mean([row["rank"] <= k for row in rows]))


def condition_breakdown(rows: list[dict]) -> list[dict]:
    grouped: dict[str, list[int]] = defaultdict(list)
    for row in rows:
        grouped[row["condition_family"]].append(int(row["rank"]))
    output: list[dict] = []
    for condition, ranks in sorted(grouped.items()):
        array = np.asarray(ranks)
        output.append(
            {
                "condition_family": condition,
                "probes": len(ranks),
                "rank1": float(np.mean(array <= 1)),
                "rank3": float(np.mean(array <= 3)),
                "rank5": float(np.mean(array <= 5)),
                "rank10": float(np.mean(array <= 10)),
                "median_rank": float(np.median(array)),
            }
        )
    return output


def write_markdown_report(run_dir: Path, output_dir: Path, training_summary: dict, eval_summary: dict | None) -> None:
    lines = [
        "# Post-training Analysis Report",
        "",
        f"Run directory: `{run_dir}`",
        "",
        "## Training-history highlights",
        "",
        f"- Epochs recorded: `{training_summary['epochs_recorded']}`",
        f"- Best Rank-1: `{training_summary['best_rank1']['value']:.4f}` at epoch `{training_summary['best_rank1']['epoch']}`",
        f"- Best Rank-5: `{training_summary['best_rank5']['value']:.4f}` at epoch `{training_summary['best_rank5']['epoch']}`",
        f"- Best verification AUC: `{training_summary['best_verification_auc']['value']:.4f}` at epoch `{training_summary['best_verification_auc']['epoch']}`",
        f"- Best verification accuracy: `{training_summary['best_verification_accuracy']['value']:.4f}` at epoch `{training_summary['best_verification_accuracy']['epoch']}`",
        f"- Best distance gap: `{training_summary['best_distance_gap']['value']:.4f}` at epoch `{training_summary['best_distance_gap']['epoch']}`",
        "",
        "## Generated graph files",
        "",
    ]
    for path in sorted([*output_dir.glob("*.png"), *output_dir.glob("*.svg")]):
        lines.append(f"- `{path.name}`")
    if eval_summary:
        lines.extend(
            [
                "",
                "## Local checkpoint evaluation",
                "",
                f"- Checkpoint: `{eval_summary['checkpoint']}`",
                f"- Test sequences: `{eval_summary['test_sequences']}`",
                f"- Test subjects: `{eval_summary['test_subjects']}`",
                f"- Rank-1: `{eval_summary['retrieval']['rank1']:.4f}`",
                f"- Rank-5: `{eval_summary['retrieval']['rank5']:.4f}`",
                f"- Rank-10: `{eval_summary['retrieval']['rank10']:.4f}`",
                f"- Verification AUC: `{eval_summary['verification']['verification_auc']:.4f}`",
                f"- Balanced verification accuracy at best threshold: `{eval_summary['verification']['best_threshold_balanced_accuracy']:.4f}`",
                f"- EER estimate: `{eval_summary['verification']['eer_estimate']:.4f}`",
                "",
                "## Per-condition retrieval",
                "",
                "| Condition | Probes | Rank-1 | Rank-5 | Rank-10 | Median rank |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        for row in eval_summary["condition_breakdown"]:
            lines.append(
                f"| {row['condition_family']} | {row['probes']} | {row['rank1']:.4f} | "
                f"{row['rank5']:.4f} | {row['rank10']:.4f} | {row['median_rank']:.1f} |"
            )
    else:
        lines.extend(
            [
                "",
                "## Local checkpoint evaluation",
                "",
                "Skipped. Run this script without `--skip-model-eval` to load the `.pt` model and evaluate on the local test dataset.",
            ]
        )
    (output_dir / "POST_TRAINING_REPORT.md").write_text("\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate post-training graphs and optional local checkpoint evaluation.")
    parser.add_argument("--run-dir", default="runs/fusion_rank1_002", type=Path)
    parser.add_argument("--checkpoint", default=None, type=Path)
    parser.add_argument("--output-dir", default=None, type=Path)
    parser.add_argument("--skeleton-dataset", default="datasets/CASIA_B_Hamilton_Skeleton")
    parser.add_argument("--silhouette-dataset", default="datasets/GaitDatasetB-silh")
    parser.add_argument("--cache-dir", default="runs/fusion_rank1_002/local_processed_cache")
    parser.add_argument("--gallery-per-subject", default=3, type=int)
    parser.add_argument("--batch-size", default=64, type=int)
    parser.add_argument("--num-workers", default=0, type=int)
    parser.add_argument("--train-subjects", default=None, type=int)
    parser.add_argument("--skip-model-eval", action="store_true")
    parser.add_argument("--force-preprocess", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir
    if args.checkpoint is None:
        args.checkpoint = run_dir / "best_rank1_model.pt"
    output_dir = args.output_dir or (run_dir / "post_training_analysis")
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = load_rows(run_dir / "metrics.jsonl")
    save_training_plots(run_dir, rows, output_dir)
    training_summary = summarize_training(rows, output_dir)
    eval_summary = None
    if not args.skip_model_eval:
        eval_summary = run_model_evaluation(run_dir, args, output_dir)
    write_markdown_report(run_dir, output_dir, training_summary, eval_summary)
    print(f"Saved post-training analysis to {output_dir}")


if __name__ == "__main__":
    main()
