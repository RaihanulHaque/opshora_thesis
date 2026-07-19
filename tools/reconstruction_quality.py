"""Quantitative evaluation of the masked-reconstruction ("generative") branch.

The training loop periodically saves reconstruction *preview images*
(runs/<design>/<run>/visuals/reconstruction_epoch_*.png) so a human can eyeball
whether the decoder's guesses for hidden frames look plausible. That is not a
mathematical answer to "is the reconstruction good" -- it is qualitative and
depends on which few example sequences happened to be sampled that epoch.

This script answers the question with numbers, computed over the *entire*
held-out test split of a trained checkpoint, using the exact same masking
protocol as training (`model(sil, top, reconstruct=True, mask_ratio=...)`):
some fraction of frames in each sequence are zeroed out before the encoder
ever sees them, and the decoder has to reconstruct the skeleton map and the
motion map for those hidden frames purely from the temporal context of the
frames it *did* see. Metrics are reported separately for:

  - "masked" frames only: the true generative-quality test (never seen by
    the encoder for that forward pass) -- this is the number that matters.
  - "all" frames: every frame the decoder reconstructed, including the ones
    the encoder saw directly. This is NOT an upper bound -- the training loss
    (`masked_reconstruction_loss` / `compute_reconstruction_loss`) only
    back-propagates through the *masked* frames (see `selected_frames` in
    those functions), so the decoder is never directly supervised on visible
    frames. In practice the "all frames" numbers come out *worse* than the
    "masked frames" numbers for exactly this reason -- it is evidence the
    masked-frame objective is doing what it is supposed to, not a bug. The
    "masked frames" number is the one that answers "is the generative branch
    good"; "all frames" is reported only for transparency/completeness.

Metrics, computed per design's two reconstruction channels (skeleton
occupancy map + motion map, both in [0, 1] after sigmoid):
  - Skeleton channel (treated as a binary segmentation problem, same as the
    training loss): Dice coefficient, IoU, pixel accuracy, precision, recall,
    plus the raw BCE value used in training, for continuity with the loss
    curve in metrics.jsonl.
  - Motion channel (continuous heatmap): MSE, MAE, PSNR, and windowed SSIM
    (structural similarity, computed with a 7x7 uniform-filter window --
    scipy is already a dependency, no new package needed).

Usage:
    python tools/reconstruction_quality.py \
        --run-dir runs/skeleton_silhouette_partset_v7/partset_rank1_001 \
        --skeleton-dataset runs/fusion_rank1_002/local_processed_cache_full \
        --cache-dir runs/fusion_rank1_002/local_processed_cache_full \
        --mask-ratio 0.3

Writes <output-dir>/reconstruction_quality.json and a Markdown summary
table next to it.
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def load_config(run_dir: Path, args: argparse.Namespace):
    from gait.config import ExperimentConfig

    values = json.loads((run_dir / "config.json").read_text())
    values.update(
        {
            "dataset_path": args.skeleton_dataset,
            "silhouette_dataset_path": args.silhouette_dataset,
            "cache_dir": args.cache_dir,
            "experiments_root": str(run_dir / "local_eval_experiments"),
            "run_name": "local_reconstruction_eval",
        }
    )
    return ExperimentConfig.from_dict(values)


def _ssim_batch(pred: np.ndarray, target: np.ndarray, win: int = 7) -> np.ndarray:
    """Per-frame SSIM. pred/target: (N, H, W) float32 in [0, 1]. Returns (N,)."""
    from scipy.ndimage import uniform_filter

    c1 = (0.01 * 1.0) ** 2
    c2 = (0.03 * 1.0) ** 2

    def local_mean_var(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        mu = uniform_filter(x, size=win, axes=(1, 2))
        var = uniform_filter(x * x, size=win, axes=(1, 2)) - mu * mu
        return mu, var

    mu_p, var_p = local_mean_var(pred)
    mu_t, var_t = local_mean_var(target)
    covar = uniform_filter(pred * target, size=win, axes=(1, 2)) - mu_p * mu_t
    numerator = (2 * mu_p * mu_t + c1) * (2 * covar + c2)
    denominator = (mu_p**2 + mu_t**2 + c1) * (var_p + var_t + c2)
    ssim_map = numerator / np.clip(denominator, 1e-12, None)
    return ssim_map.mean(axis=(1, 2))


def _binary_metrics(pred_prob: np.ndarray, target: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    pred_bin = pred_prob > threshold
    target_bin = target > threshold
    tp = float(np.logical_and(pred_bin, target_bin).sum())
    fp = float(np.logical_and(pred_bin, ~target_bin).sum())
    fn = float(np.logical_and(~pred_bin, target_bin).sum())
    tn = float(np.logical_and(~pred_bin, ~target_bin).sum())
    precision = tp / max(tp + fp, 1.0)
    recall = tp / max(tp + fn, 1.0)
    dice = 2 * tp / max(2 * tp + fp + fn, 1.0)
    iou = tp / max(tp + fp + fn, 1.0)
    accuracy = (tp + tn) / max(tp + tn + fp + fn, 1.0)
    eps = 1e-7
    bce = float(
        -np.mean(
            target * np.log(np.clip(pred_prob, eps, 1 - eps))
            + (1 - target) * np.log(np.clip(1 - pred_prob, eps, 1 - eps))
        )
    )
    return {"dice": dice, "iou": iou, "precision": precision, "recall": recall, "pixel_accuracy": accuracy, "bce": bce}


def _continuous_metrics(pred_prob: np.ndarray, target: np.ndarray) -> dict[str, float]:
    diff = pred_prob - target
    mse = float(np.mean(diff**2))
    mae = float(np.mean(np.abs(diff)))
    psnr = float(20 * np.log10(1.0) - 10 * np.log10(max(mse, 1e-12)))
    ssim = float(np.mean(_ssim_batch(pred_prob.astype(np.float32), target.astype(np.float32))))
    return {"mse": mse, "mae": mae, "psnr_db": psnr, "ssim": ssim}


def run(args: argparse.Namespace) -> dict:
    import torch
    from torch.utils.data import DataLoader

    from gait.dataset import GaitSequenceDataset
    from gait.preprocessing import prepare_dataset

    run_dir = args.run_dir
    config = load_config(run_dir, args)
    prepare_dataset(config, force=args.force_preprocess)

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
    loader = DataLoader(test_data, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=False)

    design_module = importlib.import_module(f"designs.{config.design_name}.model")
    if not hasattr(design_module, "compute_reconstruction_loss") and not hasattr(design_module, "build_reconstruction_target"):
        raise SystemExit(f"Design {config.design_name!r} has no reconstruction/decoder hooks -- nothing to evaluate.")

    model = design_module.build_model(config, len(train_data.subject_to_label))
    checkpoint_path = args.checkpoint or (run_dir / "best_rank1_model.pt")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    device = torch.device("cpu") if args.cpu else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    mask_ratio = args.mask_ratio if args.mask_ratio is not None else config.mask_ratio

    skeleton_masked_pred, skeleton_masked_tgt = [], []
    skeleton_all_pred, skeleton_all_tgt = [], []
    motion_masked_pred, motion_masked_tgt = [], []
    motion_all_pred, motion_all_tgt = [], []
    frames_seen = 0
    frames_masked = 0
    sequences_seen = 0

    torch.manual_seed(config.seed)
    with torch.no_grad():
        for batch in loader:
            silhouette = batch["silhouette"].to(device)
            topology = batch["topology"].to(device)
            output = model(silhouette, topology, reconstruct=True, mask_ratio=mask_ratio)
            if hasattr(design_module, "build_reconstruction_target"):
                target = design_module.build_reconstruction_target(silhouette, topology)
            else:
                target = topology[:, :, :2]
            prediction = output["reconstruction"]
            mask = output["mask"]  # (B, T) bool, True = frame was hidden from the encoder

            skeleton_prob = torch.sigmoid(prediction[:, :, :1]).cpu().numpy()
            motion_prob = torch.sigmoid(prediction[:, :, 1:2]).cpu().numpy()
            skeleton_target = target[:, :, :1].cpu().numpy()
            motion_target = target[:, :, 1:2].cpu().numpy()
            mask_np = mask.cpu().numpy()

            b, t = mask_np.shape
            skeleton_prob = skeleton_prob.reshape(b, t, *skeleton_prob.shape[2:])[:, :, 0]
            motion_prob = motion_prob.reshape(b, t, *motion_prob.shape[2:])[:, :, 0]
            skeleton_target = skeleton_target.reshape(b, t, *skeleton_target.shape[2:])[:, :, 0]
            motion_target = motion_target.reshape(b, t, *motion_target.shape[2:])[:, :, 0]

            skeleton_all_pred.append(skeleton_prob.reshape(-1, *skeleton_prob.shape[2:]))
            skeleton_all_tgt.append(skeleton_target.reshape(-1, *skeleton_target.shape[2:]))
            motion_all_pred.append(motion_prob.reshape(-1, *motion_prob.shape[2:]))
            motion_all_tgt.append(motion_target.reshape(-1, *motion_target.shape[2:]))

            if mask_np.any():
                skeleton_masked_pred.append(skeleton_prob[mask_np])
                skeleton_masked_tgt.append(skeleton_target[mask_np])
                motion_masked_pred.append(motion_prob[mask_np])
                motion_masked_tgt.append(motion_target[mask_np])

            frames_seen += b * t
            frames_masked += int(mask_np.sum())
            sequences_seen += b
            if args.max_batches and sequences_seen // args.batch_size >= args.max_batches:
                break

    def stack(parts: list[np.ndarray]) -> np.ndarray:
        return np.concatenate(parts, axis=0) if parts else np.zeros((0, 1, 1), dtype=np.float32)

    results: dict = {
        "design_name": config.design_name,
        "checkpoint": str(checkpoint_path),
        "test_cache_dir": config.cache_dir,
        "split_mode": config.split_mode,
        "mask_ratio_used": mask_ratio,
        "sequences_evaluated": sequences_seen,
        "frames_total": frames_seen,
        "frames_masked": frames_masked,
        "masked_frame_fraction": frames_masked / max(frames_seen, 1),
    }

    if frames_masked > 0:
        results["masked_frames"] = {
            "skeleton": _binary_metrics(stack(skeleton_masked_pred), stack(skeleton_masked_tgt)),
            "motion": _continuous_metrics(stack(motion_masked_pred), stack(motion_masked_tgt)),
        }
    results["all_frames"] = {
        "skeleton": _binary_metrics(stack(skeleton_all_pred), stack(skeleton_all_tgt)),
        "motion": _continuous_metrics(stack(motion_all_pred), stack(motion_all_tgt)),
    }
    return results


def write_markdown(results: dict, path: Path) -> None:
    lines = [
        "# Reconstruction quality: quantitative evaluation",
        "",
        f"- Design: `{results['design_name']}`",
        f"- Checkpoint: `{results['checkpoint']}`",
        f"- Test cache: `{results['test_cache_dir']}` (split_mode={results['split_mode']})",
        f"- Mask ratio used for this evaluation: {results['mask_ratio_used']}",
        f"- Sequences evaluated: {results['sequences_evaluated']}",
        f"- Frames total: {results['frames_total']}, masked (hidden from encoder): {results['frames_masked']} "
        f"({results['masked_frame_fraction']:.1%})",
        "",
        "## Masked frames (true generative-quality test: encoder never saw these frames)",
        "",
    ]
    if "masked_frames" in results:
        sk = results["masked_frames"]["skeleton"]
        mo = results["masked_frames"]["motion"]
        lines += [
            "| Channel | Metric | Value |",
            "|---|---|---|",
            f"| Skeleton (binary occupancy) | Dice | {sk['dice']:.4f} |",
            f"| Skeleton | IoU | {sk['iou']:.4f} |",
            f"| Skeleton | Precision | {sk['precision']:.4f} |",
            f"| Skeleton | Recall | {sk['recall']:.4f} |",
            f"| Skeleton | Pixel accuracy | {sk['pixel_accuracy']:.4f} |",
            f"| Skeleton | BCE | {sk['bce']:.4f} |",
            f"| Motion (continuous heatmap) | MSE | {mo['mse']:.6f} |",
            f"| Motion | MAE | {mo['mae']:.6f} |",
            f"| Motion | PSNR (dB) | {mo['psnr_db']:.2f} |",
            f"| Motion | SSIM | {mo['ssim']:.4f} |",
            "",
        ]
    else:
        lines += ["(mask_ratio was 0 for this run -- no masked frames to evaluate.)", ""]

    lines += [
        "## All frames (includes frames the encoder saw directly -- NOT an upper bound, see note below)",
        "",
    ]
    sk = results["all_frames"]["skeleton"]
    mo = results["all_frames"]["motion"]
    lines += [
        "| Channel | Metric | Value |",
        "|---|---|---|",
        f"| Skeleton (binary occupancy) | Dice | {sk['dice']:.4f} |",
        f"| Skeleton | IoU | {sk['iou']:.4f} |",
        f"| Skeleton | Precision | {sk['precision']:.4f} |",
        f"| Skeleton | Recall | {sk['recall']:.4f} |",
        f"| Skeleton | Pixel accuracy | {sk['pixel_accuracy']:.4f} |",
        f"| Skeleton | BCE | {sk['bce']:.4f} |",
        f"| Motion (continuous heatmap) | MSE | {mo['mse']:.6f} |",
        f"| Motion | MAE | {mo['mae']:.6f} |",
        f"| Motion | PSNR (dB) | {mo['psnr_db']:.2f} |",
        f"| Motion | SSIM | {mo['ssim']:.4f} |",
        "",
        "## How to read this",
        "",
        "- **Dice / IoU** (0 to 1, higher is better) treat the skeleton map as a binary",
        "  segmentation problem: does the decoder put a body-part pixel where one",
        "  actually is? Dice above ~0.7 is considered a strong overlap in medical/",
        "  gait segmentation literature; below ~0.3 means the reconstruction is close",
        "  to guessing.",
        "- **PSNR** (dB, higher is better) is the standard image-fidelity metric used in",
        "  autoencoder/GAN reconstruction papers; 20-30 dB is typical for lossy but",
        "  recognizable reconstructions of small (64x64) maps.",
        "- **SSIM** (-1 to 1, higher is better) captures structural similarity rather than",
        "  raw pixel error, so it does not reward a decoder that outputs a blurry mean",
        "  frame -- it rewards actually recovering the pose's shape and motion pattern.",
        "- These are computed strictly on the **held-out test split**, using the same",
        "  temporal masking the model saw during training, so they are not inflated by",
        "  memorization of the training set.",
        "- **Why 'all frames' looks worse than 'masked frames':** the training loss only",
        "  ever back-propagates through masked (hidden) frames -- the decoder is not",
        "  directly supervised to reconstruct frames the encoder could already see. So",
        "  'masked frames' is the metric that reflects what the model was actually",
        "  trained to do, and it is the one that should be quoted as \"reconstruction",
        "  quality\"; 'all frames' is reported only for completeness.",
        "",
    ]
    path.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description="Quantitative masked-reconstruction quality evaluation.")
    parser.add_argument("--run-dir", default="runs/fusion_rank1_002", type=Path)
    parser.add_argument("--checkpoint", default=None, type=Path)
    parser.add_argument("--output-dir", default=None, type=Path)
    parser.add_argument("--skeleton-dataset", default="datasets/CASIA_B_Hamilton_Skeleton")
    parser.add_argument("--silhouette-dataset", default="datasets/GaitDatasetB-silh")
    parser.add_argument("--cache-dir", default="runs/fusion_rank1_002/local_processed_cache")
    parser.add_argument("--mask-ratio", default=None, type=float, help="Defaults to the run's own config.mask_ratio")
    parser.add_argument("--batch-size", default=64, type=int)
    parser.add_argument("--num-workers", default=0, type=int)
    parser.add_argument("--max-batches", default=0, type=int, help="0 = evaluate the full test split")
    parser.add_argument("--force-preprocess", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    output_dir = args.output_dir or (args.run_dir / "post_training_analysis")
    output_dir.mkdir(parents=True, exist_ok=True)

    results = run(args)
    (output_dir / "reconstruction_quality.json").write_text(json.dumps(results, indent=2))
    write_markdown(results, output_dir / "RECONSTRUCTION_QUALITY.md")
    print(json.dumps(results, indent=2))
    print(f"\nWrote {output_dir / 'reconstruction_quality.json'} and {output_dir / 'RECONSTRUCTION_QUALITY.md'}")


if __name__ == "__main__":
    main()
