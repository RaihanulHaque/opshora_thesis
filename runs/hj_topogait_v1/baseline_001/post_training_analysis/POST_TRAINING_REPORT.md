# Post-Training Report: hj_topogait_v1 / baseline_001

## Training summary

- Epochs recorded: 62 (early stopped)
- Best Rank-1: **61.22%** (epoch 54)
- Best Rank-5: 90.15% (epoch 42)
- Verification AUC / accuracy / distance gap: **not available** — this run's
  `metrics.jsonl` predates those fields being added to `gait/train.py`'s
  `evaluate()`, so they were never recorded (not a data-loss issue, they
  simply didn't exist yet at the time this run was trained).

## Model evaluation

Skipped. This run trains on `casia_c` (`/data/silhouette-C.zip` on Modal,
CASIA-C), a different dataset than the CASIA-B silhouettes present locally
(`datasets/GaitDatasetB-silh`, `datasets/CASIA_B_Hamilton_Skeleton`), so a
local checkpoint re-evaluation against `tools/post_training_analysis.py`'s
default CASIA-B dataset paths isn't meaningful. Only the training-curve
plots (from `metrics.jsonl`, generated every epoch during training) are
included here.

## Note on checkpoint download

`best_model.pt` and `latest.pt` were previously downloaded corrupted
(0 bytes / unreadable). They have since been re-downloaded from Modal and
verified to load correctly via `torch.load`.
