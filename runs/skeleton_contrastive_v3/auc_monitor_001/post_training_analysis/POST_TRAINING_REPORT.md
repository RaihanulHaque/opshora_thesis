# Post-training Analysis Report

Run directory: `runs/skeleton_contrastive_v3/auc_monitor_001`

## Training-history highlights

- Epochs recorded: `52`
- Best Rank-1: `0.4699` at epoch `49`
- Best Rank-5: `0.7568` at epoch `40`
- Best verification AUC: `0.8894` at epoch `40`
- Best verification accuracy: `0.7997` at epoch `40`
- Best distance gap: `0.5709` at epoch `40`

## Generated graph files

- `00_original_training_curves.png`
- `01_losses.png`
- `02_retrieval_rank.png`
- `03_verification.png`
- `04_distances.png`
- `05_learning_rate.png`
- `06_training_steps.png`

## Local checkpoint evaluation

Skipped. Run this script without `--skip-model-eval` to load the `.pt` model and evaluate on the local test dataset.
