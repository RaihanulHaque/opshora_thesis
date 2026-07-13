# Post-training Analysis Report

Run directory: `runs/skeleton_silhouette_tcn_v1/tcn_baseline_001`

## Training-history highlights

- Epochs recorded: `100`
- Best Rank-1: `0.3610` at epoch `30`
- Best Rank-5: `0.7106` at epoch `73`
- Best verification AUC: `0.8394` at epoch `76`
- Best verification accuracy: `0.7617` at epoch `86`
- Best distance gap: `0.5513` at epoch `86`

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
