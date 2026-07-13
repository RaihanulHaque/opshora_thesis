# Post-training Analysis Report

Run directory: `runs/skeleton_silhouette_lstm_v1/lstm_baseline_001`

## Training-history highlights

- Epochs recorded: `50`
- Best Rank-1: `0.2894` at epoch `2`
- Best Rank-5: `0.5482` at epoch `38`
- Best verification AUC: `0.8064` at epoch `38`
- Best verification accuracy: `0.7452` at epoch `26`
- Best distance gap: `0.5883` at epoch `26`

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
