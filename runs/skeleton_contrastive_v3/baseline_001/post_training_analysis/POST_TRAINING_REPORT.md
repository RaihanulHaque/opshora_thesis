# Post-training Analysis Report

Run directory: `runs/skeleton_contrastive_v3/baseline_001`

## Training-history highlights

- Epochs recorded: `29`
- Best Rank-1: `0.4568` at epoch `18`
- Best Rank-5: `0.7463` at epoch `18`
- Best verification AUC: `0.8792` at epoch `27`
- Best verification accuracy: `0.7867` at epoch `27`
- Best distance gap: `0.5573` at epoch `26`

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
