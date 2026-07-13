# Post-training Analysis Report

Run directory: `runs/skeleton_rank1_v5/rank1_multigallery_001`

## Training-history highlights

- Epochs recorded: `16`
- Best Rank-1: `0.4212` at epoch `2`
- Best Rank-5: `0.7135` at epoch `16`
- Best verification AUC: `0.8530` at epoch `13`
- Best verification accuracy: `0.7680` at epoch `13`
- Best distance gap: `0.4634` at epoch `12`

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
