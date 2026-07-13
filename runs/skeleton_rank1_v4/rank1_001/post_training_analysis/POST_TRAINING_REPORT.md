# Post-training Analysis Report

Run directory: `runs/skeleton_rank1_v4/rank1_001`

## Training-history highlights

- Epochs recorded: `64`
- Best Rank-1: `0.3095` at epoch `50`
- Best Rank-5: `0.6809` at epoch `56`
- Best verification AUC: `0.8580` at epoch `36`
- Best verification accuracy: `0.7836` at epoch `34`
- Best distance gap: `0.6187` at epoch `20`

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
