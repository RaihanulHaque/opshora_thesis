# Post-training Analysis Report

Run directory: `runs/skeleton_silhouette_transformer_v1/transformer_baseline_001`

## Training-history highlights

- Epochs recorded: `61`
- Best Rank-1: `0.3801` at epoch `4`
- Best Rank-5: `0.7058` at epoch `57`
- Best verification AUC: `0.8405` at epoch `31`
- Best verification accuracy: `0.7576` at epoch `24`
- Best distance gap: `0.5462` at epoch `25`

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
