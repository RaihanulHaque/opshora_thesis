# Post-training Analysis Report

Run directory: `runs/fusion_rank1_002`

## Training-history highlights

- Epochs recorded: `88`
- Best Rank-1: `0.6199` at epoch `81`
- Best Rank-5: `0.9016` at epoch `88`
- Best verification AUC: `0.9078` at epoch `79`
- Best verification accuracy: `0.8292` at epoch `72`
- Best distance gap: `0.6147` at epoch `57`

## Generated graph files

- `00_original_training_curves.png`
- `01_losses.png`
- `01_losses.svg`
- `02_retrieval_rank.png`
- `02_retrieval_rank.svg`
- `03_verification.png`
- `03_verification.svg`
- `04_distances.png`
- `04_distances.svg`
- `05_learning_rate.png`
- `05_learning_rate.svg`
- `06_training_steps.png`
- `06_training_steps.svg`
- `07_distance_histogram.png`
- `08_roc_curve.png`
- `09_cmc_curve.png`
- `10_embedding_pca_subjects.png`

## Local checkpoint evaluation

- Checkpoint: `runs/fusion_rank1_002/best_rank1_model.pt`
- Test sequences: `1197`
- Test subjects: `50`
- Rank-1: `0.6199`
- Rank-5: `0.8940`
- Rank-10: `0.9513`
- Verification AUC: `0.9025`
- Balanced verification accuracy at best threshold: `0.8262`
- EER estimate: `0.1743`

## Per-condition retrieval

| Condition | Probes | Rank-1 | Rank-5 | Rank-10 | Median rank |
|---|---:|---:|---:|---:|---:|
| clothing | 300 | 0.3467 | 0.7167 | 0.8467 | 3.0 |
| normal | 747 | 0.7296 | 0.9652 | 0.9933 | 1.0 |
