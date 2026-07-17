# Post-training Analysis Report

Run directory: `runs/paper_smplgait_v1/clopgait_001`

## Training-history highlights

- Epochs recorded: `84`
- Best Rank-1: `0.7826` at epoch `1`
- Best Rank-5: `1.0000` at epoch `1`
- Best verification AUC: `0.8258` at epoch `60`
- Best verification accuracy: `0.7536` at epoch `74`
- Best distance gap: `0.7185` at epoch `74`

## Generated graph files

- `00_original_training_curves.png`
- `01_losses.png`
- `02_retrieval_rank.png`
- `03_verification.png`
- `04_distances.png`
- `05_learning_rate.png`
- `06_training_steps.png`
- `07_distance_histogram.png`
- `08_roc_curve.png`
- `09_cmc_curve.png`
- `10_embedding_pca_subjects.png`

## Local checkpoint evaluation

- Checkpoint: `runs/paper_smplgait_v1/clopgait_001/best_rank1_model.pt`
- Test sequences: `58`
- Test subjects: `4`
- Rank-1: `0.7826`
- Rank-5: `1.0000`
- Rank-10: `1.0000`
- Verification AUC: `0.6779`
- Balanced verification accuracy at best threshold: `0.6465`
- EER estimate: `0.3599`

## Per-condition retrieval

| Condition | Probes | Rank-1 | Rank-5 | Rank-10 | Median rank |
|---|---:|---:|---:|---:|---:|
| clothing | 24 | 0.6667 | 1.0000 | 1.0000 | 1.0 |
| normal | 22 | 0.9091 | 1.0000 | 1.0000 | 1.0 |
