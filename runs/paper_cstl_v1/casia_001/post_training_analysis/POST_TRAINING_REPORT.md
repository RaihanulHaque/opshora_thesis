# Post-training Analysis Report

Run directory: `runs/paper_cstl_v1/casia_001`

## Training-history highlights

- Epochs recorded: `76`
- Best Rank-1: `0.6810` at epoch `1`
- Best Rank-5: `0.9054` at epoch `69`
- Best verification AUC: `0.9014` at epoch `52`
- Best verification accuracy: `0.8187` at epoch `69`
- Best distance gap: `0.6051` at epoch `52`

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

- Checkpoint: `runs/paper_cstl_v1/casia_001/best_rank1_model.pt`
- Test sequences: `1197`
- Test subjects: `50`
- Rank-1: `0.6772`
- Rank-5: `0.9054`
- Rank-10: `0.9570`
- Verification AUC: `0.8989`
- Balanced verification accuracy at best threshold: `0.8283`
- EER estimate: `0.1735`

## Per-condition retrieval

| Condition | Probes | Rank-1 | Rank-5 | Rank-10 | Median rank |
|---|---:|---:|---:|---:|---:|
| clothing | 300 | 0.4033 | 0.7400 | 0.8700 | 2.0 |
| normal | 747 | 0.7871 | 0.9719 | 0.9920 | 1.0 |
