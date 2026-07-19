# Post-training Analysis Report

Run directory: `runs/skeleton_silhouette_tcn_v1/tcn_clop_crosscond_001`

## Training-history highlights

- Epochs recorded: `61`
- Best Rank-1: `0.6154` at epoch `15`
- Best Rank-5: `1.0000` at epoch `1`
- Best verification AUC: `0.6264` at epoch `37`
- Best verification accuracy: `0.5844` at epoch `37`
- Best distance gap: `0.0857` at epoch `17`

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

- Checkpoint: `runs/skeleton_silhouette_tcn_v1/tcn_clop_crosscond_001/best_rank1_model.pt`
- Test sequences: `28`
- Test subjects: `5`
- Rank-1: `0.6154`
- Rank-5: `1.0000`
- Rank-10: `1.0000`
- Verification AUC: `0.5619`
- Balanced verification accuracy at best threshold: `0.5670`
- EER estimate: `0.4672`

## Per-condition retrieval

| Condition | Probes | Rank-1 | Rank-5 | Rank-10 | Median rank |
|---|---:|---:|---:|---:|---:|
| clothing | 13 | 0.6154 | 1.0000 | 1.0000 | 1.0 |
