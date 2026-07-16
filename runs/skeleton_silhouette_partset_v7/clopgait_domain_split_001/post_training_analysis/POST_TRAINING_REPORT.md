# Post-training Analysis Report

Run directory: `runs/skeleton_silhouette_partset_v7/clopgait_domain_split_001`

## Training-history highlights

- Epochs recorded: `61`
- Best Rank-1: `0.8261` at epoch `37`
- Best Rank-5: `1.0000` at epoch `1`
- Best verification AUC: `0.8277` at epoch `37`
- Best verification accuracy: `0.7597` at epoch `48`
- Best distance gap: `0.7026` at epoch `35`

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

- Checkpoint: `runs/skeleton_silhouette_partset_v7/clopgait_domain_split_001/best_rank1_model.pt`
- Test sequences: `58`
- Test subjects: `4`
- Rank-1: `0.8261`
- Rank-5: `1.0000`
- Rank-10: `1.0000`
- Verification AUC: `0.8277`
- Balanced verification accuracy at best threshold: `0.8012`
- EER estimate: `0.2072`

## Per-condition retrieval

| Condition | Probes | Rank-1 | Rank-5 | Rank-10 | Median rank |
|---|---:|---:|---:|---:|---:|
| clothing | 24 | 0.7083 | 1.0000 | 1.0000 | 1.0 |
| normal | 22 | 0.9545 | 1.0000 | 1.0000 | 1.0 |
