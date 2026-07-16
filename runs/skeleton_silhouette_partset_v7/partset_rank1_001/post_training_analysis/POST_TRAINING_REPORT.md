# Post-training Analysis Report

Run directory: `runs/skeleton_silhouette_partset_v7/partset_rank1_001`

## Training-history highlights

- Epochs recorded: `87`
- Best Rank-1: `0.6791` at epoch `87`
- Best Rank-5: `0.9179` at epoch `74`
- Best verification AUC: `0.9148` at epoch `63`
- Best verification accuracy: `0.8382` at epoch `64`
- Best distance gap: `0.6411` at epoch `15`

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

- Checkpoint: `runs/skeleton_silhouette_partset_v7/partset_rank1_001/best_rank1_model.pt`
- Test sequences: `1197`
- Test subjects: `50`
- Rank-1: `0.6781`
- Rank-5: `0.9131`
- Rank-10: `0.9637`
- Verification AUC: `0.9113`
- Balanced verification accuracy at best threshold: `0.8357`
- EER estimate: `0.1647`

## Per-condition retrieval

| Condition | Probes | Rank-1 | Rank-5 | Rank-10 | Median rank |
|---|---:|---:|---:|---:|---:|
| clothing | 300 | 0.3633 | 0.7633 | 0.8933 | 2.0 |
| normal | 747 | 0.8046 | 0.9732 | 0.9920 | 1.0 |
