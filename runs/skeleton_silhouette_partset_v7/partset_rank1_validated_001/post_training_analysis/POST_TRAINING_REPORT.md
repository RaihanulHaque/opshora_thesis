# Post-training Analysis Report

Run directory: `runs/skeleton_silhouette_partset_v7/partset_rank1_validated_001`

## Training-history highlights

- Epochs recorded: `90`
- Best Rank-1: `0.6543` at epoch `63`
- Best Rank-5: `0.9198` at epoch `58`
- Best verification AUC: `0.9160` at epoch `60`
- Best verification accuracy: `0.8374` at epoch `60`
- Best distance gap: `0.5837` at epoch `81`

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

- Checkpoint: `runs/skeleton_silhouette_partset_v7/partset_rank1_validated_001/best_rank1_model.pt`
- Test sequences: `1197`
- Test subjects: `50`
- Rank-1: `0.6514`
- Rank-5: `0.9112`
- Rank-10: `0.9599`
- Verification AUC: `0.9125`
- Balanced verification accuracy at best threshold: `0.8409`
- EER estimate: `0.1608`

## Per-condition retrieval

| Condition | Probes | Rank-1 | Rank-5 | Rank-10 | Median rank |
|---|---:|---:|---:|---:|---:|
| clothing | 300 | 0.3800 | 0.7533 | 0.8733 | 2.0 |
| normal | 747 | 0.7604 | 0.9746 | 0.9946 | 1.0 |
