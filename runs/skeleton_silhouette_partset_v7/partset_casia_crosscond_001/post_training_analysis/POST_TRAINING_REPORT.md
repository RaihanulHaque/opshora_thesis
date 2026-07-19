# Post-training Analysis Report

Run directory: `runs/skeleton_silhouette_partset_v7/partset_casia_crosscond_001`

## Training-history highlights

- Epochs recorded: `75`
- Best Rank-1: `0.7757` at epoch `71`
- Best Rank-5: `0.9649` at epoch `74`
- Best verification AUC: `0.9042` at epoch `51`
- Best verification accuracy: `0.8053` at epoch `57`
- Best distance gap: `0.4662` at epoch `10`

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

- Checkpoint: `runs/skeleton_silhouette_partset_v7/partset_casia_crosscond_001/best_rank1_model.pt`
- Test sequences: `742`
- Test subjects: `124`
- Rank-1: `0.7757`
- Rank-5: `0.9568`
- Rank-10: `0.9757`
- Verification AUC: `0.8891`
- Balanced verification accuracy at best threshold: `0.8057`
- EER estimate: `0.1957`

## Per-condition retrieval

| Condition | Probes | Rank-1 | Rank-5 | Rank-10 | Median rank |
|---|---:|---:|---:|---:|---:|
| clothing | 370 | 0.7757 | 0.9568 | 0.9757 | 1.0 |
