# Post-training Analysis Report

Run directory: `runs/paper_3dlocal_v1/clopgait_001`

## Training-history highlights

- Epochs recorded: `73`
- Best Rank-1: `0.8478` at epoch `52`
- Best Rank-5: `1.0000` at epoch `1`
- Best verification AUC: `0.8686` at epoch `49`
- Best verification accuracy: `0.7933` at epoch `49`
- Best distance gap: `0.7026` at epoch `67`

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

- Checkpoint: `runs/paper_3dlocal_v1/clopgait_001/best_rank1_model.pt`
- Test sequences: `58`
- Test subjects: `4`
- Rank-1: `0.8478`
- Rank-5: `1.0000`
- Rank-10: `1.0000`
- Verification AUC: `0.8464`
- Balanced verification accuracy at best threshold: `0.7667`
- EER estimate: `0.2384`

## Per-condition retrieval

| Condition | Probes | Rank-1 | Rank-5 | Rank-10 | Median rank |
|---|---:|---:|---:|---:|---:|
| clothing | 24 | 0.7917 | 1.0000 | 1.0000 | 1.0 |
| normal | 22 | 0.9091 | 1.0000 | 1.0000 | 1.0 |
