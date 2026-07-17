# Post-training Analysis Report

Run directory: `runs/paper_smplgait_v1/casia_001`

## Training-history highlights

- Epochs recorded: `75`
- Best Rank-1: `0.5425` at epoch `1`
- Best Rank-5: `0.8376` at epoch `39`
- Best verification AUC: `0.8849` at epoch `51`
- Best verification accuracy: `0.8041` at epoch `69`
- Best distance gap: `0.6526` at epoch `57`

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

- Checkpoint: `runs/paper_smplgait_v1/casia_001/best_rank1_model.pt`
- Test sequences: `1197`
- Test subjects: `50`
- Rank-1: `0.4699`
- Rank-5: `0.7784`
- Rank-10: `0.9016`
- Verification AUC: `0.8819`
- Balanced verification accuracy at best threshold: `0.8086`
- EER estimate: `0.1915`

## Per-condition retrieval

| Condition | Probes | Rank-1 | Rank-5 | Rank-10 | Median rank |
|---|---:|---:|---:|---:|---:|
| clothing | 300 | 0.2567 | 0.5933 | 0.7667 | 4.0 |
| normal | 747 | 0.5556 | 0.8527 | 0.9558 | 1.0 |
