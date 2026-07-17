# Post-training Analysis Report

Run directory: `runs/paper_3dlocal_v1/casia_001`

## Training-history highlights

- Epochs recorded: `70`
- Best Rank-1: `0.4556` at epoch `46`
- Best Rank-5: `0.7650` at epoch `45`
- Best verification AUC: `0.8596` at epoch `61`
- Best verification accuracy: `0.7810` at epoch `61`
- Best distance gap: `0.5754` at epoch `63`

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

- Checkpoint: `runs/paper_3dlocal_v1/casia_001/best_rank1_model.pt`
- Test sequences: `1197`
- Test subjects: `50`
- Rank-1: `0.4546`
- Rank-5: `0.7660`
- Rank-10: `0.8749`
- Verification AUC: `0.8460`
- Balanced verification accuracy at best threshold: `0.7654`
- EER estimate: `0.2349`

## Per-condition retrieval

| Condition | Probes | Rank-1 | Rank-5 | Rank-10 | Median rank |
|---|---:|---:|---:|---:|---:|
| clothing | 300 | 0.2033 | 0.5100 | 0.7000 | 5.0 |
| normal | 747 | 0.5556 | 0.8688 | 0.9451 | 1.0 |
