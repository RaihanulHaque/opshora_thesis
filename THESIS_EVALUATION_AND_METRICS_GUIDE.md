# Thesis Evaluation and Metrics Guide

This file combines the former post-training evaluation guide and the evaluation metrics summary.

Current evaluated run:

```text
runs/fusion_rank1_002/
```

Current evaluated checkpoint:

```text
runs/fusion_rank1_002/best_rank1_model.pt
```

Current post-training output folder:

```text
runs/fusion_rank1_002/post_training_analysis/
```

## 1. Thesis-ready result block

```text
The proposed V6 skeleton-silhouette fusion model was evaluated on 50 unseen CASIA-B subjects.
It achieved 3-gallery Rank-1 accuracy of 61.99%, Rank-5 accuracy of 89.40%, and Rank-10 accuracy of 95.13%.
For blind pairwise verification, it achieved AUC of 90.25%, balanced accuracy of 82.62%, and EER of 17.43%.
The average same-subject embedding distance was 0.3158, while the average different-subject distance was 0.8748, producing a distance gap of 0.5591.
```

## 2. Which requested metrics are available?

| Requested item | Possible here? | Current status |
|---|---|---|
| Accuracy | Yes | Balanced verification accuracy = 82.62%; Rank-k retrieval also reported. |
| Precision | Yes | Pairwise verification precision = 8.74%. |
| Recall | Yes | Pairwise verification recall = 81.98%. |
| F1-score | Yes | Pairwise verification F1 = 15.79%. |
| ROC curve | Yes | Generated as `08_roc_curve.png`. |
| AUC | Yes | Verification AUC = 90.25%. |
| Data splitting and evaluation protocol | Yes | Subject-disjoint train/test split documented below. |
| Train/validation/test split | Partially | Train/test exists; no separate validation split in this completed run. |
| Cross-domain testing setup | Possible, not completed | Needs Opshora custom dataset or another domain. |
| GPU memory requirement | Approximate only | Modal GPU class known; peak VRAM was not logged. |
| Training time | Approximate only | Epochs/steps known; exact wall-clock time needs Modal logs or timestamp logging. |
| Inference efficiency | Possible | Local inference completed; exact speed should be measured with a timed rerun. |

## 3. Verification metrics

The model is not a normal classifier at test time. It creates embeddings and compares distances.

Pairwise verification task:

```text
Given two gait sequences, decide whether they are from the same subject.
```

Computed from local evaluation:

```text
Accuracy / balanced accuracy = 82.62%
Precision                    = 8.74%
Recall                       = 81.98%
F1-score                     = 15.79%
AUC                          = 90.25%
EER                          = 17.43%
```

Important note:

Precision and F1 are low because all possible pairwise testing creates a very imbalanced set:

```text
same-person pairs:      13,732
different-person pairs: 702,074
```

So for thesis reporting, emphasize:

```text
Verification AUC
Balanced accuracy
EER
same_distance / different_distance / distance_gap
```

## 4. Retrieval metrics

Retrieval task:

```text
Given a probe gait sequence, retrieve the closest matching subject from the gallery.
```

Current retrieval result:

```text
3-gallery Rank-1 = 61.99%
Rank-3           = 82.62%
Rank-5           = 89.40%
Rank-10          = 95.13%
Median rank      = 1.0
Mean rank        = 2.80
```

Because the evaluation uses:

```json
"eval_gallery_per_subject": 3
```

write:

```text
3-gallery Rank-1
```

Do not call it single-gallery Rank-1.

## 5. Distance-separation metrics

```text
same_distance      = 0.3158
different_distance = 0.8748
distance_gap       = 0.5591
```

Interpretation:

The model learned an embedding space where same-person gait sequences are much closer than different-person gait sequences.

## 6. Dataset split and protocol

Prepared fused dataset:

```text
Total subjects:   124
Total sequences:  2964
Missing pairs:    0
Sequence length:  30 frames
Resolution:       64 x 64
Representation:   silhouette + Hamilton skeleton + structure blur + temporal motion
```

Subject-disjoint split:

```text
Train subjects: 001-074
Test subjects:  075-124
```

Counts:

```text
Train subjects:  74
Test subjects:   50
Train sequences: 1767
Test sequences:  1197
```

Condition counts:

```text
Train normal sequences:   1325
Train clothing sequences: 442
Test normal sequences:    897
Test clothing sequences:  300
```

Limitation:

```text
No separate validation split was used in this completed run.
```

Stricter final protocol recommendation:

```text
Train:      subjects 001-064
Validation: subjects 065-074
Test:       subjects 075-124
```

or any other subject-disjoint train/validation/test split.

## 7. Condition-wise evaluation

| Condition | Probes | Rank-1 | Rank-5 | Rank-10 |
|---|---:|---:|---:|---:|
| Normal walking | 747 | 72.96% | 96.52% | 99.33% |
| Clothing change | 300 | 34.67% | 71.67% | 84.67% |

Interpretation:

The model is strong for normal walking but clothing changes remain difficult. This is useful for the thesis limitation/future-work section.

## 8. Generated figures

Available figures:

```text
runs/fusion_rank1_002/post_training_analysis/01_losses.png
runs/fusion_rank1_002/post_training_analysis/02_retrieval_rank.png
runs/fusion_rank1_002/post_training_analysis/03_verification.png
runs/fusion_rank1_002/post_training_analysis/04_distances.png
runs/fusion_rank1_002/post_training_analysis/05_learning_rate.png
runs/fusion_rank1_002/post_training_analysis/06_training_steps.png
runs/fusion_rank1_002/post_training_analysis/07_distance_histogram.png
runs/fusion_rank1_002/post_training_analysis/08_roc_curve.png
runs/fusion_rank1_002/post_training_analysis/09_cmc_curve.png
runs/fusion_rank1_002/post_training_analysis/10_embedding_pca_subjects.png
```

Recommended figures for thesis:

1. Training loss curves.
2. Rank-1/Rank-5 retrieval curve.
3. Verification AUC curve.
4. Same vs different distance curve.
5. Distance histogram.
6. ROC curve.
7. CMC curve.
8. Embedding PCA scatter.
9. Reconstruction preview.
10. Preprocessing preview.

## 9. Local evaluation command

Use the `ML` conda environment:

```bash
/Users/rahi/miniconda3/envs/ML/bin/python tools/post_training_analysis.py \
  --run-dir runs/fusion_rank1_002 \
  --checkpoint runs/fusion_rank1_002/best_rank1_model.pt \
  --skeleton-dataset datasets/CASIA_B_Hamilton_Skeleton \
  --silhouette-dataset datasets/GaitDatasetB-silh \
  --cache-dir runs/fusion_rank1_002/local_processed_cache_full \
  --gallery-per-subject 3
```

Use `best_rank1_model.pt` for retrieval-focused reporting.

Use `best_model.pt` for verification-AUC-focused reporting.

The script automatically selects:

```text
CUDA -> MPS -> CPU
```

The latest local run used CPU because MPS was built but not available in the current shell.

## 10. Output files from local evaluation

```text
POST_TRAINING_REPORT.md
training_summary.json
local_evaluation_summary.json
condition_breakdown.csv
probe_retrieval_rows.csv
embeddings_and_pair_scores.npz
```

The local reader supports:

```text
datasets/GaitDatasetB-silh/001/
datasets/GaitDatasetB-silh/002.tar.gz
datasets/GaitDatasetB-silh/003.tar.gz
...
```

So every subject archive does not need to be manually extracted.

## 11. Cross-domain testing

Cross-domain testing is possible but has not been completed.

Best future setup:

```text
Train: CASIA-B fused skeleton/silhouette dataset
Test:  Opshora custom dataset without fine-tuning
```

Alternative domain-shift setup:

```text
Train/evaluate normal walking gallery
Probe with clothing-change sequences
```

The current condition-wise results already show that clothing-change is harder:

```text
Normal Rank-1:   72.96%
Clothing Rank-1: 34.67%
```

## 12. GPU memory and training time

Exact peak GPU memory was not logged.

Known training setup:

```text
Platform: Modal
GPU fallback: L4 / A10 / T4
CPU cores: 2
Container memory: 8 GB
Mixed precision: enabled
Batch structure: 8 identities x 4 sequences = 32 sequences/batch
Input size: 30 frames x 4 channels x 64 x 64
Epochs completed: 88
Steps per epoch: 55
Approximate optimization steps: 4840
```

Safe thesis wording:

```text
The model was trained successfully on a single Modal cloud GPU using mixed precision and an 8 GB RAM container. Training stopped after 88 epochs via early stopping.
```

Do not invent exact peak VRAM or wall-clock training time unless recovered from Modal logs.

Future logging should add:

```python
torch.cuda.max_memory_allocated()
torch.cuda.max_memory_reserved()
time.perf_counter()
```

## 13. Inference efficiency

Possible to report, but exact speed should be measured with a timed rerun.

Current local evaluation:

```text
Device: CPU
Test sequences: 1197
Embedding dimension: 256
```

Recommended future efficiency metrics:

```text
seconds per sequence
sequences per second
model parameter count
embedding dimensionality
```

