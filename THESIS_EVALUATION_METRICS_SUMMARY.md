# Thesis Evaluation Metrics and Protocol Summary

This document answers which evaluation metrics/graphs are available for the current V6 gait-recognition model and which ones still require extra experiments/logging.

Current evaluated model:

```text
runs/fusion_rank1_002/best_rank1_model.pt
```

Current design:

```text
designs/skeleton_silhouette_fusion_v6
```

Main local evaluation report:

```text
runs/fusion_rank1_002/post_training_analysis/POST_TRAINING_REPORT.md
```

## 1. Short answer: are these possible?

| Requested item | Possible here? | Current status |
|---|---|---|
| Accuracy | Yes | Computed as balanced verification accuracy and retrieval Rank-k accuracy. |
| Precision | Yes | Computed for pairwise verification at best threshold. |
| Recall | Yes | Computed for pairwise verification at best threshold. |
| F1-score | Yes | Computed for pairwise verification at best threshold. |
| ROC curve | Yes | Already generated. |
| AUC | Yes | Already computed. |
| Train/validation/test split | Partially | Train/test split exists; no separate validation split in current run. |
| Cross-domain testing setup | Possible, but not performed yet | Needs a second dataset/domain or a domain-held-out split. |
| GPU memory requirement | Approximate only | Modal config used 8 GB RAM and L4/A10/T4 GPU; exact peak VRAM was not logged. |
| Training time | Not exactly from downloaded metrics | Metrics have no timestamps; Modal logs would be needed for exact time. |
| Inference efficiency | Possible | Local CPU inference completed; exact timing should be logged in a timed rerun if needed for thesis. |

## 2. Classification-style verification metrics

Because this model is not a closed-set classifier, these metrics are computed from pairwise same/different verification.

Task:

```text
Given two gait sequences, decide whether they belong to the same subject.
```

Using the best threshold found from pairwise distances:

```text
Accuracy  = 82.62%
Precision = 8.74%
Recall    = 81.98%
F1-score  = 15.79%
AUC       = 90.25%
EER       = 17.43%
```

Important note:

The precision and F1 are low because all-pairs verification is extremely imbalanced:

```text
same-person pairs:      13,732
different-person pairs: 702,074
```

There are far more negative pairs than positive pairs. For this reason, the most thesis-relevant verification metrics are:

```text
Verification AUC = 90.25%
Balanced accuracy = 82.62%
EER = 17.43%
Distance gap = 0.5591
```

## 3. Retrieval metrics

The model also supports retrieval evaluation:

```text
Given a probe gait sequence, retrieve the closest matching subject from the gallery.
```

Current V6 retrieval metrics:

```text
3-gallery Rank-1  = 61.99%
Rank-3            = 82.62%
Rank-5            = 89.40%
Rank-10           = 95.13%
Median rank       = 1.0
Mean rank         = 2.80
```

Important wording:

Because the config uses:

```json
"eval_gallery_per_subject": 3
```

report this as:

```text
3-gallery Rank-1 = 61.99%
```

Do not call it single-gallery Rank-1.

## 4. Distance-separation metrics

The core thesis objective is contrastive distance separation:

```text
same person -> low distance
different person -> high distance
```

Current result:

```text
same_distance      = 0.3158
different_distance = 0.8748
distance_gap       = 0.5591
```

Interpretation:

The average different-person distance is much higher than the average same-person distance. This supports the claim that the model learned a meaningful gait embedding space.

## 5. ROC curve and AUC

ROC curve figure:

```text
runs/fusion_rank1_002/post_training_analysis/08_roc_curve.png
```

AUC:

```text
AUC = 0.9025
```

Thesis wording:

```text
The proposed fused gait model achieved a verification AUC of 0.9025 on unseen test subjects, indicating strong separation between same-subject and different-subject gait pairs.
```

## 6. Data splitting and evaluation protocol

### Dataset summary

Prepared fused dataset:

```text
Total subjects:   124
Total sequences:  2964
Missing pairs:    0
Sequence length:  30 frames
Resolution:       64 x 64
Representation:   silhouette + Hamilton skeleton + structure blur + temporal motion
```

### Train/test split

The current run used a subject-disjoint split:

```text
Training subjects: 001-074
Testing subjects:  075-124
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

### Validation split

The current run does not have a fully separate validation split.

Current behavior:

- training uses subjects `001-074`;
- evaluation/early stopping uses unseen subjects `075-124`;
- the final downloaded model is evaluated again on the same unseen test split.

For a stricter thesis protocol, use:

```text
Train:      subjects 001-064
Validation: subjects 065-074
Test:       subjects 075-124
```

or:

```text
Train:      60% subjects
Validation: 20% subjects
Test:       20% subjects
```

The important rule is:

```text
No subject should appear in more than one split.
```

## 7. Condition-wise evaluation

The test set contains normal walking and clothing-change walking.

| Condition | Probes | Rank-1 | Rank-5 | Rank-10 |
|---|---:|---:|---:|---:|
| Normal walking | 747 | 72.96% | 96.52% | 99.33% |
| Clothing change | 300 | 34.67% | 71.67% | 84.67% |

Interpretation:

The model performs strongly on normal walking but clothing variation remains difficult. This is expected because clothing changes alter silhouette/body-shape cues. This can be discussed as a limitation and future improvement direction.

## 8. Cross-domain testing setup

Cross-domain testing is possible but has not been completed yet.

A proper cross-domain setup would mean:

```text
Train on Dataset A
Test on Dataset B without fine-tuning
```

Example options:

### Option A: CASIA-B to Opshora custom dataset

```text
Train: CASIA-B Hamilton skeleton + silhouette fusion
Test:  Opshora's newly collected dataset
```

This is the strongest future thesis experiment because it tests real generalization.

### Option B: CASIA-B normal to clothing-change

This is a weaker but still useful domain-shift experiment:

```text
Train/gallery emphasis: normal walking
Probe testing: clothing-change sequences
```

The current condition-wise evaluation already partially shows this difficulty:

```text
normal Rank-1:   72.96%
clothing Rank-1: 34.67%
```

### Option C: CASIA-B to CASIA-C

If the small silhouette-C dataset is compatible, it can be used as another domain. However, because the subject identities are different and the representation differs, this would be a verification/generalization experiment rather than direct closed-set classification.

## 9. GPU memory requirement

Exact peak GPU memory was not logged.

What we can honestly report:

```text
Training platform: Modal cloud GPU
Requested GPU: L4 / A10 / T4 fallback
CPU cores: 2
Container memory: 8 GB
Mixed precision: enabled
Batch structure: 8 identities x 4 sequences = 32 sequences/batch
Input size: 30 frames x 4 channels x 64 x 64
```

Practical statement:

```text
The model was trained successfully on a single Modal cloud GPU using an 8 GB RAM container and mixed precision. The selected GPU was one of L4, A10, or T4 depending on Modal availability.
```

For exact peak VRAM, a future run should log:

```python
torch.cuda.max_memory_allocated()
torch.cuda.max_memory_reserved()
```

## 10. Training time

Exact training time is not available from the downloaded `metrics.jsonl` because it does not contain timestamps.

Known information:

```text
Training completed epochs: 88
Early stopping: yes
Reason: verification AUC did not improve for 24 epochs
Steps per epoch: 55
Total approximate optimization steps: 88 x 55 = 4840
```

To report exact training time, use one of:

1. Modal app run duration from Modal dashboard/logs.
2. Add timestamp logging to `gait/train.py`.
3. Use `/usr/bin/time` or Python `time.perf_counter()` around the training call.

Suggested thesis wording if exact dashboard time is unavailable:

```text
The model trained for 88 epochs and stopped automatically via early stopping after the verification AUC failed to improve for 24 consecutive monitored epochs.
```

Avoid inventing an exact hour/minute value unless it is recovered from Modal logs.

## 11. Inference efficiency

Inference efficiency is possible to report, but should be measured with a timed rerun for exactness.

Current local evaluation:

```text
Device: CPU
Test sequences evaluated: 1197
Embedding dimension: 256
```

The script can be extended to log:

```text
seconds per sequence
sequences per second
embedding extraction time
pairwise metric computation time
```

For thesis reporting, it is better to measure inference on the target device:

- CPU laptop;
- Mac MPS if available;
- Modal GPU.

Suggested efficiency metrics:

```text
Average inference time per sequence
Sequences processed per second
Model parameter count
Embedding dimensionality
```

## 12. Figures already generated

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

## 13. Best thesis-ready metric block

Use this block in a report or slide:

```text
The proposed V6 skeleton-silhouette fusion model was evaluated on 50 unseen CASIA-B subjects.
The model achieved 3-gallery Rank-1 accuracy of 61.99%, Rank-5 accuracy of 89.40%, and Rank-10 accuracy of 95.13%.
For pairwise blind verification, it achieved AUC of 90.25%, balanced accuracy of 82.62%, and EER of 17.43%.
The average same-subject embedding distance was 0.3158, while the average different-subject distance was 0.8748, producing a distance gap of 0.5591.
```

