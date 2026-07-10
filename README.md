# Opshora Gait Thesis Codebase

This repository contains the current working prototype for Opshora's gait-recognition thesis.

The main model is a V6 skeleton-silhouette fusion network that learns a distance-based gait embedding using both generative reconstruction and contrastive metric learning.

## Main docs

Read these first:

```text
THESIS_CONTEXT.md
THESIS_EVALUATION_AND_METRICS_GUIDE.md
```

Detailed V6 architecture:

```text
designs/skeleton_silhouette_fusion_v6/MODEL_ARCHITECTURE_AND_FLOW.md
```

## Current best run

```text
runs/fusion_rank1_002/
```

Main local evaluation report:

```text
runs/fusion_rank1_002/post_training_analysis/POST_TRAINING_REPORT.md
```

Main result:

```text
3-gallery Rank-1:                61.99%
Rank-5:                          89.40%
Rank-10:                         95.13%
Verification AUC:                90.25%
Balanced verification accuracy:  82.62%
EER estimate:                    17.43%
```

## Repository layout

```text
gait/                         shared preprocessing, dataset, training, losses
designs/                      separate model designs / experiment history
designs/skeleton_silhouette_fusion_v6/
                              current best architecture
tools/post_training_analysis.py
                              local graph + checkpoint evaluation script
runs/                         downloaded Modal experiment outputs
datasets/                     local datasets
modal_app.py                  Modal training app
submit_modal.py               experiment submitter
```

## Run training on Modal

Deploy after code/config changes:

```bash
modal deploy modal_app.py
```

Submit the current V6 design:

```bash
python submit_modal.py run --design skeleton_silhouette_fusion_v6 --run fusion_rank1_003
```

Modal training runs independently, so the laptop can be closed after submission.

## Run local post-training evaluation

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

This generates plots, CSV files, saved embeddings, and a markdown report under:

```text
runs/fusion_rank1_002/post_training_analysis/
```

## Test live silhouette and skeleton

Use the webcam test script to preview:

```text
camera | white silhouette | aligned 64x64 silhouette | Hamilton skeleton
```

```bash
/Users/rahi/miniconda3/envs/ML/bin/python tools/live_silhouette_cam.py
```

Press `q` to quit.

## Current thesis status

What is working:

- paired CASIA-B silhouette + Hamilton skeleton preprocessing;
- V6 two-stream fusion model;
- masked generative reconstruction branch;
- contrastive/triplet verification branch;
- Modal training;
- local checkpoint evaluation;
- Rank-k, ROC-AUC, EER, distance-gap, condition-wise metrics.

What still needs work for a stronger final thesis:

- custom Opshora cross-domain dataset;
- separate validation split;
- exact training-time and peak GPU-memory logging;
- baseline/ablation comparison table;
- cross-domain testing on a second dataset.
