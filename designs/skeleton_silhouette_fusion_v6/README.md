# Skeleton + Silhouette Fusion V6

V6 is the next attempt after skeleton-only V4/V5 failed to improve Rank-1.

For the full architecture explanation, diagrams, loss flow, and metric
interpretation, see:

```text
designs/skeleton_silhouette_fusion_v6/MODEL_ARCHITECTURE_AND_FLOW.md
```

The hypothesis is simple:

> Hamilton skeletons capture topology and motion, but strict Rank-1 retrieval also needs body-shape/context information. Adding the original silhouette stream may recover identity cues that are lost in skeleton-only maps.

## Input representation

The fused cache stores four channels:

| Channel | Meaning |
|---|---|
| 0 | cleaned silhouette |
| 1 | Hamilton skeleton |
| 2 | blurred skeleton structure |
| 3 | temporal skeleton motion |

The shared dataset loader passes:

```text
silhouette = channel 0
topology   = channels 1, 2, 3
```

## Architecture

- Silhouette CNN stream
- Skeleton/topology CNN stream
- Per-frame gated fusion
- Temporal CNN
- Bi-GRU
- Attention + mean + max pooling
- Metric embedding
- Projection head
- Small auxiliary identity head during training
- Reconstruction decoder for skeleton + motion

At test time, the output is still an embedding vector. Rank-1 is computed through distance retrieval.

## Modal dataset requirement

V6 needs both datasets in Modal Volume:

```text
/data/CASIA_B_Hamilton_Skeleton
/data/GaitDatasetB-silh.zip
```

In the Modal Storage UI, the uploaded skeleton archive may appear as a file named `CASIA_B_Hamilton_Skeleton` without the `.zip` extension. That is okay; the config points to that exact file.

If `GaitDatasetB-silh.zip` is not uploaded yet, create/upload it from the local CASIA-B silhouette folder.

Example:

```bash
zip -r GaitDatasetB-silh.zip datasets/GaitDatasetB-silh
modal volume put gait-datasets-store GaitDatasetB-silh.zip /GaitDatasetB-silh.zip
```

## Run

```bash
modal deploy modal_app.py
python submit_modal.py run --design skeleton_silhouette_fusion_v6 --run fusion_rank1_002
```

## Important note

V6 uses:

```json
"eval_gallery_per_subject": 3
```

So report its Rank-1 as **3-gallery Rank-1**.

## Why V6 now monitors verification AUC

The first fused run produced this pattern:

- early warmup Rank-1 reached about `0.55`
- later contrastive verification became much stronger, with AUC about `0.89`
- early stopping stopped at epoch 16 because the early Rank-1 spike was not beaten

That means the fused representation was improving as a verification model, but
Rank-1 was too noisy to use as the only early-stop signal. The current config
therefore uses:

```json
"early_stopping_metric": "verification_auc",
"early_stopping_start_epoch": 12,
"early_stopping_patience": 24
```

Training still saves `best_rank1_model.pt` whenever Rank-1 improves. So the
main checkpoint follows the thesis verification objective, while the best
retrieval checkpoint is preserved separately.
