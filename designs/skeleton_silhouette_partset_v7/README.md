# Skeleton + Silhouette Part-Set Fusion V7

V7 is the successor to V6 (`skeleton_silhouette_fusion_v6`), the current
best model in `runs/MODEL_COMPARISON.md` (Tier A: Rank-1 61.99%, Rank-5
89.40%, verification AUC 90.28% on the 3-gallery, 50-unseen-subject
CASIA-B protocol).

## What stays identical to V6 (deliberately)

- The preprocessed cache: `/data/processed/casia_b_skeleton_silhouette_fusion_v6`
  (same 4-channel `skeleton_silhouette_fusion` format).
- The full loss recipe: SupCon contrastive + batch-hard triplet + small CE,
  alternated with masked skeleton+motion reconstruction (same
  `compute_reconstruction_loss`, same lambdas, same schedule).
- The retrieval protocol: 3-gallery Rank-1 over 50 unseen subjects, and the
  `verification_auc`-monitored early stopping with `start_epoch: 12`.
- All optimizer/schedule hyperparameters (`lr 0.00024`, cosine, AdamW).

Because *only the architecture changes*, V7 slots directly into the Tier A
apples-to-apples comparison against V6 and the LSTM/TCN/Transformer
baselines.

## The hypothesis

> V6 collapses each frame's feature map into a single 192-D vector before
> any temporal modeling happens. That throws away *where on the body* a
> shape or motion difference occurs — but part-localized cues (head shape,
> torso lean, stride geometry) are exactly what separates visually similar
> subjects. Keeping horizontal body-part features separate through temporal
> aggregation should recover Rank-1 headroom that V6's holistic frame
> vectors cannot express.

This is also the most consistent finding in the silhouette-gait literature
(GaitSet's horizontal strips + set pooling, GaitPart's part-level temporal
modeling, GaitGL's local temporal max): part-structured spatio-temporal
features dominate holistic ones on CASIA-B.

## Architecture changes vs. V6

| Aspect | V6 | V7 |
|---|---|---|
| Stream fusion | Per-frame 192-D *vector* gate after all pooling | Per-pixel, per-channel gate on 32x32 feature maps (spatially adaptive: skeleton trusted in limbs, silhouette on contour) |
| Backbone | Two full independent backbones to 8x8 | Two light stems to 32x32, then one shared trunk on the fused map to 16x16 |
| Spatial identity cues | Lost before temporal modeling (global+6-part avgpool -> one vector per frame) | Kept: 8 horizontal part strips (avg+max pooled) survive into temporal aggregation |
| Temporal aggregation | Conv1d + Bi-GRU on holistic frame vectors only | Per part: set-max over time + local temporal-conv branch (GaitSet/GaitGL style), each part embedded by its own linear head; *plus* V6's Conv1d + Bi-GRU global branch |
| Embedding input | attention+mean+max pooled GRU states + mean fused vector | 8x64 part embeddings + attention+mean pooled GRU states |
| Decoder | SkeletonDecoder on GRU states | Unchanged (fed by the surviving global GRU branch) |

Parameter count is slightly *below* V6 (6.20M vs 6.68M, measured): merging
the two full backbones into one shared trunk saves more than the part heads
add. The trunk keeps 16x16 resolution two conv blocks longer than V6, so
per-step compute is somewhat higher, but well within L4/A10 budget at batch
8x4x30 frames of 64x64.

## What V7 keeps from V6's hard-won lessons

- `early_stopping_metric: verification_auc` with `start_epoch: 12` (the
  early Rank-1 spike is noise — documented in V6's README and V5's Tier B
  caveat).
- `best_rank1_model.pt` still saved separately whenever Rank-1 improves.
- The reconstruction (generative) branch as a regularizer, unchanged.

## Run

```bash
modal deploy modal_app.py
python submit_modal.py run --design skeleton_silhouette_partset_v7 --run partset_rank1_001
```

Report its Rank-1 as **3-gallery Rank-1** (same as V6).
