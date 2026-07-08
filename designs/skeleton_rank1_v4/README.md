# Skeleton Rank-1 V4

This is the retrieval-focused follow-up to `skeleton_contrastive_v3`.

V3 was optimized mainly for blind verification AUC. It reached strong same/different separation, but Rank-1 retrieval stayed around `0.46`. V4 changes the architecture and objective to improve Rank-1 while still using embeddings at test time.

## What changed from V3

- Stronger frame encoder with squeeze-excitation attention.
- More horizontal body parts in the frame pooling stage.
- Multi-branch temporal CNN before the recurrent encoder.
- Bi-GRU plus a light Transformer temporal refinement block.
- Multi-pool sequence aggregation:
  - attention pooling,
  - mean pooling,
  - max pooling,
  - structure pooling.
- Auxiliary normalized identity head during training.
- Early stopping monitors `rank1`, not `verification_auc`.

## Why add an identity head?

The thesis endpoint can still be distance-based verification. However, Rank-1 retrieval is a stricter identification-style metric. To improve it, the embedding needs stronger class-separation pressure during training.

V4 therefore uses:

```json
"lambda_ce": 0.45
```

This means subject labels are used as an auxiliary training signal. At test time, the model still returns embeddings and compares distances.

## Expected behavior

Compared with V3:

- Rank-1 should improve.
- Rank-5 should improve.
- Verification AUC may stay similar or slightly change.
- Reconstruction previews may be less central because V4 gives more training weight to retrieval.

Target:

```text
Rank-1 >= 0.70
```

This is a goal, not a guarantee. If V4 does not reach it, the next likely limitation is cross-view mismatch in the skeleton dataset/evaluation protocol, not simply model capacity.

## Run

```bash
modal deploy modal_app.py
python submit_modal.py run --design skeleton_rank1_v4 --run rank1_001
```

Main metrics to watch:

- `rank1`
- `rank5`
- `verification_auc`
- `same_distance`
- `different_distance`
- `distance_gap`

