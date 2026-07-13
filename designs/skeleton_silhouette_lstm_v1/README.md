# Skeleton + Silhouette LSTM Baseline (V1)

This is a comparison baseline for `skeleton_silhouette_fusion_v6`, built to
answer the thesis committee's request for a classical recurrent-network
comparison point (LSTM) alongside the fusion architecture.

## Why this design

V6 uses a learned gated fusion between the silhouette and skeleton streams,
a multi-branch dilated temporal CNN, a bidirectional GRU, a Transformer
refinement block, and attention + mean + max + structure pooling.

This baseline strips that down to the simplest architecture that is still a
fair, legitimate comparison:

- **Fusion**: early channel concatenation (silhouette + topology stacked as
  4 input channels), not a learned gate.
- **Frame encoder**: a plain 4-layer CNN, no squeeze-excitation, no
  part-based pooling.
- **Temporal encoder**: a 2-layer bidirectional LSTM. No temporal CNN, no
  Transformer refinement.
- **Pooling**: mean pooling over time only. No attention pooling, no
  multi-pool concatenation.
- **Embedding head**: a single linear + LayerNorm + BatchNorm projection.

## What's identical to V6 (for a fair comparison)

- `config.json` is copied verbatim from `skeleton_silhouette_fusion_v6`:
  same `dataset_path`, `silhouette_dataset_path`, and — importantly — the
  same `cache_dir`. Training reuses V6's already-prepared cache, so both
  models train/evaluate on byte-identical preprocessed sequences and the
  same train/test subject split.
- Same loss weights (`lambda_contrastive`, `lambda_triplet`, `lambda_ce`,
  reconstruction losses), same optimizer/schedule, same masked-reconstruction
  pretraining setup, same early-stopping metric and patience.

The only variable between this run and V6 is the model architecture.

## Hyperparameter note

`weight_decay` is nudged up slightly from V6's `0.00012` to `0.00016`. This
model has more parameters than V6 (~4.9M vs. V6's fusion network) despite
being architecturally simpler, so a small amount of extra regularization is
standard practice to keep it from overfitting the same amount of training
data. Learning rate, schedule, and everything else is unchanged from V6 —
this is a minor, standard nudge, not a tuning pass aimed at any outcome.

## Hypothesis

A single-stream, mean-pooled recurrent baseline with early fusion is
expected to be a meaningfully weaker retrieval model than V6: it has no
mechanism to weight informative frames (no attention pooling), no explicit
cross-modal gating between silhouette and skeleton, and less temporal
receptive field than V6's dilated-CNN + GRU + Transformer stack. Whether the
gap is large or small is an empirical question this run is meant to answer,
not a conclusion decided in advance.

## Run

```bash
modal deploy modal_app.py
python submit_modal.py run --design skeleton_silhouette_lstm_v1 --run lstm_baseline_001
```

Or launch it together with the other two baselines in parallel containers —
see `submit_modal_parallel.py` in the repo root.

Main metrics to watch: `rank1`, `rank5`, `verification_auc`, `distance_gap`.
