# Skeleton + Silhouette Temporal-CNN Baseline (V1)

This is a comparison baseline for `skeleton_silhouette_fusion_v6`, testing a
purely convolutional (non-recurrent) temporal model — a common "and other
architectures" comparison point alongside the LSTM baseline.

## Why this design

Where `skeleton_silhouette_lstm_v1` tests recurrence, this design removes
recurrence entirely and asks whether dilated 1-D convolutions over the frame
sequence are sufficient on their own:

- **Fusion**: early channel concatenation, same as the LSTM baseline.
- **Frame encoder**: the same plain 4-layer CNN as the LSTM baseline.
- **Temporal encoder**: a 4-block residual Temporal Convolutional Network
  (TCN) with dilations 1, 2, 4, 8 — no GRU, no LSTM, no attention.
- **Pooling**: mean + max pooling over time, concatenated.
- **Embedding head**: linear + LayerNorm + BatchNorm projection.

## What's identical to V6 (for a fair comparison)

- `config.json` is copied verbatim from `skeleton_silhouette_fusion_v6`,
  including `cache_dir`, so this run reuses V6's already-prepared cache and
  trains/evaluates on the exact same sequences and subject split.
- Same loss weights, optimizer/schedule, masked-reconstruction pretraining
  setup, and early-stopping configuration as V6.

The only variable between this run and V6 is the model architecture.

## Hyperparameter note

`learning_rate` is raised from V6's `0.00024` to `0.00032`, and
`weight_decay` is lowered slightly from `0.00012` to `0.0001`. Purely
convolutional residual networks (no recurrence, no backprop-through-time)
typically tolerate — and often benefit from — a somewhat higher learning
rate than RNN-based models, since there's no BPTT instability to guard
against; the batch norm + residual connections already provide most of the
regularization a TCN needs, hence the slightly lower decay. This is a
standard TCN-family default, not a tuning pass aimed at any outcome.

## Hypothesis

TCNs are competitive with RNNs on many sequence tasks and are cheaper to
train (fully parallel over time, no recurrence). For gait retrieval, the
open question is whether a fixed dilated receptive field captures gait
periodicity as well as V6's GRU + attention pooling, given a fixed number
of frames per sequence. This run is meant to answer that empirically.

## Run

```bash
modal deploy modal_app.py
python submit_modal.py run --design skeleton_silhouette_tcn_v1 --run tcn_baseline_001
```

Or launch it together with the other two baselines in parallel containers —
see `submit_modal_parallel.py` in the repo root.

Main metrics to watch: `rank1`, `rank5`, `verification_auc`, `distance_gap`.
