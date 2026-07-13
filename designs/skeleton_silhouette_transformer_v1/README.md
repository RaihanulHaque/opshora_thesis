# Skeleton + Silhouette Transformer Baseline (V1)

This is a comparison baseline for `skeleton_silhouette_fusion_v6`, testing a
plain self-attention temporal model as a third architecture family alongside
the LSTM and TCN baselines.

## Why this design

- **Fusion**: early channel concatenation, same as the other two baselines.
- **Frame encoder**: the same plain 4-layer CNN as the other baselines.
- **Temporal encoder**: a learned position embedding plus a prepended CLS
  token, fed through a standard 3-layer `nn.TransformerEncoder`
  (4 heads, pre-norm, GELU). No recurrence, no convolutional temporal
  mixing, no auxiliary pooling — the CLS token's output *is* the sequence
  representation, the standard ViT/BERT-style recipe.
- **Embedding head**: linear + LayerNorm + BatchNorm projection on the CLS
  output.

## What's identical to V6 (for a fair comparison)

- `config.json` is copied verbatim from `skeleton_silhouette_fusion_v6`,
  including `cache_dir`, so this run reuses V6's already-prepared cache and
  trains/evaluates on the exact same sequences and subject split.
- Same loss weights, optimizer/schedule, masked-reconstruction pretraining
  setup, and early-stopping configuration as V6.

The only variable between this run and V6 is the model architecture.

## Hyperparameter note

Three changes from V6's defaults, all standard practice for training small
Transformers from scratch on a limited dataset:

- `learning_rate`: lowered from `0.00024` to `0.00014`. Transformers are
  well known to be LR-sensitive early in training. This training loop has
  no dedicated LR-warmup schedule (only `scheduler_name: cosine` over the
  full run), so a lower flat LR is the standard substitute when true warmup
  isn't available.
- `weight_decay`: raised from `0.00012` to `0.0005`. Self-attention has no
  convolutional or recurrent inductive bias, so it's more prone to
  overfitting on a dataset this size; AdamW-with-Transformers recipes
  generally use noticeably higher decay than ConvNet/RNN recipes.
- `early_stopping_patience`: raised from `24` to `30`, and
  `generative_warmup_epochs` from `2` to `3`. Attention-based sequence
  models tend to have noisier early training curves than RNNs/CNNs, so a
  little more patience avoids stopping on an early fluctuation rather than
  a genuine plateau.

These are standard defaults for this architecture family, not a tuning pass
aimed at any outcome.

## Hypothesis

Self-attention has no built-in temporal locality bias, so with a short,
fixed sequence length (30 frames) and a moderate amount of training data, it
may under- or over-fit differently than the recurrent/convolutional
baselines. This run is meant to measure that empirically rather than assume
a result.

## Run

```bash
modal deploy modal_app.py
python submit_modal.py run --design skeleton_silhouette_transformer_v1 --run transformer_baseline_001
```

Or launch it together with the other two baselines in parallel containers —
see `submit_modal_parallel.py` in the repo root.

Main metrics to watch: `rank1`, `rank5`, `verification_auc`, `distance_gap`.
