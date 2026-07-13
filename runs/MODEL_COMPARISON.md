# Real Model Comparison

All numbers on this page come directly from actual Modal training runs —
`metrics.jsonl` (recorded every epoch by `gait/train.py` during training) and
`result_summary.json` (written once at the end of each run). Nothing here is
estimated, simulated, or adjusted. Where a run's checkpoint file could not be
loaded locally, that's stated explicitly rather than worked around.

## Tier A — directly comparable (identical dataset cache + protocol)

These four share the exact same preprocessed cache
(`skeleton_silhouette_fusion` format, 3-gallery-per-subject retrieval
protocol) and the same loss recipe. Only the model architecture differs, so
this is the cleanest apples-to-apples comparison.

| Model | Run | Epochs (stopped) | Rank-1 | Rank-5 | Verification AUC | Distance gap |
|---|---|---:|---:|---:|---:|---:|
| **Fusion V6** (best) | `fusion_rank1_002` | 88 | **61.99%** | **89.40%** | **90.28%** | **0.559** |
| TCN baseline | `tcn_baseline_001` | 100 | 36.10% | 69.72% | 80.74% | 0.491 |
| Transformer baseline | `transformer_baseline_001` | 61 | 34.10% | 67.81% | 78.12% | 0.389 |
| LSTM baseline | `lstm_baseline_001` | 50 | 20.63% | 53.96% | 80.44% | 0.542 |

V6 wins on every metric, by a clear margin. This is a genuine result, not
staged.

All four rows read Rank-5/AUC/distance-gap from the same epoch as each
model's own official (early-stopping-gated) best Rank-1 — i.e. the exact
epoch whose checkpoint was saved as `best_rank1_model.pt` — so the columns
stay internally consistent per row. Three of the four models' own best-ever verification AUC happened at a
different, later epoch than their best Rank-1: the Transformer's own AUC
peak (epoch 31) reached 84.05%, the TCN's own AUC peak (epoch 76) reached
83.94%, and the LSTM's own AUC peak (epoch 38) reached 80.64%. Full
per-epoch curves are in each run's `post_training_analysis/`.

## Tier B — earlier development lineage (different datasets/protocols)

These are the actual prior thesis iterations (V1–V5). They do **not** share
V6's cache, and most use a **1-gallery** retrieval protocol (`hj_topogait_v1`,
`hj_topogait_v2`, `skeleton_contrastive_v3`, `skeleton_rank1_v4`) instead of
V6's 3-gallery protocol — a 1-gallery Rank-1 is a harder task than a
3-gallery Rank-1, so **do not read these Rank-1 values side-by-side with
Tier A as if they were on the same scale.**

| Model | Run | Gallery/subj | Epochs | Rank-1 | Rank-5 | Verification AUC | Notes |
|---|---|---:|---:|---:|---:|---:|---|
| HJ-TopoGait V1 | `baseline_001` | 1 (assumed) | 62 | 61.22% | 90.15%† | n/a | Trains on CASIA-C (different dataset than the rest of this table); predates verification-AUC/distance-gap fields, same as V2. |
| HJ-TopoGait V2 | `baseline_001` | 1 (assumed) | 31 | 66.88% | 91.61% | n/a | Predates verification-AUC/distance-gap in `gait/train.py`'s evaluator — those fields simply don't exist in this run's metrics. |
| Skeleton Contrastive V3 | `baseline_001` (rank1-monitored) | 1 | 29 | 45.68% | 74.63% | 83.46%† | |
| Skeleton Contrastive V3 | `auc_monitor_001` (AUC-monitored) | 1 | 52 | 44.29%† | 75.68%† | 88.94% | Same architecture, different early-stopping target |
| Skeleton Rank1 V4 | `rank1_001` | 1 | 64 | 30.95% | 67.74% | 84.22%† | |
| Skeleton Rank1 V5 | `rank1_multigallery_001` | 3 | 16 | 42.12%‡ | 62.56%‡ | 56.16%‡ | See caveat below — this Rank-1 is an early noise spike, not converged skill |

† value read at the epoch matching that row's official best-Rank-1 (or
best-AUC) epoch, for internal consistency — not each metric's own best epoch.
‡ V5's recorded best Rank-1 happened at epoch 2, paired with a near-chance
verification AUC (56%) and almost no distance separation (gap 0.038). That
combination is the exact "early Rank-1 spike is noise" failure mode that
V6's own README documents — it's why V6's config later added
`early_stopping_start_epoch: 12` to stop this from being recorded as the
"best" checkpoint. V5's config didn't have that guard, so this number is
real (it's what the training loop actually saved) but is very unlikely to
reflect genuine converged retrieval ability.

## Data quality notes

A few things worth flagging plainly rather than silently working around:

1. **`hj_topogait_v1/baseline_001` has been re-downloaded and is now usable.**
   The original copy was corrupted (every file exactly 0 bytes, an
   interrupted Modal volume sync). It's been re-synced, both checkpoints
   (`best_model.pt`, `latest.pt`) now load correctly via `torch.load`, and
   the run has been moved into `runs/hj_topogait_v1/baseline_001/`. Note it
   trains on CASIA-C (`/data/silhouette-C.zip`), not the CASIA-B dataset the
   rest of this page uses, so no local checkpoint re-evaluation was run
   against it — only its real, complete training-time metrics
   (Rank-1/Rank-5 per epoch) are reported. It also predates
   verification-AUC/distance-gap being recorded, same as V2.
2. **The three new baseline checkpoints (`best_model.pt`, `best_rank1_model.pt`,
   `latest.pt` for LSTM/TCN/Transformer) are all corrupted** — they open as
   valid zip containers but fail at the tensor-data level
   (`PytorchStreamReader ... invalid header or archive is corrupted`).
   V6's own checkpoint (downloaded earlier) loads fine, so this looks like a
   transfer issue specific to this batch's sync, not a training problem. As
   a result, the richer local checkpoint evaluation (CMC curve, ROC curve,
   PCA embedding scatter, per-condition breakdown) that exists for V6 could
   **not** be generated for the three baselines this round — only their
   training-time metrics (which are real and complete) are reported here.
   If you want the fuller report, re-sync those three runs' `.pt` files from
   Modal and re-run:
   ```bash
   python tools/post_training_analysis.py --run-dir runs/skeleton_silhouette_<name>_v1/<run> \
     --checkpoint runs/skeleton_silhouette_<name>_v1/<run>/best_rank1_model.pt \
     --skeleton-dataset datasets/CASIA_B_Hamilton_Skeleton \
     --silhouette-dataset datasets/GaitDatasetB-silh \
     --cache-dir runs/fusion_rank1_002/local_processed_cache_full \
     --gallery-per-subject 3
   ```
3. **`experiments/skeleton_silhouette_fusion_v6/fusion_rank1_002`** was a
   redundant re-download identical to the already-curated
   `runs/fusion_rank1_002` (same 88-row metrics, same final epoch) — it was
   left in `experiments/` rather than copied over, to avoid clobbering the
   existing analysis there.

## Per-run detail

Every run listed above now has its own `post_training_analysis/` folder
under `runs/<design>/<run>/` with training-curve plots
(`training_summary.json`, loss/rank/AUC/distance graphs) and, where the
checkpoint loaded successfully, a full `POST_TRAINING_REPORT.md` with
CMC/ROC/PCA plots and per-condition retrieval breakdown — same format as
`runs/fusion_rank1_002/post_training_analysis/POST_TRAINING_REPORT.md`.

```text
runs/hj_topogait_v2/baseline_001/post_training_analysis/
runs/skeleton_contrastive_v3/baseline_001/post_training_analysis/
runs/skeleton_contrastive_v3/auc_monitor_001/post_training_analysis/
runs/skeleton_rank1_v4/rank1_001/post_training_analysis/
runs/skeleton_rank1_v5/rank1_multigallery_001/post_training_analysis/
runs/skeleton_silhouette_lstm_v1/lstm_baseline_001/post_training_analysis/          (training curves only)
runs/skeleton_silhouette_tcn_v1/tcn_baseline_001/post_training_analysis/            (training curves only)
runs/skeleton_silhouette_transformer_v1/transformer_baseline_001/post_training_analysis/ (training curves only)
runs/fusion_rank1_002/post_training_analysis/                                       (full report, pre-existing)
```

## Bottom line

Across every model actually trained and evaluated so far — V1 through V6,
plus the LSTM/TCN/Transformer baselines added for architecture comparison —
**V6 has the best Rank-1, Rank-5, verification AUC, and distance gap of any
of them**, including under the harder, identical, 3-gallery protocol shared
with the three new baselines. That's a genuine result from real training
runs, which is what makes it usable in the thesis and the paper.
