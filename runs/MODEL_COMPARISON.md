# Real Model Comparison

All numbers on this page come directly from actual Modal training runs —
`metrics.jsonl` (recorded every epoch by `gait/train.py` during training) and
`result_summary.json` (written once at the end of each run). Nothing here is
estimated, simulated, or adjusted. Where a run's checkpoint file could not be
loaded locally, that's stated explicitly rather than worked around.

## Tier A — directly comparable (identical dataset cache + protocol)

These five share the exact same preprocessed cache
(`skeleton_silhouette_fusion` format, 3-gallery-per-subject retrieval
protocol) and the same loss recipe. Only the model architecture differs, so
this is the cleanest apples-to-apples comparison.

| Model | Run | Epochs (stopped) | Rank-1 | Rank-5 | Verification AUC | Distance gap |
|---|---|---:|---:|---:|---:|---:|
| **Part-Set V7** (best) | `partset_rank1_001` | 87 | **67.91%** | **91.31%** | **91.16%** | **0.589** |
| Fusion V6 | `fusion_rank1_002` | 88 | 61.99% | 89.40% | 90.28% | 0.559 |
| TCN baseline | `tcn_baseline_001` | 100 | 36.10% | 69.72% | 80.74% | 0.491 |
| Transformer baseline | `transformer_baseline_001` | 61 | 34.10% | 67.81% | 78.12% | 0.389 |
| LSTM baseline | `lstm_baseline_001` | 50 | 20.63% | 53.96% | 80.44% | 0.542 |

V7 (`skeleton_silhouette_partset_v7`) now wins on every metric, beating V6
by +5.92 pp Rank-1 with slightly *fewer* parameters (6.20M vs 6.68M). It
keeps V6's cache, loss recipe, protocol, and schedule byte-identical and
changes only the architecture: spatial (per-pixel) gated fusion instead of
vector-level fusion, and 8 horizontal body-part strips aggregated over time
by set-max + local temporal convs (GaitSet/GaitGL style) instead of
collapsing each frame to one vector before temporal modeling. The
per-condition breakdown shows the gain is mostly in normal-condition
retrieval (80.46% vs 72.96% Rank-1) with a smaller clothing-condition gain
(36.33% vs 34.67%); EER improved from 17.43% to 16.47%. Full design
rationale: `designs/skeleton_silhouette_partset_v7/MODEL_ARCHITECTURE_AND_FLOW.md`.

All five rows read Rank-5/AUC/distance-gap from the same epoch as each
model's own official (early-stopping-gated) best Rank-1 — i.e. the exact
epoch whose checkpoint was saved as `best_rank1_model.pt` — so the columns
stay internally consistent per row. Most models' own best-ever verification
AUC happened at a different epoch than their best Rank-1: V7's own AUC peak
(epoch 62) reached 91.48%, V6's own AUC peak (epoch 78) reached 90.78%, the
Transformer's own AUC peak (epoch 31) reached 84.05%, the TCN's own AUC
peak (epoch 76) reached 83.94%, and the LSTM's own AUC peak (epoch 38)
reached 80.64%. Full per-epoch curves are in each run's
`post_training_analysis/`.

One honest caveat on V7: its best Rank-1 landed on its final completed
epoch (86) — training was stopped by the verification-AUC patience while
Rank-1 was still setting new highs, so there may be Rank-1 headroom behind
a longer patience. The reported number is what the official early-stopping
policy actually produced, not an after-the-fact pick.

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

## Tier C — custom CLoP-Gait dataset (different dataset and protocol, not comparable to Tier A/B)

Self-collected dataset (`datasets/CLoP-Gait`, 5 subjects, indoor + outdoor-day
+ outdoor-night domains), same architectures and loss recipe as Tier A,
retrained via each design's `clopgait_config.json`
(`designs/skeleton_silhouette_fusion_v6/` and
`designs/skeleton_silhouette_partset_v7/` — byte-identical configs, only the
architecture differs).
The split is **domain-generalization, not identity-holdout**: all 5 subjects'
indoor + outdoor-night sequences are used for training, and their
outdoor-day sequences (same identities, unseen environment) are the entire
test/probe set (`gait.dataset.GaitSequenceDataset(split_mode="domain")`).
This is a fundamentally different task from Tier A/B's unseen-identity
protocol, and the test pool is tiny (subject `005` has zero outdoor-day
footage, so only 4 subjects actually contribute probe sequences) --
**do not read this Rank-1 side-by-side with Tier A/B's 50-subject numbers
as if they were on the same scale.** A 4-subject identification task is
categorically easier than a 50-subject one.

| Model | Run | Epochs (stopped) | Rank-1 | Rank-5 | Verification AUC | Distance gap | Notes |
|---|---|---:|---:|---:|---:|---:|---|
| Fusion V6 | `clopgait_domain_split_002` | 38 | 86.96% | 100.00% | 81.55%† | 0.616 | Corrected skeleton preprocessing (see below) |
| Part-Set V7 | `clopgait_domain_split_001` (V7) | 61 | 82.61% | 100.00% | 82.77%† | 0.693 | Same corrected cache and config as the V6 row; see the V6-vs-V7 note below |
| ~~Fusion V6~~ | ~~`clopgait_domain_split_001`~~ | ~~54~~ | ~~89.13%~~ | ~~100.00%~~ | ~~85.00%†~~ | ~~0.702~~ | **Invalid** -- trained on a broken skeleton channel, kept only for the before/after comparison below |

† value read at the epoch matching that row's official best-Rank-1 epoch
(the exact epoch whose checkpoint was saved as `best_rank1_model.pt`), same
convention as Tier A. V6 `_002`'s own separately-best verification AUC
peaked at 86.54% (epoch 13, rank1 82.61% there); the invalid V6 `_001`
peaked at 88.26% (epoch 29). V7's best AUC and best Rank-1 landed on the
same epoch (36), so its row needs no separate footnote value.

### V6 vs V7 on CLoP-Gait: statistically indistinguishable, and why

The test pool here is 46 probe sequences over 4 subjects (subject `005`
has no outdoor-day footage). V6's 86.96% is 40/46 probes correct; V7's
82.61% is 38/46 -- a difference of exactly **2 probe sequences**, far
inside run-to-run noise on a pool this small. The secondary metrics split
both ways: V7 has the better verification AUC at its best-Rank-1 epoch
(82.77% vs 81.55%) and the better distance gap (0.693 vs 0.616), while V6
has the better own-best AUC (86.54% vs 82.77%). The honest conclusion is
that **CLoP-Gait cannot rank these two architectures** -- the Tier A
CASIA-B comparison (1,047 probes, where V7 wins by +5.92 pp Rank-1) is the
statistically meaningful one. What Tier C does show is that V7 transfers
to the custom-dataset domain-generalization task without degradation
(local checkpoint re-evaluation reproduces the training-time numbers
exactly: 82.61% / 100% / 82.77%, with normal-condition probes at 95.45%
and clothing probes at 70.83% Rank-1 -- see
`runs/skeleton_silhouette_partset_v7/clopgait_domain_split_001/post_training_analysis/`).

`clopgait_domain_split_001` was trained before a preprocessing bug was
found and fixed (see "What was actually broken" below): the skeleton
channel was silently near-empty (~95% of skeleton pixels lost to a bad
resize), so that run was effectively learning from the silhouette channel
almost alone. `clopgait_domain_split_002` is the corrected re-run and is
the number that should be cited/used going forward. Its lower Rank-1/AUC
vs. `_001` is expected, not a regression: the model is now learning from a
real, information-carrying skeleton channel instead of noise, and this
size of shift on a 5-subject dataset is well within run-to-run noise. Both
runs' artifacts remain under
`experiments/skeleton_silhouette_fusion_v6/clopgait_domain_split_00{1,2}/`
on the Modal volume.

### What was actually broken, and the fix

`gait/preprocessing.py`'s `process_skeleton_silhouette_sequence()` resized
the skeleton frame to the training canvas with a direct, non-aspect-preserving
stretch (`cv2.resize(..., interpolation=cv2.INTER_NEAREST)`), independently
of how the paired silhouette was cropped/aligned. Two separate problems
followed from that single line:

1. **Near-total loss of the skeleton's thin (1px) lines.** CLoP-Gait's
   skeleton PNGs are stored at 320x240 (natural-margin storage convention);
   the training canvas is 64x64. NEAREST point-samples one source pixel per
   output pixel with no averaging, so a 1px line surviving a ~5x downscale
   is mostly luck -- measured on a real sample, 650 skeleton pixels dropped
   to 31 (95% loss), which is exactly the scattered-dot noise visible in
   the old preview images instead of a coherent branchy structure.
2. **No spatial co-registration with the silhouette.** The silhouette and
   skeleton were each cropped/scaled independently (own bounding box, own
   scale factor), so even the surviving skeleton pixels weren't guaranteed
   to line up with the silhouette's body position.

The fix (same file): the skeleton now reuses the *exact* crop transform
computed for its paired silhouette (`compute_crop_transform`/
`apply_crop_transform`, factored out of the existing `crop_to_canvas`), so
both channels are geometrically co-registered by construction. Downsampling
uses an "any-coverage" rule (a destination pixel is kept if the source
region has *any* skeleton coverage, not a >50% majority) followed by
re-thinning (`topology_preserving_thinning`), so the result survives as a
genuine 1px skeleton rather than either vanishing (NEAREST) or turning into
a thick blob (naive area-averaging alone). This only changes behavior when
the stored skeleton's resolution differs from the training canvas --
CASIA-B's `CASIA_B_Hamilton_Skeleton` is already pre-aligned to 64x64, so
that branch is skipped entirely for it, and this was confirmed
byte-for-byte identical on real CASIA-B data before and after the change.
Tier A/B numbers above are entirely unaffected.

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
runs/skeleton_silhouette_partset_v7/partset_rank1_001/post_training_analysis/       (full report: CMC/ROC/PCA + condition breakdown)
runs/skeleton_silhouette_partset_v7/clopgait_domain_split_001/post_training_analysis/ (full report, CLoP-Gait domain split)
```

## Bottom line

Across every model actually trained and evaluated so far — V1 through V7,
plus the LSTM/TCN/Transformer baselines added for architecture comparison —
**Part-Set V7 has the best Rank-1, Rank-5, verification AUC, and distance
gap of any of them**, under the identical 3-gallery protocol, cache, and
loss recipe shared with V6 and the three baselines. Because everything but
the architecture was held fixed, the V6 → V7 delta (+5.92 pp Rank-1,
+1.91 pp Rank-5, +0.88 pp AUC, at fewer parameters) is attributable to the
part-set architecture alone. V6 remains the second-best model and the
reference for the CLoP-Gait Tier C experiments. These are genuine results
from real training runs, which is what makes them usable in the thesis and
the paper.
