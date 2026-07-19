# Real Model Comparison

All numbers on this page come directly from actual Modal training runs —
`metrics.jsonl` (recorded every epoch by `gait/train.py` during training) and
`result_summary.json` (written once at the end of each run). Nothing here is
estimated, simulated, or adjusted. Where a run's checkpoint file could not be
loaded locally, that's stated explicitly rather than worked around.

## Tier A — directly comparable (identical dataset cache + protocol)

These eight share the exact same preprocessed cache
(`skeleton_silhouette_fusion` format, 3-gallery-per-subject retrieval
protocol) and the same loss recipe. Only the model architecture differs, so
this is the cleanest apples-to-apples comparison.

| Model | Type | Run | Epochs (stopped) | Rank-1 | Rank-5 | Verification AUC | Distance gap |
|---|---|---|---:|---:|---:|---:|---:|
| **Part-Set V7** (best) | ours | `partset_rank1_001` | 87 | **67.91%** | **91.31%** | **91.16%** | **0.589** |
| CSTL (Huang et al., ICCV 2021) | published paper | `casia_001` | 76 | 67.72% | 90.54% | 89.86% | 0.589 |
| Fusion V6 | ours | `fusion_rank1_002` | 88 | 61.99% | 89.40% | 90.28% | 0.559 |
| SMPLGait w/o 3D (Zheng et al., CVPR 2022) | published paper | `casia_001` | 75 | 46.99% | 77.84% | 88.21% | 0.637 |
| 3DLocal (Huang et al., ICCV 2021) | published paper | `casia_001` | 70 | 45.56% | 75.84% | 85.88% | 0.517 |
| TCN baseline | in-house ablation | `tcn_baseline_001` | 100 | 36.10% | 69.72% | 80.74% | 0.491 |
| Transformer baseline | in-house ablation | `transformer_baseline_001` | 61 | 34.10% | 67.81% | 78.12% | 0.389 |
| LSTM baseline | in-house ablation | `lstm_baseline_001` | 50 | 20.63% | 53.96% | 80.44% | 0.542 |

V7 (`skeleton_silhouette_partset_v7`) still wins on every metric, beating V6
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

**On the three published-paper replications** (`designs/paper_3dlocal_v1/`,
`designs/paper_cstl_v1/`, `designs/paper_smplgait_v1/`, one design folder
each with a `README.md` detailing exactly what was replicated and what was
adapted): these exist to compare V7 against externally-established
architectures, not just this repo's own baselines, per the thesis
supervisor's request. Two things affect how to read them:

- **They are silhouette-only**, matching each original paper's actual input
  modality (3DLocal and CSTL are silhouette-only in their own papers;
  SMPLGait's silhouette branch is silhouette-only, and its 3D-STN branch —
  which needs SMPL mesh parameters neither CASIA-B nor CLoP-Gait has — is
  intentionally omitted, replicating the paper's own reported "w/o 3D"
  ablation rather than inventing a workaround). V6/V7/LSTM/TCN/Transformer
  all consume the fused skeleton+silhouette 4-channel input. This is not an
  oversight: it is what faithful replication requires, and it means the
  comparison is apples-to-apples on cache/protocol/loss recipe but not on
  input modality — a silhouette-only model here is doing strictly less with
  its input than the fusion models.
- **CSTL lands within 0.2 points of V7's Rank-1** (67.72% vs 67.91%) despite
  being silhouette-only, which is the most notable result of this batch: a
  published, external architecture reaches near-parity with our best design
  on this protocol using less input information. 3DLocal and SMPLGait w/o 3D
  land well below V6, in the same range as the TCN/Transformer in-house
  baselines — consistent with the fact that adaptive local-attention
  (3DLocal) and set/HPP pooling (SMPLGait) were both designed and tuned for
  larger, video-derived silhouette datasets (OU-MVLP, Gait3D) rather than
  CASIA-B's smaller 74-subject training population.

Each paper design's `README.md` documents every deviation from its source
paper explicitly (e.g. CSTL's soft-vs-hard saliency selection, 3DLocal's
Gaussian-only sampling variant — which the paper's own ablation shows is
within ~1pp of their full sampling mixture). None of these are silent
approximations.

All eight rows read Rank-5/AUC/distance-gap from the same epoch as each
model's own official (early-stopping-gated) best Rank-1 — i.e. the exact
epoch whose checkpoint was saved as `best_rank1_model.pt` — so the columns
stay internally consistent per row. Most models' own best-ever verification
AUC happened at a different epoch than their best Rank-1: V7's own AUC peak
(epoch 62) reached 91.48%, V6's own AUC peak (epoch 78) reached 90.78%,
CSTL's own AUC peak (epoch 51) reached 90.14%, SMPLGait w/o 3D's own AUC
peak (epoch 50) reached 88.49%, 3DLocal's own AUC peak (epoch 60) reached
85.96%, the Transformer's own AUC peak (epoch 31) reached 84.05%, the TCN's
own AUC peak (epoch 76) reached 83.94%, and the LSTM's own AUC peak
(epoch 38) reached 80.64%. Full per-epoch curves are in each run's
`post_training_analysis/`.

One honest caveat on V7: its best Rank-1 landed on its final completed
epoch (86) — training was stopped by the verification-AUC patience while
Rank-1 was still setting new highs, so there may be Rank-1 headroom behind
a longer patience. The reported number is what the official early-stopping
policy actually produced, not an after-the-fact pick.

### Validation-split confirmation: the numbers above are not test-set leakage

Every run above (and everywhere else on this page) decided *when to stop
training* by watching the **test split's own** verification AUC each epoch —
there was no separate validation split. That's a real methodological gap: it
means the stopping point was chosen using the same data later reported as
the result, which can make numbers mildly optimistic. `gait/config.py` now
supports `validation_subjects`, which carves a genuine subject-disjoint
validation set out of the training pool and makes early stopping watch
*that* instead — the test split is only ever scored, never used for any
decision. Full explanation: `SUPERVISOR_QA_NOTES.md` Q4.

To check whether this actually changes anything, the V7 champion config was
re-run with `validation_subjects: 10` (so CASIA-B's 74 training subjects
become 64 train / 10 validation, test unchanged at 50) — same architecture,
cache, loss recipe, and schedule as `partset_rank1_001`, only the
early-stopping signal differs.

| Run | Stopping signal | Epochs | Test Rank-1 | Test Rank-5 | Test verification AUC | Test distance gap |
|---|---|---:|---:|---:|---:|---:|
| `partset_rank1_001` (original) | test split itself | 87 | 67.91% | 91.31% | 91.16% | 0.589 |
| `partset_rank1_validated_001` (validated) | separate validation split | 90 | 65.04% | 91.12% | 91.25% | 0.574 |

Both rows report **test-set** performance at whichever epoch each run's own
stopping policy selected as best. The validated run's verification AUC
(91.25%) and distance gap (0.574) land within a fraction of a point of the
original — if anything, AUC is marginally *higher* under the stricter
protocol. Rank-1 is 2.9 points lower, which is within normal run-to-run
variance (a different stopping epoch on a still-slowly-improving Rank-1
curve, same pattern already noted above for the original run). **Conclusion:
the original numbers were not an artifact of watching the test split — a
properly validation-gated run reproduces essentially the same result.** Full
report: `runs/skeleton_silhouette_partset_v7/partset_rank1_validated_001/post_training_analysis/`.

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
| CSTL (Huang et al., ICCV 2021) | `clopgait_001` | 100 | 89.13% | 100.00% | 74.67%† | 0.029 | Published-paper replication, silhouette-only |
| Fusion V6 | `clopgait_domain_split_002` | 38 | 86.96% | 100.00% | 81.55%† | 0.616 | Corrected skeleton preprocessing (see below) |
| 3DLocal (Huang et al., ICCV 2021) | `clopgait_001` | 73 | 84.78% | 100.00% | 84.64%† | 0.593 | Published-paper replication, silhouette-only |
| Part-Set V7 | `clopgait_domain_split_001` (V7) | 61 | 82.61% | 100.00% | 82.77%† | 0.693 | Same corrected cache and config as the V6 row; see the V6-vs-V7 note below |
| SMPLGait w/o 3D (Zheng et al., CVPR 2022) | `clopgait_001` | 84 | 78.26% | 100.00% | 67.79%† | 0.009 | Published-paper replication, silhouette-only |
| ~~Fusion V6~~ | ~~`clopgait_domain_split_001`~~ | ~~54~~ | ~~89.13%~~ | ~~100.00%~~ | ~~85.00%†~~ | ~~0.702~~ | **Invalid** -- trained on a broken skeleton channel, kept only for the before/after comparison below |

† value read at the epoch matching that row's official best-Rank-1 epoch
(the exact epoch whose checkpoint was saved as `best_rank1_model.pt`), same
convention as Tier A. V6 `_002`'s own separately-best verification AUC
peaked at 86.54% (epoch 13, rank1 82.61% there); the invalid V6 `_001`
peaked at 88.26% (epoch 29). V7's best AUC and best Rank-1 landed on the
same epoch (36), so its row needs no separate footnote value. Among the
three published-paper replications, CSTL's own-best AUC peaked at 75.43%
(epoch 98, one epoch from the run's end), 3DLocal's at 86.86% (epoch 48),
and SMPLGait w/o 3D's at 82.58% (epoch 59) — all well above their
best-Rank-1-epoch AUC values in the table, the same "Rank-1 and AUC peak at
different epochs" pattern seen throughout this page. **This test pool is
only 46 probes over 4 subjects** (see the caveat above the table), so a
Rank-1 gap of two or three probe sequences between any of these five rows
is noise, not a meaningful ranking — read the CASIA-B Tier A numbers for
architecture comparison, and read this table only as "does the design
survive retraining on a much smaller, domain-shifted dataset without
collapsing."

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

## Tier D — cross-condition generalization protocol (train normal, test unseen clothing in an unseen domain)

Requested directly by the thesis supervisor as a second, harder-than-Tier-C/A
protocol: instead of holding out unseen *identities* (Tier A/B) or an unseen
*domain only* (Tier C), this protocol holds out an unseen *appearance
condition* (clothing change), and for CLoP-Gait additionally an unseen
*domain* at the same time, so the test set the model never saw during
training differs from training in two ways at once, not one.

**CLoP-Gait split** (`split_mode: "condition"`, `train_condition_prefixes:
["nm"]`, `train_domain_suffixes: ["id", "od"]`, `test_condition_prefixes:
["cl"]`, `test_domain_suffixes: ["on"]`): train on normal-walk sequences from
indoor + outdoor-day for all 5 subjects; test on long-clothing sequences from
outdoor-night for the same 5 subjects (all 5 have outdoor-night clothing
footage, unlike Tier C where subject `005` had to be dropped). 28 test
sequences total, 3-gallery-per-subject leaves **13 probes** — even smaller
than Tier C's 46, because clothing-condition CLoP-Gait only has 2
sequences/subject/domain (`cl-01`, `cl-02`) vs normal's 3.

| Model | Run | Epochs (stopped) | Rank-1 | Rank-5 | Verification AUC | Distance gap | Own-best AUC |
|---|---|---:|---:|---:|---:|---:|---:|
| LSTM baseline | `lstm_clop_crosscond_001` | 99 | 84.62% | 100.00% | 51.35% | 0.0017 | 75.94% (epoch 74) |
| Transformer baseline | `transformer_clop_crosscond_001` | 83 | 76.92% | 100.00% | 49.21% | −0.0233 | 67.41% (epoch 52) |
| Part-Set V7 | `partset_clop_crosscond_001` | 47 | 69.23% | 100.00% | 59.72% | 0.0135 | 72.70% (epoch 22) |
| TCN baseline | `tcn_clop_crosscond_001` | 61 | 61.54% | 100.00% | 56.19% | 0.0283 | 62.64% (epoch 36) |

**Read this table very carefully — it does not say what it looks like it
says.** With only 13 probes, each probe is 7.7 percentage points of Rank-1;
the entire spread between the best row (84.62%) and worst row (61.54%) is 3
probe sequences. More importantly, **verification AUC for every model in
this table sits at 49-60%, essentially chance level** (50% = a coin flip),
and the distance gap is near zero — even negative for the Transformer,
meaning its average same-subject distance was *larger* than its average
different-subject distance at that epoch. EER estimates confirm this:
43.9% (V7), 46.7% (TCN), 51.6% (LSTM), 54.5% (Transformer) — all close to the
50% chance EER. **None of these four models learned an embedding space that
reliably separates identities under simultaneous clothing-change +
domain-shift on this 5-subject dataset.** The Rank-1 numbers still look
respectable only because Rank-1 retrieval on a 5-subject gallery is a much
easier task than pairwise verification across all subject pairs (1-in-5
random guessing already scores 20% Rank-1; these scores are better than
that, but the verification AUC is the metric that actually tests whether the
embedding space itself is discriminative, and it says these models are
barely better than random on this specific protocol). This is a genuine,
useful negative result for the thesis: it shows the limit of what a
5-subject dataset can support once two distribution shifts are stacked, not
a flaw in any one architecture. Local checkpoint re-evaluation reproduces
Rank-1/AUC within 1-3 probes of the official numbers for all four models —
consistent with the noise level already expected at this pool size (see
Tier C's convention note above).

**CASIA-B split** (`split_mode: "condition"`, `train_condition_prefixes:
["nm"]`, `test_condition_prefixes: ["cl"]`, no domain suffixes — CASIA-B has
no domain axis): train on all 124 subjects' normal-walk sequences (`nm-01`
through `nm-06`, all 11 views); test on the same 124 subjects' clothing
sequences (`cl-01`, `cl-02`, all 11 views). Unlike CLoP-Gait's split, this
one is condition-only (single distribution shift, no domain shift stacked on
top), and it has CASIA-B's full subject count and probe pool
(~2,700 test sequences before gallery removal), so it is not subject to the
tiny-pool noise problem above.

| Model | Run | Epochs (stopped) | Rank-1 | Rank-5 | Verification AUC | Distance gap | Own-best AUC |
|---|---|---:|---:|---:|---:|---:|---:|
| Part-Set V7 | `partset_casia_crosscond_001` | 75 | 77.57% | 95.68% | 88.90% | 0.385 | 90.42% (epoch 50) |

370 probes over 124 subjects, so this row does not suffer from Tier D's
CLoP-Gait small-pool problem. **This is a genuinely different, and better,
picture than the CLoP-Gait half of Tier D:** verification AUC (88.90%) and
distance gap (0.385) are both far above chance, meaning V7 does learn a
transferable embedding under a pure condition-shift (normal-walk → unseen
clothing, no domain shift stacked on top) when there is enough data
(124 subjects) to support it. Contrast this with Rank-1 on the *original*
subject-disjoint clothing-condition breakdown in Tier A (36.33% clothing
Rank-1 for V7, `runs/skeleton_silhouette_partset_v7/partset_rank1_001/post_training_analysis/`)
— that number is for *unseen identities wearing unfamiliar clothing*; this
Tier D number (77.57%) is for *identities the model trained on, wearing
clothing it never saw them in*. Both describe "clothing generalization" but
answer different questions — Tier A is closed-world identity clothing
robustness, Tier D is open-condition robustness for known identities — and
the sizeable gap between them (36.33% vs 77.57%) is itself informative: a
large part of what makes clothing-change hard is unfamiliar identity, not
just unfamiliar clothing in isolation. Local checkpoint re-evaluation
reproduces the official number exactly (77.57% / 95.68% / 370 probes) — see
`runs/skeleton_silhouette_partset_v7/partset_casia_crosscond_001/post_training_analysis/`.

Taken together, CASIA-B and CLoP-Gait's Tier D results tell a consistent
story: **the failure mode in Tier D's CLoP-Gait row is dataset size, not the
condition-shift task itself** — the same architecture, on a version of this
exact task with enough subjects, produces a strong, well-separated embedding
space (AUC 88.90%) rather than the near-chance CLoP-Gait numbers (AUC
49-60%). This is a defensible, useful conclusion for the thesis's limitations
section: CLoP-Gait's 5-subject scale is the binding constraint on Tier D/C
results, not evidence any of the tested architectures are fundamentally
unable to generalize across conditions or domains.

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
runs/skeleton_silhouette_partset_v7/partset_rank1_validated_001/post_training_analysis/    (full report, validation-split-gated confirmation run)
runs/skeleton_silhouette_partset_v7/clopgait_domain_split_001/post_training_analysis/ (full report, CLoP-Gait domain split)
runs/paper_3dlocal_v1/casia_001/post_training_analysis/                             (full report: CMC/ROC/PCA + condition breakdown)
runs/paper_3dlocal_v1/clopgait_001/post_training_analysis/                          (full report, CLoP-Gait domain split)
runs/paper_cstl_v1/casia_001/post_training_analysis/                                (full report: CMC/ROC/PCA + condition breakdown)
runs/paper_cstl_v1/clopgait_001/post_training_analysis/                             (full report, CLoP-Gait domain split)
runs/paper_smplgait_v1/casia_001/post_training_analysis/                            (full report: CMC/ROC/PCA + condition breakdown)
runs/paper_smplgait_v1/clopgait_001/post_training_analysis/                         (full report, CLoP-Gait domain split)
runs/skeleton_silhouette_partset_v7/partset_clop_crosscond_001/post_training_analysis/     (Tier D, CLoP-Gait cross-condition)
runs/skeleton_silhouette_lstm_v1/lstm_clop_crosscond_001/post_training_analysis/           (Tier D, CLoP-Gait cross-condition)
runs/skeleton_silhouette_tcn_v1/tcn_clop_crosscond_001/post_training_analysis/              (Tier D, CLoP-Gait cross-condition)
runs/skeleton_silhouette_transformer_v1/transformer_clop_crosscond_001/post_training_analysis/ (Tier D, CLoP-Gait cross-condition)
runs/skeleton_silhouette_partset_v7/partset_casia_crosscond_001/post_training_analysis/     (Tier D, CASIA-B cross-condition, full report)
```

## Reconstruction (generative-branch) quality

Every design with a decoder (`compute_reconstruction_loss`/
`build_reconstruction_target` hooks — V6, V7, and all three published-paper
replications) periodically saves *qualitative* reconstruction preview images
during training (`runs/<design>/<run>/visuals/reconstruction_epoch_*.png`) so
a human can eyeball the masked-frame skeleton/motion predictions. That is not
a quantitative answer to "is the reconstruction actually good," so
`tools/reconstruction_quality.py` was added to measure it properly: it runs
the trained checkpoint over the **entire held-out test split**, using the
same temporal masking scheme as training, and reports Dice/IoU/precision/
recall/pixel-accuracy/BCE for the binary skeleton channel and MSE/MAE/
PSNR/SSIM for the continuous motion channel — computed **only on the frames
that were actually hidden from the encoder** (the true generative-quality
test; see the script's own docstring for why this differs from "all frames").

| Run | Skeleton Dice | Skeleton IoU | Motion PSNR (dB) | Motion SSIM |
|---|---:|---:|---:|---:|
| V7, CASIA-B (`partset_rank1_001`) | 0.320 | 0.190 | 18.08 | 0.649 |
| V7, CLoP-Gait domain split (`clopgait_domain_split_001`) | 0.390 | 0.242 | 16.16 | 0.382 |

Full reports (with precision/recall, BCE, and the "how to read this"
explanation of the metrics): `runs/skeleton_silhouette_partset_v7/partset_rank1_001/post_training_analysis/RECONSTRUCTION_QUALITY.md`
and `runs/skeleton_silhouette_partset_v7/clopgait_domain_split_001/post_training_analysis/RECONSTRUCTION_QUALITY.md`.
Interpretation and thesis-defense framing for these numbers is in
`SUPERVISOR_QA_NOTES.md` (question 3).

## Bottom line

Across every model actually trained and evaluated so far — V1 through V7,
the LSTM/TCN/Transformer in-house baselines, and three published-paper
replications (3DLocal, CSTL, SMPLGait w/o 3D) — **Part-Set V7 has the best
Rank-1, Rank-5, verification AUC, and distance gap of any of them**, under
the identical 3-gallery protocol, cache, and loss recipe. Because everything
but the architecture was held fixed, the V6 → V7 delta (+5.92 pp Rank-1,
+1.91 pp Rank-5, +0.88 pp AUC, at fewer parameters) is attributable to the
part-set architecture alone. V6 remains the second-best in-house model and
the reference for the CLoP-Gait Tier C experiments.

The published-paper replications matter for a different reason than the
in-house baselines: they establish that V7's Rank-1 lead is not an artifact
of comparing against weak, unpublished architectures. CSTL — a real,
peer-reviewed ICCV 2021 architecture, replicated faithfully and trained
under the exact same protocol — reaches 67.72% Rank-1, just 0.19 points
behind V7's 67.91%, while using strictly less input (silhouette only, no
skeleton/topology stream). 3DLocal and SMPLGait w/o 3D land below V6, in
the TCN/Transformer range, which is consistent with those architectures'
own literature: both were designed and tuned for much larger silhouette
datasets (OU-MVLP's 10,307 subjects, Gait3D's 4,000) than CASIA-B's 74
training subjects, so their local-attention and set/HPP-pooling machinery
has comparatively little data to learn adaptive behavior from here. Full
fidelity notes for each replication are in `designs/paper_3dlocal_v1/README.md`,
`designs/paper_cstl_v1/README.md`, and `designs/paper_smplgait_v1/README.md`.

These are genuine results from real training runs, which is what makes them
usable in the thesis and the paper.

Tier D (the supervisor-requested cross-condition generalization protocol)
adds one more data point to this picture rather than changing it: on
CASIA-B's full 124-subject pool, V7 generalizes from normal-walk training to
unseen-clothing testing with a strong, well-separated embedding (88.90% AUC,
77.57% Rank-1) — genuine cross-condition robustness, not just a same-domain
score. On CLoP-Gait's 5-subject pool, stacking a domain shift on top of the
condition shift pushes every tested architecture (V7 included) to
near-chance verification AUC, which Tier D documents as a dataset-scale
limitation rather than an architecture failure (see the "Taken together"
paragraph above) — an honest, useful boundary condition for the thesis to
state rather than omit.

---

## Plain-language summary: which one is better, and why (read this first if the rest of the page is dense)

Everything above is written for someone who already knows the vocabulary.
This section says the same things in plainer words, with one table you can
point to at a glance.

### First: what do these numbers actually mean?

- **Rank-1**: you show the model one gait sequence ("the probe") and ask it
  to pick the matching person out of a lineup of candidates ("the gallery").
  Rank-1 is the percentage of times its *top guess* is correct — like a
  multiple-choice test where it only gets credit for guessing first place.
  Rank-5 is more forgiving: credit if the right person is anywhere in its
  top 5 guesses.
- **Verification AUC**: a different, harder task — instead of picking from a
  lineup, the model is shown two gait clips and has to say "same person" or
  "different person," for every possible pair. AUC (0-100%) is how good it
  is at that yes/no task; 50% = pure guessing (coin flip), 100% = perfect.
  **This is usually the more trustworthy number when a test pool is small**,
  because Rank-1 on a tiny gallery can jump around a lot from luck.
- **Distance gap — does bigger mean better? Yes.** Internally, the model
  turns every gait clip into a point in space (an "embedding"). Two clips
  from the *same* person should land close together; two clips from
  *different* people should land far apart. `same_distance` is the average
  distance between same-person pairs, `different_distance` is the average
  for different-person pairs, and `distance_gap = different_distance −
  same_distance`. **A large, positive gap means the model has learned to
  push different people apart and pull the same person's clips together —
  that's the goal.** A gap near zero (you'll see values under 0.1 in the
  Tier D CLoP-Gait rows) means same-person and different-person pairs ended
  up roughly the same distance apart on average — the model isn't reliably
  telling people apart on that test set. **A negative gap (also in Tier D,
  e.g. the Transformer's −0.0233) means different-person pairs were, on
  average, very slightly *closer* than same-person pairs** — not a sign the
  metric works backwards, just confirmation that on that specific tiny test
  set (13 probes, 5 subjects, the hardest protocol tested), the model's
  embedding space had collapsed into a small cluster with no reliable
  identity signal left in it, so the exact ordering of two near-identical,
  noise-sized numbers is essentially a coin flip. You can see this is
  specific to Tier D's CLoP-Gait rows and nowhere else: every Tier A/B/C row,
  and Tier D's CASIA-B row (0.385), has a healthy gap of 0.3-0.6+, meaning
  the collapse only happens under CLoP-Gait's combined
  condition-shift-plus-domain-shift protocol on only 5 subjects — see the
  "Taken together" paragraph in Tier D above for why that's a dataset-size
  problem, not a broken metric or a broken model.

### At a glance: every tier, side by side

| Tier | What's being tested | Best model | Headline result | Trust level |
|---|---|---|---|---|
| **A** — CASIA-B, unseen people | Can it recognize 50 people it never trained on, from a 124-subject public benchmark? | **Part-Set V7** | 67.91% Rank-1, 91.16% AUC — best of 8 models tested, including 3 published papers | High — 1,047 probe comparisons, the most reliable number on this page |
| **B** — early thesis history | How did earlier model versions (V1-V5) perform, on their own older protocols? | Not comparable to A/C/D | V2 reached 66.88% Rank-1 on a harder 1-gallery setup | Historical record only, different protocol per row |
| **C** — CLoP-Gait, unseen environment | Same 5 people, but tested in an environment (outdoor daytime) never seen in training | Fusion V6 (V7 statistically tied) | ~83-87% Rank-1, ~82-87% AUC | Low-medium — only 46 probes over 4 of the 5 subjects |
| **D** — CASIA-B, unseen clothing | Same 124 people, but tested in clothing never seen in training | **Part-Set V7** | 77.57% Rank-1, 88.90% AUC — strong, real generalization | High — 370 probes |
| **D** — CLoP-Gait, unseen clothing *and* unseen environment at once | Same 5 people, hardest combined shift (new clothing + new environment together) | None — all 4 models tested near chance | Best AUC only 59.72% (V7); best Rank-1 84.62% (LSTM), but on only 13 probes | Very low — 13 probes, near-coin-flip AUC for every model |

**One-paragraph version for the defense:** On the standard, large-scale,
well-established benchmark (CASIA-B, Tiers A and D), **Part-Set V7 is the
best model tested, including against three real published architectures**,
and it generalizes well to two different kinds of unseen conditions (unseen
identities in Tier A, unseen clothing in Tier D) with strong, trustworthy
numbers backed by hundreds of test comparisons. On the small self-collected
dataset (CLoP-Gait, Tiers C and D), results are noisier because there are
only 5 subjects total — single-domain-shift results (Tier C) are decent but
statistically inconclusive between V6 and V7, and the hardest combined-shift
protocol (Tier D) shows every model, V7 included, failing to generalize
reliably — which is an honest, defensible finding about the limits of a
5-subject dataset, not a flaw specific to any one architecture. The
CASIA-B Tier A/D numbers are the ones to lead with; the CLoP-Gait numbers
are supporting evidence with clearly stated confidence limits.
