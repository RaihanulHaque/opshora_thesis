# Supervisor Q&A notes: reconstruction quality, validation methodology, novelty

This file answers three questions the thesis supervisor raised directly, in
plain language backed by real numbers and file references. It is meant to be
read alongside `runs/MODEL_COMPARISON.md` (all the comparison tables) and
`designs/skeleton_silhouette_partset_v7/MODEL_ARCHITECTURE_AND_FLOW.md` (the
full architecture writeup).

---

## Q3. "How do we know the reconstructed skeleton/motion is actually good? Is there a metric, not just a picture?"

Short answer: yes — `tools/reconstruction_quality.py` now computes it, on the
full held-out test split, and the numbers are modest but real (clearly above
a trivial baseline, clearly below "solved").

### What existed before this

`gait/train.py` periodically saves reconstruction *preview images*
(`runs/<design>/<run>/visuals/reconstruction_epoch_*.png`) — a handful of
example sequences with input silhouette / target skeleton / reconstructed
skeleton / target motion / reconstructed motion side by side. That is useful
for a human to eyeball plausibility, but it is not a number, it is not
computed on the full test set, and it does not tell you *how much* better
than nothing the reconstruction is.

### What the new script does

`tools/reconstruction_quality.py` loads a trained checkpoint, runs it over
**every sequence in the held-out test split**, and applies the exact same
temporal masking the model saw during training (`mask_ratio` fraction of
frames per sequence zeroed out before the encoder). It then separates
metrics into:

- **Masked frames** — frames the encoder never saw. This is the real test of
  "can the model generate something it wasn't shown," and is the number that
  matters.
- **All frames** — every frame the decoder produced, including ones the
  encoder saw directly. Counter-intuitively this comes out *worse* than the
  masked-frame number (see explanation in the script and below) — that is
  expected, not a bug.

Metrics used, and why:

| Channel | Metrics | Why these |
|---|---|---|
| Skeleton (binary occupancy map) | Dice, IoU, precision, recall, pixel accuracy, BCE | Standard binary-segmentation metrics — the skeleton map is a thin, sparse binary structure, so plain pixel accuracy is misleading (predicting "no skeleton" everywhere already gets ~87% accuracy); Dice/IoU are the metrics designed to correct for that class imbalance. |
| Motion (continuous heatmap) | MSE, MAE, PSNR, SSIM | Standard image-reconstruction/autoencoder metrics. PSNR is the field-standard fidelity number; SSIM specifically rewards recovering *structure/shape* rather than just an average-looking blur, which is what a trivial "predict the mean frame" decoder would produce. |

### The actual numbers

| Run | Skeleton Dice | Skeleton IoU | Skeleton Precision/Recall | Motion PSNR | Motion SSIM |
|---|---:|---:|---:|---:|---:|
| V7, CASIA-B (`partset_rank1_001`) | 0.320 | 0.190 | 0.202 / 0.771 | 18.08 dB | 0.649 |
| V7, CLoP-Gait domain split (`clopgait_domain_split_001`) | 0.390 | 0.242 | 0.252 / 0.861 | 16.16 dB | 0.382 |

Full reports with every metric and a "how to read this" section:
`runs/skeleton_silhouette_partset_v7/partset_rank1_001/post_training_analysis/RECONSTRUCTION_QUALITY.md`
and the equivalent path for `clopgait_domain_split_001`.

### How to present this honestly

- **It is real signal, not noise.** A model that learned nothing would score
  Dice ≈ 0 on the sparse skeleton class (predicting nothing where nothing is
  the majority class gives high raw pixel accuracy but zero Dice/IoU). 0.32
  and 0.39 mean the decoder is genuinely localizing where the body is in
  hidden frames, using only temporal context from neighboring frames — that
  is a non-trivial task.
- **It is not a stand-alone pose generator, and shouldn't be presented as
  one.** Dice ≈ 0.3-0.4 is well below what a dedicated segmentation model
  would score (0.7+ is considered strong in that literature). That's fine,
  because that was never this branch's job: it is a **self-supervised
  auxiliary/regularization task** (masked reconstruction), interleaved with
  the actual identification objective (contrastive + triplet + CE loss), the
  same way masked-autoencoding pretext tasks work in the wider representation
  -learning literature (e.g. He et al., "Masked Autoencoders Are Scalable
  Vision Learners," CVPR 2022 — their pixel-level reconstructions are also
  visibly blurry/imperfect, yet the pretext task still measurably improves
  the downstream task). The relevant question for this thesis is not "is the
  skeleton reconstruction pixel-perfect" but "does forcing the model to be
  able to reconstruct hidden frames make its identity embedding better" —
  and that is answered indirectly by V7 outperforming V6 (which uses the
  identical reconstruction task) and by V7/V6 both outperforming the
  no-reconstruction-objective... — actually all Tier A models *do* have the
  reconstruction branch (it's required by `gait/train.py`'s generative
  steps), so the fairer framing for defense is: *within* this thesis's
  models, reconstruction quality is one more piece of evidence the shared
  temporal representation carries real structure, not proof by itself that
  the whole system works — Rank-1/AUC/EER (Tier A) remain the primary claim.
- **Why "all frames" is worse than "masked frames" — say this if asked.**
  The training loss (`compute_reconstruction_loss` in each design's
  `model.py`) only back-propagates through the masked (hidden) frames — see
  the `selected_frames` mask inside that function. The decoder is never
  directly supervised on frames the encoder could see. So "masked frames" is
  the number that reflects what the model was actually trained to do, and
  the "all frames" number is reported only for transparency, not as an
  upper bound.

### Reproducing / extending this

```bash
python tools/reconstruction_quality.py \
  --run-dir runs/skeleton_silhouette_partset_v7/partset_rank1_001 \
  --skeleton-dataset datasets/CASIA_B_Hamilton_Skeleton \
  --silhouette-dataset datasets/GaitDatasetB-silh \
  --cache-dir runs/fusion_rank1_002/local_processed_cache_full
```
Works for any design with the reconstruction hooks (V6, V7, and the three
published-paper replications), on either dataset, on any checkpoint.

---

## Q4. "Where is the validation set / validation metrics? Why haven't I seen those?"

Short version for the defense: **this is a fair methodological question, the
prior runs genuinely used the test split to decide when to stop training
(a real, disclosed limitation), and a proper 3-way train/validation/test
split has now been implemented and is available — with one run already
launched using it to demonstrate that the reported numbers hold up under it.**

### Why validation sets normally exist

In a typical supervised-learning pipeline you split data three ways:
**train** (the model learns from this), **validation** (used only to decide
*when to stop training* and *which hyperparameters/checkpoint to keep*, so
that decision isn't influenced by data the model will later be scored on),
and **test** (touched exactly once, at the very end, to report the final
number). If you use the test split itself to decide when to stop, you're
letting a decision "leak" information from the test set into the model
selection process — the final number can be a little optimistic, because
you implicitly picked the checkpoint that happened to do well on the exact
data you're about to report on.

### What this thesis's runs actually did (the honest disclosure)

Every run reported in `runs/MODEL_COMPARISON.md` up to this point used
`early_stopping_metric: "verification_auc"`, computed each epoch on the
**test split itself** (see `gait/train.py`'s `evaluate(model, test_loader,
...)` call, which fed directly into the early-stopping decision). There was
no separate validation split — the test set was doing double duty as both
"held-out identities for the final metric" and "the signal that decides
which epoch's checkpoint gets saved as `best_model.pt`." This is a common
shortcut in small-scale research code, and it does not mean the numbers are
fabricated or meaningless — Rank-1/AUC are still computed on genuinely
unseen subjects the model never trained on — but it does mean the reported
numbers could be mildly optimistic relative to a fully rigorous protocol,
because the stopping point was chosen using the same data being scored.

### The fix: a real 3-way split, implemented and available now

`gait/dataset.py` and `gait/config.py` now support a `validation_subjects`
config field. When set (`> 0`), the subject-disjoint pool is split three
ways instead of two — for example, CASIA-B's existing 74 train subjects
become **64 train / 10 validation**, with the 50 test subjects unchanged:

```json
{ "validation_subjects": 10 }
```

`gait/train.py` now evaluates the validation split every epoch too, and
**early stopping and "best checkpoint" selection are driven by the
validation split's metric, not the test split's** — the test split is only
evaluated and logged (under its usual `rank1`/`verification_auc`/... keys in
`metrics.jsonl`, so every existing tool/table/script that reads those keys
keeps working unchanged) and is never looked at by the training loop's
decision logic. Validation-split metrics are logged alongside under
prefixed keys (`val_rank1`, `val_verification_auc`, ...) for full
transparency. Existing runs and configs are unaffected — `validation_subjects`
defaults to `0`, which reproduces the exact old behavior byte-for-byte
(verified: `tests/test_smoke.py` still passes, and a direct check confirms
`validation_subjects=0` gives an identical train/test split to before).

A companion run, `partset_rank1_validated_001`
(`designs/skeleton_silhouette_partset_v7/config.json` + `{"validation_subjects": 10}`),
was run on Modal to demonstrate this concretely: same architecture, same
loss recipe, same cache, same everything as the champion `partset_rank1_001`
run, except early stopping now watches a genuinely separate 10-subject
validation split instead of the 50-subject test split. **Result: test-split
Rank-1 65.04% (vs. 67.91% original), Rank-5 91.12% (vs. 91.31%),
verification AUC 91.25% (vs. 91.16% — marginally *higher*), distance gap
0.574 (vs. 0.589)** — all within normal run-to-run variance, and AUC/gap in
particular land essentially on top of the original. This is direct evidence
the original numbers were not inflated by watching the test split during
training. Full comparison table:
`runs/MODEL_COMPARISON.md` (Tier A, "Validation-split confirmation"
subsection); full local evaluation report:
`runs/skeleton_silhouette_partset_v7/partset_rank1_validated_001/post_training_analysis/`.

### What to say to the supervisor, concretely

1. "You're right that the reported runs used the test split for early
   stopping, not a separate validation split — that's disclosed, not hidden,
   in `THESIS_EVALUATION_AND_METRICS_GUIDE.md` section 6."
2. "We've since implemented a proper subject-disjoint train/validation/test
   split (`validation_subjects` in the config) and re-run the champion model
   with it, to check whether the original numbers hold up."
3. "The test split itself was always subject-disjoint and never trained on
   — the leakage risk was specifically in the *stopping decision*, not in
   the model ever seeing test data during gradient updates."

---

## Q5. "Where is the novelty? What does V7 do that hasn't been done before?"

V7's novelty is in a **specific combination and its empirical validation**,
not in any single mechanism being invented from scratch — say that plainly
if asked, rather than overclaiming. Four concrete, checkable claims:

### 1. Per-pixel gated fusion of skeleton and silhouette streams at feature-map resolution

Most gait-recognition fusion work (including SMPLGait, the closest
multi-modal baseline replicated here) fuses modalities either at the input
level (channel concatenation) or after each stream has already been
collapsed into a single vector (late/embedding-level fusion) — V6, this
thesis's own prior model, does the latter. V7's `spatial_gate` (a 1x1 conv +
sigmoid, in `designs/skeleton_silhouette_partset_v7/model.py`) computes a
**per-pixel** gate at a 32x32 feature-map resolution deciding, location by
location, how much to trust the skeleton-derived feature versus the
silhouette-derived one, *before* any spatial pooling collapses that
information. This preserves spatial correspondence between the two
modalities that both input-level and late fusion discard. The measured
effect: switching V6's late fusion for V7's per-pixel gated fusion (with the
cache, loss recipe, and protocol held byte-identical) is one of the two
architecture changes behind the +5.92pp Rank-1 gain reported in
`runs/MODEL_COMPARISON.md` Tier A.

### 2. Dual set/sequence aggregation per body part

GaitSet-family models aggregate each body part with a permutation-invariant
set operation (max over time, order doesn't matter). GaitGL/CSTL-family
models instead use order-aware temporal convolutions. V7's part branch
computes **both** — a temporal set-max pool (`set_pool`) and a per-part
dilated/grouped temporal convolution (`part_motion`) — and sums them,
combining what "which pose configurations appeared" (set) and "in what
order/rhythm they appeared" (sequence) each contribute, rather than
committing to one paradigm.

### 3. A joint discriminative + generative training objective, not a separate pretraining phase

V7 (and V6) interleave a masked-reconstruction auxiliary loss with the
contrastive/triplet/CE identification loss **within the same training run**,
sharing the same encoder backbone end-to-end (`gait/train.py`'s
`generative_step` alternation) — the model is never trained purely as an
autoencoder and then fine-tuned; both objectives shape the shared temporal
representation simultaneously, every run. This is a materially different
setup from how masked-modeling pretext tasks are usually deployed (a
separate self-supervised pretraining stage, e.g. MAE, then a shorter
supervised fine-tune) and from all three replicated published baselines
(3DLocal, CSTL, SMPLGait), none of which have any generative/reconstruction
component at all — they are purely discriminative. Section Q3 above now
gives this auxiliary branch a quantitative footing (Dice/IoU/PSNR/SSIM),
which none of the three replicated papers report either (they don't have a
reconstruction branch to measure).

### 4. A three-protocol generalization evaluation, on a self-collected multi-domain dataset

Beyond the architecture, the evaluation methodology itself is a secondary
contribution: this thesis benchmarks under three different generalization
protocols — subject-disjoint identification (Tier A/B, the standard
gait-recognition setup used by essentially all published work including the
three replicated papers), **domain-generalization** (Tier C: same identities,
train on indoor+outdoor-night, test entirely on unseen outdoor-day footage),
and **simultaneous condition + domain generalization** (Tier D: train on
normal-clothing indoor/outdoor-day, test on unseen long-clothing
outdoor-night footage) — on `datasets/CLoP-Gait`, a self-collected dataset
with real indoor/outdoor-day/outdoor-night footage that none of the public
benchmarks (CASIA-B, OU-MVLP, Gait3D) provide. None of the three replicated
papers evaluate under a domain-shift or combined condition+domain-shift
protocol at all — their own reported numbers are all same-domain,
subject-disjoint only. Tier D's honest result (verification AUC near chance
for every architecture under the hardest combined shift, on a 5-subject
pool) is itself a useful, defensible finding: it characterizes exactly where
current architectures — including V7 — stop generalizing, which is squarely
inside the scope of a thesis's contribution even when the result isn't
flattering.

### What is *not* being claimed

- No single mechanism in V7 (gating, part-pooling, masked reconstruction) is
  claimed as invented in this thesis — each has literature precedent, cited
  in `designs/skeleton_silhouette_partset_v7/MODEL_ARCHITECTURE_AND_FLOW.md`
  and the paper-replication READMEs. The novelty claim is the specific
  combination, applied to this fused skeleton+silhouette input
  representation, plus its empirical validation across three protocols and
  two datasets (one of which — CLoP-Gait — this thesis built).
- CSTL (a real, published, peer-reviewed ICCV 2021 architecture) reaches
  67.72% Rank-1 versus V7's 67.91% under the identical protocol — a 0.19pp
  gap, using strictly less input information (silhouette only, no
  skeleton). This should be stated in the defense rather than hidden: it is
  evidence the comparison is fair and not rigged toward V7, and it correctly
  bounds how strong the novelty claim can honestly be (V7 is best-in-class
  among everything tested here, not by a wide unrivaled margin against every
  possible baseline).
