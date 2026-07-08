# Thesis Objective Audit

This file checks whether the current implementation and experiments satisfy Opshora's stated thesis objectives.

## Short verdict

The current work is a **good working prototype**, but the full thesis is **not completely fulfilled yet**.

Current status:

| Objective | Status | Reason |
|---|---|---|
| Objective 1: Construct a cross-domain gait dataset | **Not fully fulfilled** | Public CASIA-B/CASIA-C style datasets are available, and Hamilton skeleton sequences exist, but the required custom 20-subject indoor/outdoor day/night clothing dataset is not confirmed. |
| Objective 2: Design a generative and contrastive framework | **Mostly fulfilled** | `skeleton_contrastive_v3` implements a shared encoder with self-supervised generative reconstruction and contrastive/triplet metric learning. |
| Objective 3: Evaluate against existing work | **Partially fulfilled** | V1/V2/V3 results exist, but formal baselines, ablations, and protocol-level comparison to existing work are still missing. |

So the honest answer is:

> The central model objective is fulfilled as a functioning prototype. The dataset-construction and full evaluation objectives are not fully fulfilled yet.

## Objective 1: Cross-domain gait dataset

Original objective:

> To construct a cross-domain gait dataset covering indoor-outdoor settings under varying day-night illumination and clothing conditions.

### What is currently done

- CASIA-B silhouette data is available.
- CASIA-C silhouette data is available.
- CASIA-B Hamilton skeleton data is available.
- The code can load zipped or folder-based skeleton/silhouette sequence datasets.
- Modal storage and preprocessing cache are working.

### What is missing

The stated objective requires a custom or explicitly documented cross-domain dataset with:

- indoor sequences;
- outdoor sequences;
- day or well-lit sequences;
- night or low-light sequences;
- normal clothing;
- heavy clothing or coat condition;
- multiple views such as `0°`, `90°`, and `180°`;
- metadata showing subject ID, view, condition, illumination, and environment.

The public datasets help, but they do not fully replace the custom cross-domain dataset:

- CASIA-B mainly supports view, normal walking, carrying bag, and clothing variation.
- CASIA-C helps with infrared/night-like or outdoor-style domain shift.
- They do not share identities, so they cannot directly form same-person positive pairs across CASIA-B and CASIA-C.

### Verdict

**Not fully fulfilled.**

It is acceptable to say:

> Public datasets were used for prototype validation and benchmarking.

But it is not yet safe to say:

> We constructed the full cross-domain dataset required by Objective 1.

unless Opshora has actually collected and documented that custom dataset.

## Objective 2: Generative and contrastive framework

Original objective:

> To design a generative and contrastive paradigm framework for robust representation accuracy.

### What is currently done

`skeleton_contrastive_v3` satisfies the core model structure:

```text
Hamilton skeleton sequence
  -> shared spatial-temporal encoder
  -> generative reconstruction branch
  -> contrastive/triplet embedding branch
  -> verification by embedding distance
```

Implemented components:

- Hamilton skeleton sequence input.
- Temporal motion map input.
- Shared spatial-temporal encoder.
- Self-supervised generative reconstruction of skeleton/motion.
- Supervised contrastive loss.
- Batch-hard triplet loss.
- No subject-ID classifier in V3 by default (`lambda_ce = 0.0`).
- Early stopping based on `verification_auc`.
- Persistent Modal training workflow.
- Saved metrics, checkpoints, visualizations, and model snapshots.

The model also follows the advisor's closed-loop idea:

```text
generative loss       -> updates shared encoder
contrastive/triplet loss -> updates shared encoder
```

### Current evidence

Best V3 run:

```text
best_verification_auc ≈ 0.889
same_distance         ≈ 0.28
different_distance    ≈ 0.84
distance_gap          ≈ 0.56
rank1                 ≈ 0.46
rank5                 ≈ 0.74
```

This means the model learned an embedding where same-person gait sequences are usually closer than different-person gait sequences.

### Important limitation

The implemented generative branch is **not a full GAN**. It is a stable masked skeleton/motion reconstruction decoder.

This is defensible because:

- it is genuinely generative/self-supervised;
- it updates the same encoder used by the contrastive branch;
- it is much less likely to crash than adversarial GAN training;
- the presentation can describe adversarial discrimination as a future extension.

### Verdict

**Mostly fulfilled.**

The model architecture objective is fulfilled enough for a working thesis prototype. If the supervisor strictly demands an actual discriminator, then this becomes **partially fulfilled** until a small optional adversarial branch is added.

## Objective 3: Evaluation against existing work

Original objective:

> To evaluate the performance of our dataset and model based on existing work.

### What is currently done

The project has run several internal designs:

- V1: HJ topology/silhouette prototype.
- V2: improved HJ-TopoGait version.
- V3: skeleton-first generative-contrastive verification model.

Current V3 evaluation includes:

- Rank-1;
- Rank-5;
- same-person embedding distance;
- different-person embedding distance;
- distance gap;
- verification AUC;
- verification accuracy;
- early stopping on verification AUC.

### What is missing for a strong thesis defense

Formal evaluation against existing work still needs at least some of the following:

1. A simple baseline such as GEI + small CNN.
2. A silhouette-only baseline.
3. A skeleton-only contrastive baseline without generative reconstruction.
4. A no-generative ablation.
5. A no-contrastive or reconstruction-only ablation.
6. A comparison against a known gait protocol or paper result, if possible.
7. A condition-wise table:
   - normal to clothing;
   - same view to cross view;
   - normal to bag;
   - day/well-lit to night/infrared if dataset supports it.
8. Mean and standard deviation over more than one run or fold, if time allows.

### Verdict

**Partially fulfilled.**

The current metrics prove the model runs and learns useful verification structure, but they do not yet fully satisfy "evaluate against existing work" in the academic sense.

## What can be safely claimed now

Safe claim:

> A working generative-contrastive Hamilton skeleton gait framework was implemented and evaluated on an unseen-subject split. The model achieved a best verification AUC of approximately 0.889, indicating that same-identity gait sequences are embedded closer than different-identity sequences.

Safe claim:

> The framework avoids explicit subject-ID prediction at test time and performs blind verification through embedding-distance comparison.

Safe claim:

> The current result supports the feasibility of the proposed generative-contrastive gait representation.

## What should not be claimed yet

Do not claim:

> The full thesis objectives are completely fulfilled.

Do not claim:

> The model is state-of-the-art.

Do not claim:

> The model is fully robust to indoor/outdoor, day/night, and clothing changes.

Do not claim:

> The dataset objective is complete.

unless the custom cross-domain dataset and metadata are actually finished.

## Minimum next steps to make the thesis defensible

If time is short, do these in order:

1. Preserve the current V3 run as the main proposed model baseline.
2. Run one no-generative ablation:

   ```text
   skeleton_contrastive_v3 without reconstruction branch
   ```

   This proves whether generative learning helps.

3. Run one silhouette-only or skeleton-only baseline using the same split.
4. Create a final comparison table:

   | Method | Input | Generative? | Contrastive? | Rank-1 | Rank-5 | Verification AUC |
   |---|---|---|---|---:|---:|---:|
   | Baseline | silhouette or skeleton | no | yes/no | ... | ... | ... |
   | V3 no-generative | Hamilton skeleton | no | yes | ... | ... | ... |
   | V3 full | Hamilton skeleton | yes | yes | 0.467 | 0.737 | 0.889 |

5. Clearly write limitations:

   - Rank-1 retrieval is moderate.
   - Verification is stronger than strict identification.
   - Full cross-domain custom dataset validation remains future or ongoing work if not collected.

## Final decision

Current thesis fulfillment level:

```text
Objective 1: 40% fulfilled, unless custom dataset is already collected.
Objective 2: 85% fulfilled.
Objective 3: 55% fulfilled.
Overall: approximately 60–70% thesis-ready as a prototype, not yet 100% thesis-complete.
```

If Opshora's supervisor mainly wants a working generative-contrastive architecture tonight, the current system is enough.

If the supervisor wants a complete thesis defense package, more baseline and ablation experiments are still needed.
