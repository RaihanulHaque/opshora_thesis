# Thesis Method Proposal

## Recommended title

**Cross-Domain Gait Representation Learning using Hamilton–Jacobi Topology and Generative–Contrastive Learning**

This title is more defensible than “CT-PoseGAN.” A medial-axis skeleton is a topological representation derived from a silhouette; it is not an anatomical pose estimate. The method below is generative because it reconstructs or predicts topology sequences, but it does not require an adversarial GAN.

## 1. Thesis claim and research questions

The thesis should test one focused claim:

> A Hamilton–Jacobi topological representation, combined with generative temporal learning and condition-aware contrastive learning, improves gait recognition across lighting, environment, clothing, and view changes.

This becomes three research questions:

1. Does Hamilton–Jacobi topology outperform a silhouette-only baseline under clothing and degraded-segmentation conditions?
2. Does generative temporal reconstruction improve the representation over identification training alone?
3. Does cross-condition contrastive learning make the representation more invariant to indoor/outdoor, day/night, clothing, and view changes?

The Hamilton–Jacobi algorithm itself is not new. The thesis contribution is its use and systematic evaluation as a gait representation inside a cross-domain generative–contrastive framework.

## 2. Terminology corrections

- Use **Hamilton–Jacobi skeleton** or **Hamilton–Jacobi medial axis**, not “Hamilton medial axis.” The cited Siddiqi et al. method detects shocks using the average outward flux of a distance-field gradient and combines this with homotopy-preserving thinning.
- **Fast Marching Method (FMM)** computes the distance/arrival-time field. FMM alone is not the complete skeleton extraction algorithm.
- The extracted structure is a **topological skeleton**, not a joint skeleton and not ground-truth human pose.
- Lighting invariance cannot be claimed solely from a binary silhouette. Lighting affects the upstream segmentation. The experiment must therefore measure both silhouette quality and final recognition under day/night capture.
- A coat can change the medial axis and add branches. Clothing invariance is a hypothesis to test, not an assumed property.

## 3. Proposed system: HJ-TopoGait

### 3.1 End-to-end flow

```text
Video
  |
  v
Person segmentation and tracking
  |
  v
Binary silhouette sequence
  |
  +--> cleanup, alignment, 64x44 normalization, T=30 sampling
  |
  v
Euclidean distance field (FMM or exact distance transform)
  |
  v
Average-outward-flux shock detection
  |
  v
Homotopy-preserving thinning + conservative geodesic pruning
  |
  +--> topology heatmap K_t
  +--> normalized distance/radius map R_t
  +--> cleaned silhouette S_t
  |
  v
Lightweight two-stream spatial encoder
  |                         |
  | silhouette context      | HJ topology + radius
  +------------+------------+
               v
       gated feature fusion
               |
               v
       temporal encoder/pooling
               |
               +--> reconstruction/prediction decoder (training only)
               |
               +--> contrastive projection head (training only)
               |
               +--> identity embedding (gallery/probe matching)
```

All segmentation, distance-field, skeletonization, pruning, alignment, and temporal sampling should be performed offline. Training should load small precomputed tensors rather than recomputing coordinate grids.

### 3.2 Input representation

For frame `t`, retain three aligned maps:

- `S_t`: cleaned binary silhouette;
- `K_t`: one-pixel Hamilton–Jacobi skeleton, softly dilated into a narrow heatmap for stable convolution;
- `R_t`: normalized distance-to-boundary values sampled on or near the skeleton.

The primary proposed representation is `[K_t, R_t]`. `S_t` is a complementary context stream and a direct connection to strong silhouette baselines. This design permits three fair variants: silhouette-only, topology-only, and fused.

Do not convert `K_t` into anatomical joints unless a separate, validated graph-construction method is introduced. Branches of a medial axis do not consistently correspond to elbows, knees, or hips.

### 3.3 Backbone and temporal representation

A thesis-scale model should be intentionally small:

1. A shallow ResNet-like frame encoder for `S_t`.
2. A second shallow encoder for `[K_t, R_t]`.
3. Gated fusion so the model can reduce its reliance on a damaged modality.
4. Temporal max/mean pooling plus a small temporal attention block, producing embedding `h`.
5. A BNNeck-style identity embedding `z_id` for retrieval.

The OpenGait/GaitBase pattern is the appropriate recognition baseline: strong, simple, and practical. A large transformer is unnecessary for 20 custom subjects and would make the ablation less convincing.

### 3.4 Generative branch

Use a lightweight decoder only during training. Recommended pretext task:

> Mask 30–50% of the topology frames or temporal patches and reconstruct the missing `K_t` and `R_t` maps.

An alternative is reverse-sequence reconstruction, grounded in the supplied self-supervised gait paper. Masked reconstruction is easier to scale and prevents trivial copying. The loss is:

```text
L_gen = BCE(K_hat, K) + lambda_r * SmoothL1(R_hat, R)
```

This forces `h` to encode posture evolution and gait dynamics. Reconstruct topology maps, not RGB frames; RGB reconstruction wastes memory learning background and illumination.

If the supervisor explicitly requires adversarial training, add a very small temporal PatchGAN discriminator over reconstructed topology sequences as an optional experiment. It should not be the default because the literature itself notes GAN instability, and adversarial realism is not necessary for learning a recognition embedding.

### 3.5 Contrastive and recognition branches

Use a subject-condition sampler: each batch contains multiple identities and at least two different conditions for every sampled identity.

Positive pairs:

- same subject, different indoor/outdoor setting;
- same subject, different day/night setting;
- same subject, normal/heavy clothing;
- same subject, different view when available.

Negatives are different subjects. Apply supervised contrastive loss to a small projection head, not directly to the final retrieval vector. Add standard identity cross-entropy and batch-hard triplet loss for a fair gait-recognition objective:

```text
L_recognition = L_supcon + lambda_id * L_CE + lambda_tri * L_triplet
L_total       = lambda_gen * L_gen + L_recognition
```

An optional gradient-reversal domain classifier can predict environment, illumination, and clothing from `h`; reversing its gradient encourages condition invariance. It must be reported as an ablation rather than hidden inside the main model.

### 3.6 Skeleton-first verification variant implemented as V3

Opshora's updated flowchart can be implemented as a skeleton-first variant rather than forcing every detail of a full GAN. The practical version now in `designs/skeleton_contrastive_v3/` uses her Hamilton skeleton maps as the primary input:

```text
Hamilton skeleton sequence
  |
  +--> skeleton map + structure blur + temporal motion map
  |
  v
shared spatial-temporal encoder
  |
  +--> masked skeleton/motion reconstruction decoder
  |
  +--> contrastive projection head
  |
  v
distance-based verification embedding
```

This satisfies the required closed-loop mechanism: the generative reconstruction loss and the contrastive/triplet metric loss both update the same primary encoder. The endpoint is not a subject-ID classifier. Subject labels are used only during training to form positive and negative pairs; test behavior is blind cross-view verification by embedding distance.

For reporting, the most important V3 metrics are:

- `same_distance`: average embedding distance between samples of the same person;
- `different_distance`: average embedding distance between different people;
- `distance_gap`: separation between different-person and same-person distances;
- `verification_auc`: probability that a random same-person pair is closer than a random different-person pair;
- Rank-1/Rank-5: retained only as retrieval comparison metrics.

The adversarial discriminator shown in the presentation should be described as an optional future extension. A stable masked generative decoder is the safer thesis implementation because it learns temporal skeleton dynamics without the convergence instability of GAN training.

## 4. Memory-safe training schedule

Detaching `h` before the contrastive head is not a valid general solution: it prevents contrastive learning from improving the shared encoder. A single shared forward graph also does not grow “exponentially”; the memory problem comes from retaining large decoder and adversarial activations.

Recommended curriculum:

1. **Warm-up:** train encoder + generative decoder with `L_gen`.
2. **Coupled training:** alternate two independent mini-steps:
   - G-step: fresh forward, update encoder + decoder with `L_gen`;
   - C-step: fresh forward, update encoder + projection/identity heads with `L_recognition`.
3. **Recognition fine-tuning:** remove the decoder and fine-tune the encoder and identity head.

This remains one coupled generative–contrastive framework, but only one graph is alive at a time. Use mixed precision, precomputed inputs, a 30-frame sequence, and gradient accumulation if a larger identity batch is needed. Do not use `retain_graph=True`.

## 5. Dataset design

### 5.1 Custom dataset matrix

For each of 20 consenting subjects, collect:

```text
2 environments  x 2 illumination levels x 2 clothing states
x 3 views       x 2 repetitions          = 48 sequences/subject
20 subjects x 48 sequences               = 960 sequences
```

Suggested labels:

- environment: indoor, outdoor;
- illumination: day/well-lit, night/low-light;
- clothing: normal, heavy/coat;
- view: 0°, 90°, 180°;
- repetition: 1, 2.

Each sequence should contain at least two complete gait cycles. Keep frame rate, camera height, subject-camera distance range, walking path, and output resolution documented. Retain raw video, silhouette, topology map, and metadata so preprocessing can be audited.

Use a consent form, anonymized subject IDs, restricted raw-video access, and a stated retention policy.

### 5.2 Public datasets

- **CASIA-B:** use its official identity split and NM/BG/CL gallery-probe protocol for clothing, carrying, and view evaluation.
- **CASIA-C:** treat it as a separate infrared/outdoor evaluation domain using its own protocol.
- Do not create cross-dataset positive identity pairs between CASIA-B and CASIA-C: their identities are not shared.
- Public datasets can be combined for unlabeled generative/augmentation-based pretraining, but their identity labels remain dataset-local.

The custom dataset is the only supplied dataset that can provide controlled same-person pairs across the complete indoor/outdoor, day/night, and clothing matrix.

### 5.3 Leakage-safe custom protocol

Use five-fold subject-disjoint evaluation. In each fold:

- 12 subjects for training;
- 4 for validation and threshold/pruning selection;
- 4 unseen subjects for testing.

For each test identity, enroll one gallery sequence, for example indoor + day + normal clothing, then probe all other condition/view combinations. Rotate the gallery condition in a secondary experiment. The model may use gallery samples for nearest-neighbor enrollment, but it must never optimize weights on test identities.

## 6. Evaluation plan

### 6.1 Primary recognition metrics

- Rank-1 and Rank-5 identification accuracy;
- mean Average Precision (mAP);
- Equal Error Rate (EER) and ROC-AUC for verification;
- mean and standard deviation across subject folds, with confidence intervals where possible.

Report a condition-transition matrix rather than only one average:

```text
indoor-day -> outdoor-day
indoor-day -> indoor-night
indoor-day -> outdoor-night
normal     -> heavy clothing
0°         -> 90° / 180°
```

### 6.2 Preprocessing measurements

Manually annotate a small stratified subset of silhouette frames across all conditions and report segmentation IoU/F1. For topology, report measurable stability quantities such as temporal Chamfer distance, connected-component count, and branch-count variation. Do not invent universal pass/fail thresholds unless a cited standard defines them.

### 6.3 Required baselines and ablations

| Experiment | Purpose |
|---|---|
| GEI + small CNN | classical low-cost baseline |
| OpenGait/GaitBase silhouette baseline | strong modern baseline |
| silhouette-only proposed encoder | controls for architecture |
| HJ topology-only | tests the central representation claim |
| silhouette + HJ topology | tests complementarity |
| ordinary thinning vs HJ flux thinning | tests whether HJ extraction matters |
| no generative branch | measures generative contribution |
| no contrastive branch | measures contrastive contribution |
| reconstruction + contrastive | tests the proposed combination |
| with/without domain adversary | tests optional domain invariance |

The thesis succeeds scientifically even if the fused model wins only under specific difficult conditions, provided those conditions and limitations are reported honestly.

## 7. Objective-to-deliverable mapping

| Thesis objective | Concrete deliverable |
|---|---|
| Construct a cross-domain dataset | 20-subject, 960-sequence controlled dataset, metadata schema, consent/ethics statement, and subject-disjoint protocol |
| Design a generative–contrastive framework | HJ-TopoGait with masked topology reconstruction and cross-condition supervised contrastive learning |
| Evaluate against existing work | GaitBase/GEI baselines, public protocols, condition matrix, ablations, Rank-k/mAP/EER, and statistical reporting |

## 8. Corrections needed in the current presentation

1. The OpenGait reference is currently paired with the Vision Transformer authors/title. Replace it with Fan et al., “OpenGait: Revisiting Gait Recognition Toward Better Practicality,” arXiv:2211.06597.
2. GaitGAN uses gait energy images made from silhouette sequences; its input modality should not be marked as non-silhouette.
3. “Cross-View Gait Recognition by Discriminative Feature Learning” is discriminative, not generative.
4. The literature comparison needs evidence for every indoor/outdoor, illumination, and clothing checkmark.
5. Replace “FMM Skeleton Extraction” with the complete chain: distance field/FMM -> flux-based shock detection -> homotopy-preserving thinning -> pruning.
6. Remove the weighted `VS` score and claimed universal thresholds unless every component, weight, and threshold is supported by a valid citation and validated on more than one frame.
7. Do not present ROC-AUC 0.78, EER 26.2%, or a one-frame skeleton score as final evidence for a model that has not yet been trained and tested under a leakage-safe protocol.

## 9. How the supplied papers support the design

- **Hamilton–Jacobi Skeletons:** correct flux-based topological skeletonization and its stability limitations.
- **OpenGait/GaitBase:** strong simple recognition backbone, public protocols, and need for outdoor evaluation.
- **Self-Supervised Gait Encoding with Locality Awareness:** reconstruction plus contrastive gait learning and reverse reconstruction.
- **Contrast-Reconstruction Representation Learning:** separation and later fusion of posture reconstruction and motion contrastive learning.
- **GaitGAN:** identity-preserving canonical generation, while also documenting GAN training difficulty.
- **GaitSTR and GaitCSF:** complementary silhouette/skeleton or silhouette/heatmap streams.
- **Cross-View Gait Recognition by Discriminative Feature Learning:** view-aware discriminative learning and temporal attention.
- **Aligning Silhouette Topology:** topology extraction from distance fields and the use of topology under domain shift.

## 10. Minimum viable experiment before full implementation

Before building the entire system, run one small falsification experiment:

1. Select 5–10 subjects with normal and heavy-clothing sequences.
2. Precompute silhouettes, ordinary skeletons, and Hamilton–Jacobi skeleton maps.
3. Train the same small encoder separately on silhouette-only, ordinary-skeleton, and HJ-skeleton inputs.
4. Test normal-gallery to heavy-clothing-probe Rank-1 accuracy.
5. Inspect failure cases and branch stability.

If HJ topology does not beat ordinary thinning or silhouette input under clothing change, revise the central novelty before spending time on the full generative–contrastive model.
