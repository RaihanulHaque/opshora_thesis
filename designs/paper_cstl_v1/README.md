# paper_cstl_v1 — replication of "Context-Sensitive Temporal Feature Learning
for Gait Recognition" (CSTL)

Baseline comparison model requested by the thesis supervisor: an established,
published architecture trained under this repo's exact protocol, to compare
against `skeleton_silhouette_partset_v7` (our best model) on equal footing.

Source paper: Huang, Zhu, Wang, Wang, Yang, He, Liu, Feng. *"Context-Sensitive
Temporal Feature Learning for Gait Recognition."* ICCV 2021.
(`papers/models/Huang_Context_Sensitive_Temporal_Feature_Learning...pdf`)

## What the paper proposes

Two complementary modules on top of a 4-layer CNN + K-part horizontal
pooling:

1. **Multi-Scale Temporal Extraction + Adaptive Temporal Aggregation (ATA)**:
   part-pooled per-frame features are expanded into frame-level, short-term
   (two serial 1D temporal convs) and long-term (attention-weighted temporal
   pooling) representations. A relation-modeling FC looks across all three
   scales and predicts a soft gate that adaptively re-weights them before
   averaging into one sequence-level feature `T`.
2. **Salient Spatial Feature Learning (SSFL)**: a per-frame, per-part
   saliency score selects the best-appearing frame for each body part,
   recombining them into one occlusion-robust "recombinant frame" `S`.

Final embedding = FC(concat(T, S)). Reported CASIA-B mean rank-1: NM 97.8 /
BG 93.6 / CL 84.2 (their Table 1).

## What we replicated exactly

- The three-scale decomposition (frame-level / short-term / long-term) and
  its cascade-sum relation modeling (`T̃f = Tf`, `T̃s = Tf+Ts`,
  `T̃l = Tf+Ts+Tl`, their Eq. 2).
- The soft cross-scale gate that adaptively re-weights the three scales
  before aggregation (their core contribution).
- The saliency-driven per-part recombinant spatial feature, using their own
  normalized soft weighting (Eq. 6).
- K = 8 horizontal body-part division, matching this repo's other
  part-based designs.

## Adaptations (documented, not silent)

- **Gate granularity**: the paper's relation-modeling gate is defined
  per-channel-per-part (`W_T` has shape `B x N x 3 x C x K`). We compute it
  at a coarser per-frame-per-scale granularity (global-pooled context -> a
  3-way sigmoid gate), broadcast over channels/parts. This keeps the
  relation-modeling FC small while preserving the paper's central idea --
  adaptively re-weighting temporal scales using cross-scale context.
- **SSFL selection is soft, not hard**: the paper picks the single most
  salient frame per part via `argmax` (Eq. 9), and separately supervises the
  saliency scores with an auxiliary cross-entropy loss on a soft-weighted
  feature (Eq. 6-7). This repo's training recipe is fixed across every
  design for comparability and does not support per-design auxiliary
  losses -- a hard `argmax` here would leave the saliency network's
  gradient permanently disconnected from the primary loss. We therefore use
  their own soft, normalized saliency weighting (Eq. 6) directly as the
  aggregation weights, which is differentiable and functionally the same
  "recombine the most discriminative parts across frames" idea.
- **Input modality**: silhouette-only, exactly matching the paper. The
  `topology` channel is used only as the framework's mandatory
  reconstruction target, never fed to the encoder.

## Training protocol (identical to every Tier A design in this repo)

Same cache, same SupCon + batch-hard triplet + CE loss recipe (not the
paper's own CE+triplet-only loss, since every design in this repo shares one
training recipe for comparability), same 3-gallery CASIA-B protocol, same
masked-reconstruction auxiliary task. Base config: `config.json` (lr
0.00032). CLoP-Gait domain-generalization overrides applied via `--config`
at submission time.

## Run commands

```bash
python submit_modal.py run --design paper_cstl_v1 --run casia_001
python submit_modal.py run --design paper_cstl_v1 --run clopgait_001 \
  --config designs/clopgait_dataset_overrides.json
```
