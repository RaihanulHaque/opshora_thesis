# Skeleton Contrastive V3

This design follows Opshora's requested central architecture while keeping the implementation stable enough for rapid Modal experiments.

For a shareable architecture-and-flow explanation with Mermaid diagrams, see [`MODEL_ARCHITECTURE_AND_FLOW.md`](MODEL_ARCHITECTURE_AND_FLOW.md).

## What changed from V2

- Uses `datasets/CASIA_B_Hamilton_Skeleton/` / Modal `CASIA_B_Hamilton_Skeleton.zip`.
- Treats each Hamilton skeleton sequence as the primary input, not a derived side-channel.
- Builds compact 3-channel frame maps:
  1. binary Hamilton skeleton,
  2. blurred structural skeleton field,
  3. temporal motion-difference field.
- Uses a self-supervised generative stage: masked skeleton/motion reconstruction from surrounding frames.
- Uses contrastive verification instead of ID classification:
  - supervised contrastive loss,
  - batch-hard triplet loss,
  - `lambda_ce = 0`, so no subject-ID softmax is optimized.

## Thesis wording

The model learns latent pose dynamics from continuous Hamilton skeleton tracks using a temporal encoder. A reconstruction decoder forces the encoder to retain motion continuity and skeleton structure. The same latent representation is projected into a metric embedding space where same-identity sequences are pulled together and different-identity sequences are pushed apart. At test time the system performs blind verification/retrieval by embedding distance, not closed-set identity classification.

## Why no GAN discriminator here?

The advisor diagram shows a GAN-like feature-learning block. This implementation uses masked/future-style reconstruction as the generative objective because it is significantly more stable under the thesis time constraint and still sends the generative loss through the same encoder. If the stable version works, a later `v4_gan` design can add an adversarial discriminator without risking the current pipeline.

## Run

```bash
modal deploy modal_app.py
python submit_modal.py run --design skeleton_contrastive_v3 --run baseline_001
```

Expected Modal outputs:

- `/data/experiments/skeleton_contrastive_v3/<run>/metrics.jsonl`
- `/data/experiments/skeleton_contrastive_v3/<run>/best_model.pt`
- `/data/experiments/skeleton_contrastive_v3/<run>/visuals/training_curves.png`
- `/data/experiments/skeleton_contrastive_v3/<run>/visuals/reconstruction_epoch_*.png`

For an explanation of the reconstruction preview panels, see [`../../V3_RECONSTRUCTION_VISUAL_GUIDE.md`](../../V3_RECONSTRUCTION_VISUAL_GUIDE.md).

Important metrics:

- `same_distance`: lower is better.
- `different_distance`: higher is better.
- `distance_gap`: should become positive and increase.
- `verification_auc`: higher is better; shows same/different separation.
- `rank1`, `rank5`: retrieval-style comparison metrics.

For a full explanation of every logged field, see [`../../THESIS_METRICS_GUIDE.md`](../../THESIS_METRICS_GUIDE.md).

## Current result interpretation

The AUC-monitored run reached approximately:

- best verification AUC: `0.889`
- same-person distance: about `0.28`
- different-person distance: about `0.84`
- distance gap: about `0.56`
- Rank-1 retrieval: about `0.46`
- Rank-5 retrieval: about `0.74`

This means V3 is currently more convincing as a blind verification model than as a strict nearest-neighbor identification model. That matches the updated thesis requirement: the model should show that same-identity gait samples cluster closely while different identities are pushed apart, without relying on explicit subject-ID prediction at test time.
