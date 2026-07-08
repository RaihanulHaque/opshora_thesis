# V3 Reconstruction Visual Guide

This guide explains the reconstruction preview images saved by `skeleton_contrastive_v3`.

The visual is produced by the generative branch of the model. It is not the final recognition output. It is evidence that the network is learning temporal skeleton structure and motion patterns.

## What the five panels mean

Current older images may show these labels:

```text
input map | target channel 1 | reconstructed channel 1 | target channel 2 | reconstructed channel 2
```

For V3, read them as:

| Old label in image | Correct V3 meaning | Explanation |
|---|---|---|
| `input map` | Input Hamilton skeleton | The visible skeleton frame given to the model. White pixels are the Hamilton medial-axis skeleton. |
| `target channel 1` | Target skeleton | The true skeleton map the decoder should reconstruct for a masked frame. |
| `reconstructed channel 1` | Reconstructed skeleton | The model's predicted skeleton map. This shows what the generative decoder thinks the missing skeleton should look like. |
| `target channel 2` | Target motion map | The true temporal motion-difference map. It highlights skeleton pixels that changed from the previous sampled frame. |
| `reconstructed channel 2` | Reconstructed motion map | The model's predicted motion-change map. This shows whether the model learned walking dynamics, not just static shape. |

Future images will use clearer labels:

```text
input skeleton | target skeleton | reconstructed skeleton | target motion | reconstructed motion
```

## Why there are two targets

The V3 generative branch reconstructs two things:

```text
target 1 = skeleton structure
target 2 = temporal motion/change
```

This is intentional.

If the model only reconstructs skeleton structure, it may learn a static body shape. But gait recognition needs temporal walking dynamics. The motion channel forces the encoder to learn how the skeleton changes across frames.

## How the image is generated

During a generative training step:

1. The model receives a skeleton sequence.
2. Some time frames are masked or hidden.
3. The shared encoder sees the remaining sequence.
4. The decoder tries to reconstruct the missing skeleton and motion maps.
5. The reconstruction loss is backpropagated through the shared encoder.

So this preview visual shows:

```text
ground truth skeleton/motion
          vs.
model reconstructed skeleton/motion
```

The closer the reconstructed panels look to the target panels, the better the generative branch is learning.

## How to interpret the attached image

Your image shows:

```text
input map:
  A thin white Hamilton skeleton.

target channel 1:
  A softened/blurred skeleton target for the masked frame.

reconstructed channel 1:
  The model reconstructs the broad vertical body skeleton region.
  It captures the main body axis, but fine branches are blurry.

target channel 2:
  The temporal motion/change target.
  It highlights where the skeleton moved between adjacent frames.

reconstructed channel 2:
  The model predicts a broad motion field.
  It captures approximate motion location, but not sharp branch-level detail.
```

This is a normal result for a compact generative decoder. The reconstruction is blurry because the model is trained to learn useful latent motion representation, not to produce a perfect image segmentation.

## Is this good or bad?

This is acceptable for the thesis prototype.

Good signs:

- The reconstructed skeleton is in the correct body region.
- The reconstructed motion is not random noise.
- The output preserves a vertical human gait structure.
- The model reached strong verification separation, with best `verification_auc ≈ 0.889`.

Weak signs:

- Fine skeleton branches are not reconstructed sharply.
- The reconstruction is thicker and blurrier than the target.
- The motion reconstruction is approximate.

This means:

> The generative branch is learning coarse gait structure and motion, but not exact pixel-level Hamilton skeleton geometry.

For the thesis, that is okay because the final goal is not image reconstruction. The final goal is embedding-space verification.

## What to write in the thesis

Good wording:

> The reconstruction preview demonstrates that the generative branch learns a coarse temporal representation of Hamilton skeleton dynamics. Although the reconstructed skeleton and motion maps are smoother than the target maps, the decoder preserves the major body axis and motion region. This suggests that the shared encoder captures useful gait structure, which is further refined by contrastive metric learning.

Careful limitation:

> The reconstruction branch is not intended to produce pixel-perfect Hamilton skeleton maps. Its purpose is to regularize the shared encoder toward temporal motion awareness. Fine branch-level reconstruction remains imperfect and should be improved in future work.

Avoid saying:

> The model perfectly reconstructs the skeleton.

That would not be true from the image.

## How this connects to the thesis objective

The image supports the generative part of Objective 2:

> Design a generative and contrastive paradigm framework for robust representation accuracy.

The reconstruction preview shows the generative side.

The metrics such as `same_distance`, `different_distance`, and `verification_auc` show the contrastive verification side.

Together:

```text
reconstruction preview = evidence of generative temporal learning
verification metrics   = evidence of contrastive embedding separation
```

## Important note

The preview is only one frame from one sequence. It should be used as qualitative evidence, not as the final proof of model performance.

The quantitative proof is still:

```text
best_verification_auc ≈ 0.889
same_distance         ≈ 0.28
different_distance    ≈ 0.84
distance_gap          ≈ 0.56
```
