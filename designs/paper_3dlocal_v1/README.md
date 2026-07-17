# paper_3dlocal_v1 — replication of "3D Local CNNs for Gait Recognition"

Baseline comparison model requested by the thesis supervisor: an established,
published architecture trained under this repo's exact protocol, to compare
against `skeleton_silhouette_partset_v7` (our best model) on equal footing.

Source paper: Huang, Xue, Shen, Tian, Li, Huang, Hua. *"3D Local Convolutional
Neural Networks for Gait Recognition."* ICCV 2021. (`papers/models/Huang_3D_Local_...pdf`)

## What the paper proposes

A generic 3D-local building block, inserted into a GaitPart-style CNN
backbone after every conv block. For each of six body parts (head, left/right
arm, torso, left/right leg), a learned localization network predicts an
adaptive per-frame 3D sampling region (spatial center/scale + temporal
offset/stride); Gaussian spatial filters and trilinear temporal filters
extract that region; a small conv block extracts local features from it;
local + global features are fused by channel-concat + 1x1x1 conv. Reported
CASIA-B mean rank-1: NM 97.5 / BG 94.3 / CL 83.7 (their Table 1, best setting).

## What we replicated exactly

- The core idea: adaptive, *learned* per-part, per-frame spatial localization
  (not fixed horizontal stripes), via a differentiable Gaussian gate with a
  temporal-aware localization head.
- Six body-part paths with standard anatomical priors (head/torso/arms/legs).
- Insertion after every backbone block (their best-performing "setting f").
- Fusion by channel-concatenation + 1x1x1 conv back to the block's channel
  width, and GaitPart-style temporal max-pooling after spatial avg+max
  pooling of the final feature map.

## Adaptations (documented, not silent)

- **Sampling mechanism**: we implement the Gaussian-only variant of their
  local operation (multiplicative spatial gate at native feature-map
  resolution) instead of the full Gaussian + trilinear-temporal resampling
  onto a smaller M x N x L grid. Their own ablation (Table 4) shows
  Gaussian-only sampling performs within 0.1-1.3 rank-1 points of the full
  Gaussian+Trilinear "Mixture" sampling, so this is a paper-endorsed
  simplification, not an arbitrary one.
- **Input modality**: silhouette-only, exactly matching the paper. The
  `topology` (Hamilton-Jacobi skeleton) channel in our shared 4-channel
  cache is never fed to the encoder; it is used only as the reconstruction
  *target* required by this repo's shared masked-reconstruction training
  step (see `gait/train.py`), which has no equivalent in the original paper.
- **Backbone width**: we use 32/64/128 channels for the three blocks (their
  paper leaves the exact channel schedule flexible, "based more on
  convenience than necessity" per their own Sec. 3.2.3).

## Training protocol (identical to every Tier A design in this repo)

Same cache (`/data/processed/casia_b_skeleton_silhouette_fusion_v6`), same
SupCon + batch-hard triplet + CE loss recipe, same 3-gallery CASIA-B protocol
(74 train / 50 test subjects), same masked-reconstruction auxiliary task.
Base config: `config.json` (lr 0.00032, matching this repo's other
non-V6/V7 baselines). CLoP-Gait domain-generalization overrides applied via
`--config` at submission time (see `runs/MODEL_COMPARISON.md`).

## Run commands

```bash
python submit_modal.py run --design paper_3dlocal_v1 --run casia_001
python submit_modal.py run --design paper_3dlocal_v1 --run clopgait_001 \
  --config designs/clopgait_dataset_overrides.json
```

The CLoP override file only swaps dataset/split fields (paths, `split_mode`,
`train_subjects`, batch composition); this design's own architecture and
optimization hyperparameters (`config.json`: lr 0.00032 etc.) are preserved.
