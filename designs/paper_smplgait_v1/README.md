# paper_smplgait_v1 — replication of "SMPLGait" (silhouette branch / "w/o 3D" ablation)

Baseline comparison model requested by the thesis supervisor: an established,
published architecture trained under this repo's exact protocol, to compare
against `skeleton_silhouette_partset_v7` (our best model) on equal footing.

Source paper: Zheng, Liu, Liu, He, Yan, Mei. *"Gait Recognition in the Wild
with Dense 3D Representations and A Benchmark."* CVPR 2022.
(`papers/models/Zheng_Gait_Recognition_in_the_Wild_With_Dense_3D_Representations.pdf`)

## What the paper proposes

SMPLGait has two branches: a Silhouette Learning Network (SLN, a GaitSet-style
CNN) and a 3D Spatial-Transformation Network (3D-STN) that consumes per-frame
SMPL body-mesh parameters (pose/shape/viewpoint, recovered from RGB video via
a 3D human-mesh-recovery model) to align silhouette features across
viewpoints via a learned transformation matrix. It is trained/evaluated on
their own Gait3D dataset, which ships SMPL fits for every frame.

## Why we can only replicate part of it, and which part

**Neither CASIA-B nor CLoP-Gait provides SMPL parameters** -- no 3D mesh
recovery was ever run on either dataset, so there is no 3D input to feed the
3D-STN branch. Rather than silently drop it or fabricate 3D data, we
implement exactly the ablation the paper itself reports and names:
**"SMPLGait w/o 3D"** (their Table 2), i.e. the SLN branch alone, with the
3D-STN branch and the feature-alignment matrix multiplication removed. This
is a real, authors-defined configuration with its own reported numbers
(Gait3D, 88x128: R-1 47.70 vs. full SMPLGait's 53.20), not an improvised
simplification.

## What we replicated

- SLN: a 6-layer 2D CNN, matching the paper's description ("similar to the
  backbone of GaitSet").
- **Set Pooling**: max-pooling over the time dimension at the raw
  feature-map resolution (before any spatial pooling) -- GaitSet's defining
  operation, referenced explicitly in the paper's Fig. 2.
- **Horizontal Pyramid Pooling (HPP)**: multi-scale horizontal stripe
  pooling at scales {1, 2, 4, 8} with a separate FC per stripe (GaitSet/GLN
  convention; the paper's Fig. 2 shows "SP -> HPP -> Feature Aggregation"
  without stating exact stripe scales, so we use the standard convention
  shared by the other two baseline papers replicated in this repo).

## Adaptations (documented, not silent)

- 3D-STN branch and SMPL-based viewpoint alignment: **omitted entirely**, no
  data available (see above) -- this is the paper's own "w/o 3D" ablation,
  not a shortcut.
- Input modality: silhouette-only, matching both the paper's SLN and the
  w/o-3D ablation. The `topology` channel is used only as the framework's
  mandatory reconstruction target, never fed to the encoder.

## Training protocol (identical to every Tier A design in this repo)

Same cache, same SupCon + batch-hard triplet + CE loss recipe (not the
paper's own triplet+CE-only loss, for cross-design comparability), same
3-gallery CASIA-B protocol, same masked-reconstruction auxiliary task. Base
config: `config.json` (lr 0.00032). CLoP-Gait domain-generalization
overrides applied via `--config` at submission time.

## Run commands

```bash
python submit_modal.py run --design paper_smplgait_v1 --run casia_001
python submit_modal.py run --design paper_smplgait_v1 --run clopgait_001 \
  --config designs/clopgait_dataset_overrides.json
```
