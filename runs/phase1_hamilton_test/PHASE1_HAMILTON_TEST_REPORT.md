# Phase 1 Hamilton Skeleton Test Report

Tested file:

```text
opshora_archive/phase1_hamilton.py
```

Sample silhouette:

```text
datasets/GaitDatasetB-silh/001/nm-01/090/001-nm-01-090-050.png
```

## Result

The code runs successfully after installing:

```text
scikit-fmm
```

The main callable function is:

```python
run_hamilton_pipeline(image)
```

It returns:

```text
binary_silhouette
signed_distance
distance_map
gradient_x
gradient_y
gradient_magnitude
flux_map
thinned_mask
hamilton_skeleton
```

## Full-size silhouette test

Input size:

```text
240 x 320
```

Output statistics:

```text
foreground pixels: 3637
skeleton pixels:   599
mask components:   1
skeleton components: 1
distance max:      14.6129
flux min/max:      -0.6954 / 0.3313
```

Output files:

```text
runs/phase1_hamilton_test/input_binary.png
runs/phase1_hamilton_test/hamilton_skeleton.png
runs/phase1_hamilton_test/skeleton_overlay.png
runs/phase1_hamilton_test/phase1_hamilton_panel.png
```

## V6-style 64x64 aligned test

Output statistics:

```text
foreground pixels: 664
skeleton pixels:   175
skeleton components: 1
```

Output files:

```text
runs/phase1_hamilton_test/aligned_64_input.png
runs/phase1_hamilton_test/aligned_64_phase1_skeleton.png
runs/phase1_hamilton_test/aligned_64_phase1_overlay.png
runs/phase1_hamilton_test/aligned_64_phase1_panel.png
```

## Interpretation

The implementation is functional. It produces a connected Hamilton-style thinned medial structure from a binary silhouette.

However, the skeleton is branchy/ribbed compared with the smoother skeleton maps currently used by the V6 model. This is expected from a topology-preserving thinning process, but it may not be ideal as-is for neural-network training unless the extra branches are useful.

## Recommendation

Use this code if Opshora specifically wants a Hamilton-Jacobi / average-outward-flux thinning implementation.

For the V6 training pipeline, do not replace the current preprocessing immediately. First run a small ablation:

```text
V6 current skeleton preprocessing
vs.
V6 phase1_hamilton skeleton preprocessing
```

Compare:

```text
Rank-1
Rank-5
Verification AUC
Distance gap
Condition-wise clothing performance
```

If the phase1 skeleton improves clothing robustness or verification AUC, then integrate it into the official preprocessing pipeline.

