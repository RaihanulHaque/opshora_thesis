# Opshora Archive Code Review

This file explains the three old scripts in `opshora_archive/`.

## Short verdict

The archive is useful as **preprocessing research evidence**, not as current model-training code.

It shows that Opshora previously worked on a Hamilton-Jacobi skeleton extraction pipeline from CASIA-B silhouettes. It does not contain the final generative-contrastive neural network.

## Files

| File | What it is | Main purpose |
|---|---|---|
| `01_Skeleton_Pipeline_Development_2_.py` | Exported notebook | Step-by-step development of Hamilton-Jacobi skeleton extraction on one CASIA-B frame. |
| `phase1_hamilton.py` | Reusable module extracted from notebook 01 | Packaged functions for binary verification, signed distance/FMM, gradient, average outward flux, homotopy thinning, and skeleton extraction. |
| `02_Dataset_Pipeline_Validation_1_.py` | Exported validation notebook | Runs `phase1_hamilton.run_hamilton_pipeline()` on representative frames across a CASIA-B subset and saves skeleton images. |

## 1. `01_Skeleton_Pipeline_Development_2_.py`

This is the original exploratory notebook converted to `.py`.

It starts with:

- locating one CASIA-B silhouette frame;
- loading the frame;
- checking whether it is binary;
- thresholding it if needed;
- displaying the image;
- then progressively developing the Hamilton-Jacobi skeleton pipeline.

The important algorithmic steps inside it are:

1. Binary silhouette verification.
2. Signed distance computation using `scikit-fmm`.
3. Gradient computation on the full signed distance field.
4. Average outward flux calculation.
5. Homotopy-preserving thinning.
6. Hamilton skeleton extraction.
7. Visual inspection plots.
8. Packaging the logic into `phase1_hamilton.py`.

This file is useful for understanding the mathematical preprocessing history, but it is not clean production code because it is a notebook export with many display cells and global variables.

## 2. `phase1_hamilton.py`

This is the cleanest and most reusable file in the archive.

It exposes functions such as:

- `verify_binary_silhouette()`
- `compute_signed_distance()`
- `compute_gradient()`
- `compute_average_outward_flux()`
- `homotopy_preserving_thinning()`
- `extract_hamilton_skeleton()`
- `run_hamilton_pipeline()`

This file is the archive's actual Hamilton-Jacobi preprocessing module.

Important detail:

```python
import skfmm
```

So it requires `scikit-fmm`. Our current Modal image does not install `scikit-fmm`; the current active pipeline uses `scipy.ndimage.distance_transform_edt` instead, which is simpler and easier to deploy.

### Useful parts

- Good mathematical comments.
- Clear separation of distance map, gradient, flux, and thinning.
- Better theoretical link to Hamilton-Jacobi skeletonization than the quick prototype in `gait/preprocessing.py`.
- Can be cited as Opshora's original preprocessing algorithm development.

### Risky parts

- It processes one image at a time.
- The thinning loop is heap-based and can be slow over many frames.
- It depends on `scikit-fmm`, which may complicate Modal deployment.
- It has many plotting/helper functions that are not needed for training.
- The flux threshold is manually chosen:

  ```python
  FLUX_THRESHOLD = -0.10
  ```

  This should be treated as a tunable hyperparameter, not a universal scientific value.

## 3. `02_Dataset_Pipeline_Validation_1_.py`

This is a dataset-level validation notebook export.

It does not implement deep learning. It does not train a model.

It does:

- auto-detect a CASIA-B dataset root;
- index subjects `001` to `040`;
- use conditions:

  ```text
  nm-01, nm-02, nm-03, nm-04, nm-05, nm-06, cl-01, cl-02
  ```

- use views:

  ```text
  000, 090, 180
  ```

- select one representative middle frame per sequence;
- call `run_hamilton_pipeline()` on that single frame;
- save:

  ```text
  Output/<subject>/<condition>/<angle>/<filename>_binary.png
  Output/<subject>/<condition>/<angle>/<filename>_skeleton.png
  ```

- generate coverage and success/failure reports.

This likely helped create or validate the Hamilton skeleton images Opshora later gave as a dataset.

### Main limitation

It processes only one representative frame per sequence. For gait recognition, we need full temporal sequences, not one frame.

Our current V3 pipeline uses full skeleton sequences:

```text
subject / condition / view / many skeleton frames
```

So V3 is better aligned with gait recognition than this validation notebook.

## Relationship to the current V3 pipeline

Current V3 uses:

```text
datasets/CASIA_B_Hamilton_Skeleton/
```

That dataset likely came from a later version of the archive preprocessing idea.

The archive focuses on:

```text
silhouette -> Hamilton skeleton
```

Current V3 focuses on:

```text
Hamilton skeleton sequence -> generative reconstruction + contrastive verification
```

So the archive is upstream preprocessing. V3 is downstream learning.

## Should we reuse the archive code?

Not directly for tonight.

Best use:

- cite it as preprocessing development history;
- compare its mathematical skeletonization logic against the current simplified preprocessing;
- maybe later create a `preprocessing_exact_hj.py` version if we want a more faithful Hamilton-Jacobi extractor.

Do not directly mix it into current Modal training unless we intentionally add:

- `scikit-fmm` to requirements and Modal image;
- batch/sequence processing;
- caching;
- speed tests;
- failure handling for all frames.

## Why it may have crashed before

Likely reasons:

1. Notebook-style code with many plotting calls.
2. Per-frame `scikit-fmm` computation is expensive.
3. Heap-based homotopy thinning can be slow on many frames.
4. Dataset loops can process hundreds/thousands of images without strong caching.
5. It was not designed as a GPU training pipeline.
6. It only validated representative frames, so scaling to full sequences would multiply runtime heavily.

## How to explain it to Opshora

Good wording:

> The archive code represents the earlier preprocessing phase of the thesis. It develops and validates Hamilton-Jacobi skeleton extraction from CASIA-B silhouettes. The current V3 model does not replace this work; it consumes the skeleton sequences produced by this kind of preprocessing and learns a generative-contrastive embedding for gait verification.

Important limitation:

> The archive code is not the final neural model. It is mainly preprocessing and validation. It does not fulfill the generative-contrastive learning objective by itself.

## Final recommendation

Keep `opshora_archive/` as historical evidence.

Do not delete it.

Do not rely on it as the main runnable training code.

Use the current `gait/` and `designs/` folders for experiments, because they are modular, Modal-friendly, and already produce training metrics/checkpoints.
