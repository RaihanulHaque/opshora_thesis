# HJ-TopoGait v2

V2 responds to the v1 baseline (`best Rank-1 = 61.22%`) and its blurred topology reconstruction.

## Hypothesis

V1 discarded useful body-part structure through global pooling, under-sampled cross-condition positives, and optimized a sparse topology target with background-dominated BCE. Correcting these three issues should improve unseen-subject Rank-1 accuracy and produce sharper reconstructions.

## Changes from v1

- global + four horizontal-part pooling in both input streams;
- condition-aware P×K batches spanning normal, bag, fast, and slow sequences;
- weighted topology BCE plus soft Dice loss;
- radius loss concentrated on the target topology;
- deeper temporal encoder with one lightweight self-attention refinement layer;
- higher-resolution topology decoder seed and final convolutional refinement;
- one generative update per four batches after a two-epoch warm-up;
- cosine learning-rate decay;
- accurate per-objective average loss logging.

## Fair comparison

V2 uses exactly the same cached preprocessing, first-100-subject training split, remaining-53-subject test split, gallery selection, and Rank-1/Rank-5 evaluation as v1.

Run:

```bash
python submit_modal.py run --design hj_topogait_v2 --run baseline_001
```
