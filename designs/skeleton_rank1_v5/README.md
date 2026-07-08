# Skeleton Rank-1 V5

V5 is the safer Rank-1 recovery attempt after V4 failed.

V4 added a strong auxiliary identity classifier and Rank-1 dropped below V3. That suggests the classifier overfit the training identities or damaged the distance embedding. V5 goes back to metric learning as the main objective, but adds better pooling and a more retrieval-friendly evaluation protocol.

## Main changes

- No auxiliary subject-ID classifier:

  ```json
  "lambda_ce": 0.0
  ```

- Stronger metric learning:

  ```json
  "lambda_triplet": 1.6,
  "triplet_margin": 0.35,
  "temperature": 0.06
  ```

- Less frequent generative reconstruction after warmup:

  ```json
  "generative_step_interval": 6
  ```

- Multi-gallery Rank-1 evaluation:

  ```json
  "eval_gallery_per_subject": 3
  ```

## Why multi-gallery?

The earlier Rank-1 metric used one normal sequence as the gallery for each test subject. If that single reference sequence has an unlucky view or noisy skeleton, the probe can be punished even when the embedding is generally good.

V5 uses up to three normal gallery sequences per subject and scores each subject by the best matching gallery. This is a legitimate retrieval/verification protocol, but it must be reported clearly because it is not the same as single-gallery Rank-1.

## Run

```bash
modal deploy modal_app.py
python submit_modal.py run --design skeleton_rank1_v5 --run rank1_multigallery_001
```

## How to compare

Use this comparison:

| Design | Purpose | Gallery protocol |
|---|---|---|
| `skeleton_contrastive_v3` | verification baseline | single-gallery |
| `skeleton_rank1_v4` | failed classifier-heavy Rank-1 attempt | single-gallery |
| `skeleton_rank1_v5` | metric-heavy Rank-1 attempt | 3-gallery |

Do not directly claim V5 is better than V3 unless the gallery protocol is stated.

