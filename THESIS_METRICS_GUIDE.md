# Thesis Metrics Guide

This file explains the training logs produced by the gait experiments, especially the `skeleton_contrastive_v3` design.

## 1. What the V3 model is trying to prove

`skeleton_contrastive_v3` is not mainly a subject-ID classifier. It is a verification and metric-learning model.

The thesis question is:

> When two unseen gait sequences are passed through the network, are sequences from the same person close in the learned embedding space, and are sequences from different people far apart?

So, for V3, the most important metrics are:

```text
same_distance
different_distance
distance_gap
verification_auc
verification_accuracy
```

`rank1` and `rank5` are still useful, but they are secondary retrieval metrics.

## 2. Example V3 result

Recent run:

```json
{
  "epoch": 50,
  "generative_avg": 1.0781,
  "recognition_avg": 1.1323,
  "rank1": 0.4673,
  "rank5": 0.7367,
  "same_distance": 0.2843,
  "different_distance": 0.8430,
  "distance_gap": 0.5588,
  "verification_auc": 0.8842,
  "verification_accuracy": 0.7922,
  "best_verification_auc": 0.8894
}
```

Interpretation:

- The best verification AUC was about **0.889**, or **88.9% pair-separation quality**.
- Same-person pairs had average distance about **0.28**.
- Different-person pairs had average distance about **0.84**.
- The distance gap was about **0.56**, meaning the embedding space clearly separates many same/different pairs.
- Rank-1 was about **46.7%**, so nearest-neighbor identity retrieval is still moderate.
- Rank-5 was about **73.7%**, meaning the correct person often appears in the top five gallery matches.

This is a defensible early thesis result for a blind verification model, but it is not yet a strong closed-set identification result.

## 3. Field-by-field explanation

| Metric | Meaning | Desired direction | Thesis interpretation |
|---|---|---:|---|
| `epoch` | The completed training epoch index. Epoch `50` means the 51st pass because counting starts at zero. | More until validation stops improving | Training progress marker only. |
| `steps` | Number of mini-batches processed in that epoch. | Stable | Confirms how much data was used per epoch. |
| `generative_steps` | Number of mini-batches used for the self-supervised reconstruction branch. | Stable | Shows how often the model learned skeleton/motion reconstruction. |
| `recognition_steps` | Number of mini-batches used for contrastive/triplet metric learning. | Stable | Shows how often the model learned embedding separation. |
| `generative` | Sum of generative reconstruction losses across the epoch. | Lower | Raw total loss; use `generative_avg` for easier comparison. |
| `generative_avg` | Average reconstruction loss per generative step. | Lower | Measures how well masked skeleton/motion frames are reconstructed. |
| `recognition` | Sum of contrastive/triplet losses across the epoch. | Lower | Raw total metric-learning loss; use `recognition_avg` for easier comparison. |
| `recognition_avg` | Average contrastive/triplet loss per recognition step. | Lower | Measures whether embeddings are being pulled/pushed correctly during training. |
| `learning_rate` | Current optimizer step size. | Scheduled downward | In V3 it decreases by cosine schedule, so later epochs make smaller updates. |
| `rank1` | Retrieval accuracy where the nearest gallery sample must be the correct person. | Higher | Useful but secondary for V3; this is strict identity retrieval. |
| `rank5` | Retrieval accuracy where the correct person must appear in the top five gallery matches. | Higher | Shows whether the embedding is close even when not the nearest match. |
| `same_distance` | Average embedding distance between two sequences from the same subject. | Lower | Core verification metric: same person should be close. |
| `different_distance` | Average embedding distance between sequences from different subjects. | Higher | Core verification metric: different people should be far apart. |
| `distance_gap` | `different_distance - same_distance`. | Higher and positive | Direct separation margin between different-person and same-person pairs. |
| `verification_auc` | Probability that a random same-person pair is closer than a random different-person pair. | Higher, max 1.0 | Best single V3 metric for blind verification. |
| `verification_accuracy` | Balanced same/different decision accuracy using a simple threshold. | Higher | Practical verification score if the system must say match/non-match. |
| `monitor_metric` | Metric used for early stopping. | Should match objective | V3 uses `verification_auc`, not `rank1`. |
| `best_verification_auc` | Best validation AUC seen so far. | Higher | The checkpoint `best_model.pt` corresponds to this best monitored score. |
| `best_rank1` | Rank-1 value shown for comparison. | Higher | For V3 this is not the early-stopping target. |

## 4. Why early stopping happened

This log:

```text
Early stopping after 52 epochs; verification_auc did not improve for 12 epochs.
```

means training stopped because the model failed to beat its previous best `verification_auc` by at least the configured minimum improvement for 12 consecutive epochs.

For this run:

```text
best_verification_auc = 0.889426...
```

At epoch 50:

```text
verification_auc = 0.884237...
```

At epoch 51:

```text
verification_auc = 0.875222...
```

So the model was no longer improving. Early stopping protected the best checkpoint and avoided wasting GPU time.

The important file is:

```text
/data/experiments/skeleton_contrastive_v3/<run>/best_model.pt
```

That checkpoint is better than the final epoch if the final epoch's AUC is lower than `best_verification_auc`.

## 5. How to write this in the thesis

Good wording:

> The proposed skeleton-contrastive model achieved a best verification AUC of 0.889 on unseen subjects. The mean embedding distance for same-identity pairs was substantially lower than for different-identity pairs, producing a distance gap of approximately 0.56 near the end of training. This supports the hypothesis that the learned representation clusters gait sequences by identity without requiring explicit closed-set identity prediction at test time.

Careful limitation:

> Although the verification separation was strong, Rank-1 retrieval remained moderate. This suggests that the current representation is more reliable for same/different verification than for strict nearest-neighbor identification, and future work should improve view-invariant retrieval.

Avoid saying:

> The model recognizes every person correctly.

That would be false because Rank-1 is still around 45–47%.

## 6. What result would be considered good?

For this thesis prototype:

| Metric | Weak | Usable | Strong |
|---|---:|---:|---:|
| `verification_auc` | below 0.75 | 0.80–0.90 | above 0.90 |
| `verification_accuracy` | below 0.65 | 0.70–0.82 | above 0.85 |
| `distance_gap` | below 0.20 | 0.40–0.60 | above 0.60 |
| `rank1` | below 0.35 | 0.45–0.65 | above 0.70 |
| `rank5` | below 0.60 | 0.70–0.85 | above 0.90 |

These are practical guide ranges for judging the prototype, not universal scientific thresholds.

## 7. What to try next

The current V3 model already shows useful verification separation. The next experiments should test whether the contribution comes from the proposed design components:

1. Skeleton-only contrastive model without generative reconstruction.
2. Skeleton generative + contrastive model with a slightly lower learning rate.
3. Fused skeleton + silhouette model if skeleton-only retrieval remains weak.
4. View-aware evaluation, because CASIA-B cross-view changes can strongly hurt Rank-1.

Do not discard V3. Preserve this run as a verification baseline.
