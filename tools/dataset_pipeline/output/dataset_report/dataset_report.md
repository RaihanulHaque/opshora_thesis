# CLoP-Gait Dataset Composition Report

- Subjects: `5`
- Raw sequences discovered: `210` (`205` usable)

## Coverage by domain

| Domain | Subjects | Sequences (ok/total) |
|---|---:|---:|
| indoor | 5 | 75/75 |
| outdoor-day | 4 | 58/60 |
| outdoor-night | 5 | 72/75 |

## Silhouette frame-count distribution

- Overall: n=`205`, mean=`56.7`, median=`56`, min=`35`, max=`90`, stdev=`10.6`
- indoor: n=`75`, mean=`56.1`, min=`38`, max=`82`
- outdoor-day: n=`58`, mean=`48.9`, min=`35`, max=`74`
- outdoor-night: n=`72`, mean=`63.8`, min=`42`, max=`90`

## Skeleton frame-count distribution

- Overall: n=`205`, mean=`56.7`, median=`56`, min=`35`, max=`90`, stdev=`10.6`

## Domain-generalization train/test split

Train = every domain except `od` (indoor + outdoor-night); test = only `od` (outdoor-day) -- same subjects in both, per `gait.dataset.GaitSequenceDataset(split_mode="domain")`.

- Total train sequences: `147`
- Total test sequences: `58`

| Subject | Train | Test |
|---|---:|---:|
| 001 | 30 | 15 |
| 002 | 30 | 14 |
| 003 | 29 | 15 |
| 004 | 28 | 14 |
| 005 | 30 | 0 |

**Note:** subject(s) `005` have zero outdoor-day footage, so they contribute no probe sequences to the test split (still used in training via indoor/outdoor-night).
