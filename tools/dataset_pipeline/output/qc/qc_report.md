# CLoP-Gait Dataset QC Report

## Coverage

- total_raw_sequences: `450`
- known_gaps: `1`
- usable_sequences: `449`
- processed: `9`
- not_yet_processed: `440`

## Known/Confirmed Gaps

Pre-existing raw-collection gaps, not fabricated or backfilled.

- `004/cl-01-od/090` (missing) <- `datasets/CLoP-Gait/OUTDOOR/Outdoor_Day/Sub04_OD/Lc1_04_OD/Lc1_04_OD_90`

## Unexpected Failures

Sequences that were attempted but produced nothing usable.

- none

## Low Frame Count (< 30)

- `001/nm-01-on/000`: 26 frames

## Missing Pairs (silhouette/skeleton mismatch)

- none

## Diagnostics

- Multi-person-detected frames across all processed sequences: `34` (the tracker picks the closest-centroid match; an unexpectedly high count may mean background pedestrians in outdoor footage).
- Not yet processed (no markers found): `440`

