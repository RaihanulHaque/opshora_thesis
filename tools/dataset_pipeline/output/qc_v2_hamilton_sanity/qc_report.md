# CLoP-Gait Dataset QC Report

## Coverage

- total_raw_sequences: `150`
- known_gaps: `3`
- usable_sequences: `147`
- processed: `60`
- not_yet_processed: `87`

## Known/Confirmed Gaps

Pre-existing raw-collection gaps, not fabricated or backfilled.

- `003/nm-01-on/090` (missing) <- `datasets/CLoP-Gait_v2/CLoP-Gait - Copy/OUTDOOR/Outdoor_Night/Sub03_ON/Nm1_03_ON/Nm1_03_ON_90`
- `004/cl-01-on/000` (missing) <- `datasets/CLoP-Gait_v2/CLoP-Gait - Copy/OUTDOOR/Outdoor_Night/Sub04_ON/Lc1_04_ON/Lc1_04_ON_0`
- `004/cl-01-on/180` (missing) <- `datasets/CLoP-Gait_v2/CLoP-Gait - Copy/OUTDOOR/Outdoor_Night/Sub04_ON/Lc1_04_ON/Lc1_04_ON_180`

## Unexpected Failures

Sequences that were attempted but produced nothing usable.

- none

## Low Frame Count (< 30)

- none

## Missing Pairs (silhouette/skeleton mismatch)

- none

## Diagnostics

- Multi-person-detected frames across all processed sequences: `120` (the tracker picks the closest-centroid match; an unexpectedly high count may mean background pedestrians in outdoor footage).
- Not yet processed (no markers found): `87`

