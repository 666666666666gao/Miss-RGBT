# Paired Official Metrics

Recorded November 2, 2026. Scores are percentages. Each row contains all
five metrics from one checkpoint and one full input condition. These are
historical comparison candidates, not a new checkpoint selection.

Each condition covers LasHeR 245 sequences / 220,703 frames and RGBT234
234 sequences / 116,650 frames, totaling 479 sequences / 337,353 frames.
No val97 or partial benchmark result is included.

| Checkpoint | Condition | LasHeR PR | LasHeR NPR | LasHeR SR | RGBT234 MPR | RGBT234 MSR |
|---|---|---:|---:|---:|---:|---:|
| E0 epoch20 | Complete | 75.636503 | 71.772837 | 60.125714 | 90.677134 | 66.845228 |
| E0 epoch20 | Miss | 63.472507 | 58.581851 | 50.583149 | 82.794815 | 59.574435 |
| Batch10 Utility epoch17 | Complete | 75.751492 | 71.965679 | 60.195400 | 89.839668 | 65.314284 |
| Batch10 Utility epoch17 | Miss | 70.397396 | 65.127562 | 55.789882 | 85.097378 | 60.685934 |
| Batch10 Random epoch17 | Complete | 75.704048 | 72.171251 | 60.419370 | 89.946245 | 65.472600 |
| Batch10 Random epoch17 | Miss | 68.695416 | 63.738912 | 54.710283 | 85.859985 | 61.231719 |
| Fixed A Random epoch20 | Complete | 74.141654 | 70.468525 | 59.185266 | 89.960694 | 65.247968 |
| Fixed A Random epoch20 | Miss | 67.823154 | 62.731542 | 53.736184 | 86.184864 | 61.418962 |

## Observed Tradeoffs

Batch10 Utility epoch17 passes both LasHeR-Miss targets, but its RGBT234-Miss
MPR/MSR remain below 88.1/64.3. Fixed A Random epoch20 has higher
RGBT234-Miss MPR/MSR than that candidate, but fails both LasHeR-Miss targets.
None of these checkpoints meets all four requested Miss targets.

Complete-input RGBT234 MSR is lower than E0 for all three listed candidates.
The roughly two-point complete-input improvement is not established.
These observed tradeoffs do not identify a mechanism or validate a module.
Do not infer cross-dataset robustness from numerical comparisons between
LasHeR PR/SR and RGBT234 MPR/MSR, which use different metric aggregations.

The separate per-metric maxima in FINAL5_OFFICIAL_RECALC_AUDIT_20261102.md
include Fixed A Random epoch16 MPR and epoch19 Miss MSR. They must not be
combined into a score attributed to any checkpoint in this table.
All forty final-five evaluations completed and passed the November 2
05:46 CST full recalculation audit. Older individual audit notes describe
queue status at their own timestamps.

## Evidence

- Baseline: reports/METRICS_SUMMARY.csv, E0_epoch20 rows.
- docs/BATCH10_UTILITY_EPOCH17_COMPLETE_AUDIT_20261102.md.
- docs/BATCH10_UTILITY_EPOCH17_MISS_AUDIT_20261102.md.
- docs/BATCH10_RANDOM_EPOCH17_COMPLETE_AUDIT_20261102.md.
- docs/ADDITIONAL_FINAL5_RESULT_AUDITS_20261102.md.
- docs/FIXED_A_EPOCH20_RANDOM_MISS_AUDIT_20261102.md.
- docs/FINAL5_OFFICIAL_RECALC_AUDIT_20261102.md.

The seed2028 partitioned replication is separate and still training.
It contributes no scores to this table. No training, evaluation,
checkpoint-selection or retention policy was changed for this report.
