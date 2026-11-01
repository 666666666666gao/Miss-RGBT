# Fixed A Random Epoch 19 Miss Audit

Server28 completed this official Miss evaluation at 2026-11-02 01:30 CST.

## Scope and integrity

- Condition: `miss`; `limit=null`.
- Total: 479 sequences and 337,353 frames.
- LasHeR-Miss: 245 sequences and 220,703 frames.
- RGBT234-Miss: 234 sequences and 116,650 frames.
- `run.json` and `completed.json` agree.
- All 479 per-sequence receipt JSON files match `all_receipts.json`.
- All 479 bbox SHA256 values match their prediction files.
- All 110 runtime source-file hashes, checkpoint hash, and standalone evaluator hash match the run manifest.
- Checkpoint: `/home/gaob/Miss-RGBT/runs/server28-recovery-20261031/a_random_full20_seed2027/epoch19.pth.tar`
- Checkpoint SHA256: `0444578e5d157e8801e537a166c83f4973ed0f56846374b3fd139f7da4ffe8e9`

## Metrics

| Dataset | PR | NPR | SR | MPR | MSR |
|---|---:|---:|---:|---:|---:|
| LasHeR-Miss | 68.365216 | 63.171326 | 54.312404 | n/a | n/a |
| RGBT234-Miss | n/a | n/a | n/a | 86.073187 | 61.660594 |

This candidate does not improve the current LasHeR-Miss PR/SR maxima. Its RGBT234-Miss MPR/MSR are the new audited per-metric maxima, improving the preceding maxima by 0.033932/0.206349 points. Against the original CADTrack E0 Miss scores of 82.794815/59.574435, the gains are 3.278372/2.086159 points. The requested 88.1/64.3 thresholds remain short by 2.026813/2.639406 points.

## Interpretation

This is one candidate in the final-five official evaluation campaign, not the selected checkpoint. Comparing checkpoints on official test results is post-hoc and introduces test-selection bias. Keep all candidate weights until the campaign and final-five comparison are complete. No val97 result is used.

At 2026-11-02 01:37 CST the remaining active official evaluations were Fixed A Utility epoch19 Miss (275/479 receipts), Batch10 Random/Utility epoch17 Complete (145/479 and 271/479), and Fixed A Random epoch20 Complete (0/479). Receipt counts are operational progress, not metrics.
