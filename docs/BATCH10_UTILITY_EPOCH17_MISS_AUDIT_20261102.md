# Batch10 Utility Epoch 17 Miss Audit

Completed on server28 at 2026-11-02 03:31 CST.

## Scope and integrity

- Condition: `miss`; `limit=null`.
- Full official scope: 479 sequences and 337,353 frames.
- LasHeR-Miss: 245 sequences and 220,703 frames.
- RGBT234-Miss: 234 sequences and 116,650 frames.
- `run.json` and `completed.json` agree on full sequence/frame counts.
- All 479 sequence receipts and bbox hashes were independently checked.
- The independent verifier rechecked GT and mask hashes, all runtime source hashes, the evaluator hash, and recalculated official benchmark metrics and curves from saved trajectories.
- Verifier: `verify_recovery_campaign28.py`, SHA256 `63151883bd58736c5c8a4882638e624ac1dc065e5bd3e4993f3383fe74307577`.
- Evaluator SHA256: `bc45da1293bb2e4735a3ac79cf7c7bb5fb195b7ef9069744b2ddd891eae307b1`.
- Checkpoint: `/home/gaob/Miss-RGBT/runs/server28-recovery-20261031/batch10_utility_seed2027/epoch17.pth.tar`.
- Checkpoint SHA256: `fbeaf71a9b4bbfb46e926251b34c7fb1960c4ba41d3da75745342638fd4dc062`.

## Metrics

| Dataset | PR | NPR | SR | MPR | MSR |
|---|---:|---:|---:|---:|---:|
| LasHeR-Miss | 70.397396 | 65.127562 | 55.789882 | n/a | n/a |
| RGBT234-Miss | n/a | n/a | n/a | 85.097378 | 60.685934 |

This sets new audited LasHeR-Miss per-metric maxima. Relative to original
CADTrack E0 Miss, the changes are +6.924889 PR, +6.545711 NPR, and +5.206733
SR. The requested LasHeR-Miss PR/SR thresholds (69.3/54.4) are exceeded by
1.097396/1.389882 points.

This result does not improve the RGBT234-Miss maxima, currently
86.073187/61.660594 from Fixed A Random epoch19. Those remain
2.026813/2.639406 points below the requested 88.1/64.3 thresholds.

This is not a final model selection; other final-five evaluations remain in
progress. Do not select a model or clean checkpoints from these test results;
comparisons across official test checkpoints are post-hoc and
test-selection-biased. No val97 result was used.
