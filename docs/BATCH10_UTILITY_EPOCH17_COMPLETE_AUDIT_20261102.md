# Batch10 Utility Epoch 17 Complete Audit

Server28 completed this official complete-input evaluation at 2026-11-02 01:50 CST.

## Scope and integrity

- Input condition: `complete`; `limit=null`.
- Total: 479 sequences and 337,353 frames.
- Benchmark split receipts: LasHeR-Miss 245 sequences/220,703 frames and RGBT234-Miss 234 sequences/116,650 frames. The `-Miss` suffix identifies the benchmark split; it does not mean the evaluated input condition was Miss.
- `run.json` and `completed.json` agree.
- All 479 per-sequence receipt JSON files match `all_receipts.json`.
- All 479 bbox SHA256 values match their prediction files.
- All 110 runtime source-file hashes, checkpoint hash, and standalone evaluator hash match the run manifest.
- Checkpoint: `/home/gaob/Miss-RGBT/runs/server28-recovery-20261031/batch10_utility_seed2027/epoch17.pth.tar`
- Checkpoint SHA256: `fbeaf71a9b4bbfb46e926251b34c7fb1960c4ba41d3da75745342638fd4dc062`

## Metrics

| Dataset | PR | NPR | SR | MPR | MSR |
|---|---:|---:|---:|---:|---:|
| LasHeR | 75.751492 | 71.965679 | 60.195400 | n/a | n/a |
| RGBT234 | n/a | n/a | n/a | 89.839668 | 65.314284 |

LasHeR PR is a new audited per-metric complete-input maximum, improving the prior maximum by 0.291668 points. Compared with original CADTrack E0 LasHeR PR 75.636503, this is a gain of 0.114989 points. This candidate does not update LasHeR NPR/SR or either RGBT234 complete-input maximum. The roughly +2 point complete-input improvement objective is not established.

## Interpretation

This is one candidate in the final-five official evaluation campaign, not the selected checkpoint. Selecting among official test scores is post-hoc and introduces test-selection bias. Preserve all candidate weights until the campaign and final-five comparison are complete. No val97 result is used.
