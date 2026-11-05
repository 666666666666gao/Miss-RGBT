# Batch10 Random Epoch 17 Complete Audit

Server28 completed this official complete-input evaluation at 2026-11-02 02:13 CST.

## Scope and integrity

- Input condition: `complete`; `limit=null`.
- Total: 479 sequences and 337,353 frames.
- Benchmark split receipt scope: LasHeR-Miss 245 sequences/220,703 frames and RGBT234-Miss 234 sequences/116,650 frames. The `-Miss` suffix is the benchmark split label, not the evaluated input condition.
- `run.json` and `completed.json` agree.
- All 479 per-sequence receipt JSON files match `all_receipts.json`.
- All 479 bbox SHA256 values match their prediction files.
- All 110 runtime source-file hashes, checkpoint hash, and standalone evaluator hash match the run manifest.
- Checkpoint: `/home/gaob/Miss-RGBT/runs/server28-recovery-20261031/batch10_random_seed2027/epoch17.pth.tar`
- Checkpoint SHA256: `995679a1fbf061da0915984a295596072f2ee277894aa571ba97d63f4011695a`

## Metrics

| Dataset | PR | NPR | SR | MPR | MSR |
|---|---:|---:|---:|---:|---:|
| LasHeR | 75.704048 | 72.171251 | 60.419370 | n/a | n/a |
| RGBT234 | n/a | n/a | n/a | 89.946245 | 65.472600 |

LasHeR NPR and SR are new audited per-metric complete-input maxima, improving prior maxima by 0.122045 and 0.103244 points. LasHeR PR remains below the current maximum 75.751492, and neither RGBT234 metric changes its maximum. Versus original CADTrack E0 complete metrics, LasHeR NPR/SR improve by 0.398414/0.293656 points. The approximately +2 point complete-input objective is not established.

## Interpretation

This is one candidate in the final-five official evaluation campaign, not the selected checkpoint. Selecting among official test scores is post-hoc and introduces test-selection bias. Preserve all candidate weights until the campaign and final-five comparison are complete. No val97 result is used.
