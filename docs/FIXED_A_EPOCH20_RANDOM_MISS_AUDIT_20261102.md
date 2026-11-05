# Fixed A Random Epoch 20 Miss Audit

Completed on server28 at 2026-11-02 05:05 CST.

## Scope and integrity

- Condition: `miss`; `limit=null`.
- Full official scope: 479 sequences and 337,353 frames.
- LasHeR-Miss: 245 sequences and 220,703 frames.
- RGBT234-Miss: 234 sequences and 116,650 frames.
- `run.json` and `completed.json` agree on sequence and frame counts.
- All 479 sequence receipts, bbox hashes, and GT/mask hashes were independently checked.
- All runtime source hashes and standalone evaluator hash match the run manifest.
- Official LasHeR and RGBT234 metrics and curves were recalculated from saved predictions and match the recorded artifacts.
- Verifier SHA256: `63151883bd58736c5c8a4882638e624ac1dc065e5bd3e4993f3383fe74307577`.
- Evaluator SHA256: `bc45da1293bb2e4735a3ac79cf7c7bb5fb195b7ef9069744b2ddd891eae307b1`.
- Checkpoint: `/home/gaob/Miss-RGBT/runs/server28-recovery-20261031/a_random_full20_seed2027/epoch20.pth.tar`.
- Checkpoint SHA256: `6f014866e45393edd0c32594a916d1b0e677f1c6bea8f2d898daa7938f4e5e39`.

## Metrics

| Dataset | PR | NPR | SR | MPR | MSR |
|---|---:|---:|---:|---:|---:|
| LasHeR-Miss | 67.823154 | 62.731542 | 53.736184 | n/a | n/a |
| RGBT234-Miss | n/a | n/a | n/a | 86.184864 | 61.418962 |

This establishes a new audited RGBT234-Miss MPR maximum, +0.111677 points
over Fixed A Random epoch19. It does not change the RGBT234-Miss MSR maximum
of 61.660594 (Fixed A Random epoch19). Against original CADTrack E0 Miss,
the RGBT234 gains are +3.390049 MPR and +1.844527 MSR.

The best LasHeR-Miss scores remain Batch10 Utility epoch17 at PR/NPR/SR
70.397396/65.127562/55.789882. This epoch20 result is below those values and
does not clear the 69.3 PR or 54.4 SR target. The current RGBT234-Miss MPR
and MSR maxima remain 1.915136 and 2.639406 points below the requested
88.1/64.3 thresholds.

This remains a final-five comparison candidate, not a selected model.
Official-test checkpoint selection is post-hoc and test-selection-biased.
No val97 result was used.
