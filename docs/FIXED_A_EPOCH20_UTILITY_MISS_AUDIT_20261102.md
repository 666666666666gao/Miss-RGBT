# Fixed A Utility Epoch 20 Miss Audit

Completed on server28 at 2026-11-02 05:33 CST.

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
- Checkpoint: `/home/gaob/Miss-RGBT/runs/server28-recovery-20261031/a_utility_full20_seed2027/epoch20.pth.tar`.
- Checkpoint SHA256: `19c008d03f261ebda37c89eff63a0b959bcb6795c0da548e761ef12c91878170`.

## Metrics

| Dataset | PR | NPR | SR | MPR | MSR |
|---|---:|---:|---:|---:|---:|
| LasHeR-Miss | 68.569074 | 63.435064 | 54.347391 | n/a | n/a |
| RGBT234-Miss | n/a | n/a | n/a | 84.703935 | 60.637694 |

This result does not update any current per-metric maximum. It is a full-set
comparison candidate, not a selected checkpoint. No val97 result was used.
