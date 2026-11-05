# Additional Final-Five Result Audits

Completed on server28 by 2026-11-02 04:12 CST.

## Independent verification

These runs were independently checked with `verify_official` from
`verify_recovery_campaign28.py`. Verification requires `limit=null`, matching
run/completion scope, 479 unique sequences and 337,353 frames; checks
per-sequence receipt identities, bbox and ground-truth hashes, runtime source
and evaluator hashes; then recomputes official metrics and curves from the
saved predictions.

- Verifier SHA256: `63151883bd58736c5c8a4882638e624ac1dc065e5bd3e4993f3383fe74307577`.
- Evaluator SHA256: `bc45da1293bb2e4735a3ac79cf7c7bb5fb195b7ef9069744b2ddd891eae307b1`.
- Scope: 245 LasHeR sequences/220,703 frames and 234 RGBT234 sequences/116,650 frames for each run.

## Results

| Candidate | Condition | LasHeR PR/NPR/SR | RGBT234 MPR/MSR | Checkpoint SHA256 |
|---|---|---:|---:|---|
| Fixed A Random epoch20 | Complete | 74.141654 / 70.468525 / 59.185266 | 89.960694 / 65.247968 | `6f014866e45393edd0c32594a916d1b0e677f1c6bea8f2d898daa7938f4e5e39` |
| Fixed A Utility epoch20 | Complete | 74.434070 / 70.701109 / 59.488002 | 90.566223 / 65.948372 | `19c008d03f261ebda37c89eff63a0b959bcb6795c0da548e761ef12c91878170` |
| Batch10 Random epoch17 | Miss | 68.695416 / 63.738912 / 54.710283 | 85.859985 / 61.231719 | `995679a1fbf061da0915984a295596072f2ee277894aa571ba97d63f4011695a` |

None of these results changes the current per-metric maxima. They are
official full-set results, not partial scores, and remain comparison
candidates rather than selected checkpoints. The two Fixed A epoch20 Miss
evaluations remain in progress; retain all candidate weights until every
registered run and audit is complete. No val97 result was used.
