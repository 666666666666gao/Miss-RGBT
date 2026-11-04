# Actual A Official Completion And Audit

## Corrected Artifact Interpretation

The official queue does not write
`reports/server28_partitioned_a_20261102_official_queue_v2/verification_v2.json`.
Its actual terminal artifact is `completed.json`, containing both arm
completion records and20 independently recomputed full evaluation rows.
The existing `verification_v2.json` is under the formal training campaign
and has `status=passed, phase=formal`. Earlier monitoring descriptions that
implied a separate queue `verification_v2.json` would eventually appear
were inaccurate. Its absence must not cause an indefinite wait.

The training verification alone is also insufficient: both arm official
completion receipts and the queue terminal receipt must exist and agree.

## Read-Only Terminal Audit

`scripts/verify_partitioned_a_official.py` now audits those actual artifacts.
It returns `WAITING_FOR_A_OFFICIAL_COMPLETION` before terminal evidence is
present, without scoring, creating an audit file or launching any process.
When complete, it:

- Checks the original queue source manifest and formal training verification.
- Binds both training completion receipts, verifies final-five development
  selections and confirms every saved epoch checkpoint still exists.
- Checks both arm job plans, cached verified rows, exact terminal copies,
  successful subprocess exit receipts and all evaluated checkpoint hashes.
- Calls the existing full-dataset verifier for every one of the20 results.
  That verifier checks ground truth, masks, inventories, boxes, receipts,
  saved curves and frozen evaluator/runtime sources before recomputing scores.
- Preserves Complete/Miss checkpoint pairing and the registered development
  selections, returning `status=passed, phase=formal_official_queue` only
  after20 full479-sequence/337353-frame audits pass.

The result is an audit, not permission to bypass any R/P/M1 or module gate.
`dispatch_authorized` remains false. Its optional output is created only on
success, with exclusive creation; it does not overwrite queue artifacts.

## CPU Checks And Current State

Five unittest methods passed in the server's existing Python3.7 environment.
Fixture checks exercise all20 outer audit calls, preserved selections,
missing terminal evidence, changed jobs/metrics/selection, failed exits,
changed sources/training receipts and missing retained weights.
The scoring function is mocked and weights/history are fixture data, so
these tests are not an actual completed A audit.

Files are staged at `/tmp/codex_a_official_audit_20261104.keD6BN/scripts/`:

| File | SHA256 |
| --- | --- |
| verify_partitioned_a_official.py | 0fb6420ec2625c65afc350068ef8cd25588058adbb898b68726398c3967439fa |
| check_partitioned_a_official.py | 74f0e1e15229ff88abc14a25a076d4e79b5a9e38c0376c841ae08332c9cd749b |

The actual probe at2026-11-04 11:29:33UTC returned waiting on exactly
the queue `completed.json` and `utility_completed.json`. Random's arm
receipt is complete. No successful audit output was created.

For the eventual actual audit, first verify staging identities and run from
the frozen A runtime:

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
PYTHONPATH=/tmp/codex_a_official_audit_20261104.keD6BN:/home/gaob/Miss-RGBT/runtime/task_utility_20261018:/home/gaob/Miss-RGBT \
/home/gaob/Miss-RGBT/conda-envs/miss_rgbt/bin/python \
-m scripts.verify_partitioned_a_official \
--output /tmp/codex_a_official_audit_20261104.keD6BN/formal_official_audit.json
```

An unfinished probe is read-only. Once successful, retain the audit and
its evidence bindings; do not infer success from an output filename alone.
No existing trainer, queue source, config or checkpoint was modified.

## Available Selected-Checkpoint Comparison

The already completed full selected-checkpoint results below were read from
the controllers' independently recomputed rows at11:31UTC. Utility uses its
registered epoch14; Random uses epoch15. Each model's Complete/Miss results
use one checkpoint. This is not the final20-evaluation queue audit.

| Condition and metrics | Utility14 | Random15 |
| --- | --- | --- |
| LasHeR Complete PR/NPR/SR | 74.159/70.543/59.154 | 73.602/70.003/58.641 |
| RGBT234 Complete MPR/MSR | 87.474/63.973 | 88.415/64.549 |
| LasHeR Miss PR/NPR/SR | 67.174/61.863/53.313 | 67.891/62.762/53.761 |
| RGBT234 Miss MPR/MSR | 84.736/60.410 | 83.891/59.847 |

Utility improves Complete LasHeR and Miss RGBT234 relative to Random, but
loses on Miss LasHeR and Complete RGBT234. Neither selected model clears
all requested Miss thresholds. This is one paired seed with registered
early stopping; trajectories ended at17 versus18 epochs and selected
checkpoints differ. It is not a fixed-budget, multi-seed mechanism proof.
Do not replace these selected weights with per-metric official maxima.

The existing five-minute automation remains active with the same target
thread and cadence. Its prompt now names the actual artifacts and the
read-only audit rather than an unproduced queue verification file.

## Actual Terminal Audit Passed

The original parent queue and both official arm controllers finished.
The actual read-only audit ran at11:46UTC and was confirmed successful
at11:58UTC on2026-11-04. It returned `status=passed`,
`phase=formal_official_queue`, with20 recomputed full evaluations,
9,580 trajectories and6,747,060 frames. Each evaluation has the required
479 sequences/337,353 frames. Both training receipts, internal selections,
all evaluated weights and all produced checkpoints passed the checks.

The raw result is retained locally as
`reports/server28_partitioned_a_official_terminal_audit_20261104.json`
and remotely at the same filename under `/home/gaob/Miss-RGBT/reports/`.
SHA256: `ab0316642d9aeb92784a541e7142b524c3e043f5d60340afac23b2a76f8c826f`.
Do not rerun the completed20-result audit every heartbeat or overwrite it.

The selected-checkpoint table above is now included in the complete audit.
Utility14 and Random15 remain selected. Neither meets all four Miss
thresholds, their relative wins are mixed, and this one paired seed with
unequal early stopping is not three-module efficacy evidence.
All checkpoints remain retained.

Passing A permits eligible independent preparation; it does not change
the failed R1-selected Complete condition or authorize the old P queue.
The independent R3 pair was actually prepared and source/config read-back
verified at12:00:41UTC, but independent review and both real-image smokes
remain unfinished. No R3 training was launched.

The existing five-minute automation was updated through the app to retain
the passed audit, avoid redundant full re-audits and track the actual R3
preparation status. Its thread, active status and quiet-unless-meaningful
notification policy remain unchanged.
