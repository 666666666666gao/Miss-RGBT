# Candidate Evaluator Source Audit

Audited 2026-10-30 19:05 UTC while the candidate training controllers were
still live.

The candidate launch receipt records `evaluate_candidate_official28.py` at
SHA256 `cb9dafa1f39c453ec1dab53321b3184111a38dbb9abd6a8a6d0bfa802aa57190`.
The current server and checked-out `codex/server28-research` version both
hash to `bc45da1293bb2e4735a3ac79cf7c7bb5fb195b7ef9069744b2ddd891eae307b1`.
The file modification time is 2026-10-31 02:26 server-local time. Git history
attributes the change to commit `7085616` (2026-10-30 18:34 UTC), after the
candidate launch.

The commit diff only broadens supported physical GPU selections, adds
`--workers-per-gpu` and `--limit`, and makes subset smoke runs record their
actual frame/sequence counts. It does not change tracker inference, masks,
ground-truth handling, or metric computation. The change was made to recover
from a demonstrated four-worker CUDA OOM while GPU1 was occupied by training.

The live `candidate_campaign28.py` is unchanged from its launch hash and
invokes `evaluate_candidate_official28.py` as a subprocess only after all
five retained checkpoints have completed val97 complete/alternating-outage
validation, the best is selected, and the other four are removed with
receipts. Therefore the post-selection full evaluation will use the current
tracked evaluator. Its `run.json` records the evaluator SHA at execution.
The current candidate validation script hash still matches its launch
receipt. No running process or frozen runtime was modified during this audit.

This is a resolved provenance difference, not evidence of an active
evaluation failure. Continue the current training and do not replace or
restart its files.
