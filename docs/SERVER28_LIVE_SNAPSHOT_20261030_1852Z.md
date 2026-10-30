# Server 28 Live Snapshot

Observed 2026-10-30 18:52 UTC (2026-10-31 02:52 server local time).

## GPU policy and load

All four RTX 3090 GPUs report a target temperature of 85 C and a 250 W power limit. At the observation, current temperatures were 84/80/79/75 C for GPUs 0-3. GPU0-3 remain allocated to the existing jobs; do not launch another training run.

## Training

- Fixed utility/random full20 runs remain live on GPUs 0/1. Latest log records observed during this check were update 10200 and 10600 respectively, epoch 5; recorded loss fields were finite. The evaluation controller remains in `WAITING_FOR_TRAINING`.
- Batch10 utility/random candidates remain live on GPUs 2/3. Latest log records were update 4250 and 4375, epoch 3; recorded losses were finite, with peak allocated memory 21,118,406,144 bytes per process. Both have completed only epoch1/2 task-loss validations; epoch3 validation has not yet been recorded.
- Candidate validation uses a fixed 128-sample task-loss proxy. These losses are not tracking metrics or convergence proof. Full val97 tracking selection among the final five epochs remains pending.

## Historical Miss evaluation

Historical A utility20 LasHeR-Miss evaluation remains live as PID 4138772 with one worker on each of GPUs 0 and 1. Four of 479 sequence receipts were present: `blkboy`shead`, `boyshead9684`, `midredboy`, and `broom`. The latest completed sequence log was `broom` (9,538 frames, 978.92 seconds). No full evaluation metrics are available yet. The earlier four-worker single-GPU OOM attempt remains preserved separately.

## Interpretation

This is a progress/health snapshot only. No benchmark improvement, convergence, or validated new module is established. Preserve all active jobs and continue polling at 180-300 second intervals, checking nearer the next candidate epoch boundaries (estimated around 20:16-20:26 UTC from the 18:49 launch receipt).
