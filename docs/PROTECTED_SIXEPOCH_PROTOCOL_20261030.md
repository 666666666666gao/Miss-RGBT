# Frozen Baseline Six-Epoch CPU Pilot

This extends the eight-update LocalRepair control, which showed only tiny,
modality-dependent validation changes. It tests whether a longer branch-only
budget merits a subsequent GPU campaign. Existing LocalRepair is an ordinary
control, not a new contribution.

## Fixed Protocol

- CPU only, four Torch threads. No GPU or frozen-runtime modification.
- Original E0 epoch20 checkpoint; baseline parameters frozen and all modules
  in eval mode, including BN. Only the 398466 LocalRepair parameters train.
- AdamW LR1e-4, decay1e-4, gradient clipping0.1, effective batch4.
- Train882 sequence order: indices divisible by5 form development; the other
  indices form training. Both are asserted disjoint from val97.
- Six fixed pilot epochs. Each epoch freshly samples64 training tuples using
  seed20270+epoch and stores their input hashes. Both search-outage directions
  see the same effective batches; templates remain complete. No patch mask.
  Each epoch has32 optimizer updates,192 total.
- Development16 tuples are cached once with seed27182, using validation
  transforms on the train882 development sequence partition. Evaluate11/01/10
  after each epoch. No stopping or parameter tuning from these scores.
- Save repair snapshots for epochs2..6. Select the lowest development mean
  task loss over01/10; exact ties select the earlier epoch.
- Val97 has32 cached tuples, seed31415. Initial losses are measured privately
  before optimization; heldout losses are reported only after development
  selection and never affect training or selection.
- Verify complete11 prediction-map/box hashes against pretraining outputs,
  unchanged baseline parameter/buffer hash, absent baseline gradients, and
  finite training loss/norms. These checks cover the sampled complete inputs;
  they do not certify tracking after a missing dynamic-template history.
- After all five development records and final heldout checks pass, retain
  the selected small repair snapshot and delete the other four with size/SHA
  receipts. No historical or live GPU training checkpoints are in this scope.

## Interpretation and Timing

Report raw per-state task losses and selected-epoch deltas. These are sampled
supervised losses, not PR/SR, tracking recovery, convergence, or benchmark gains.
A positive result still requires full GPU training and closed-loop validation.
A negative result is a failed control, not grounds to tune on heldout val97.

Estimated runtime45-75 minutes based on the earlier CPU8-update control;
replace the estimate with actual update throughput after launch. Poll near
the estimated milestone at180-300 second intervals.

Output: `/home/gaob/Miss-RGBT/reports/server28-protected-sixepoch-20261030`.
Log: the sibling `.log` file. Launch through `launch_protected_sixepoch28.py`;
the exclusive output/log creation prevents accidentally overwriting this run.

## Verified Launch

PID61899 started2026-10-30T19:31:47Z. At19:36:17Z the same process was live,
CPU377%, RSS3.6GB, and had completed3 finite updates with finite gradient
norms. Updates1/2/3 used availability01/10/01 and task losses
5.872172/2.717501/4.297597. Update2/3 intervals were9.06/8.53 seconds.
The initial reference/sample stage took approximately4 minutes; this is
expected computation before the optimizer updates.

The persisted input manifest confirms705 training,177 development and97
val sequences with zero pairwise intersections,8 development batches and
16 heldout batches at batch2. These are sampled-tuple controls; E0 pretraining
exposure to val97 remains uncertified, so this does not establish an entirely
unseen benchmark evaluation. Main GPU training/evaluation processes remained
live. The next combined observation is planned near20:11 UTC, with180-300
second checks near actual completion or epoch boundaries.

## First Development Observation

At2026-10-30T19:46:15Z PID61899 was live and update59 of192 was
recorded in epoch2. Epoch1 completed32 updates; fixed development task
losses11/01/10 were2.9881724119/4.6337577999/3.4688829631, with missing-state
mean4.0513203815. Complete-output SHA256 matched the pretraining reference:
e07039e0653cf7cad9104fcf69275a5331683e3614b26961f755efd1955e2f50.
The initial loss reference remains private until development selection.
These absolute losses therefore establish neither improvement nor regression.
Epoch1 is outside the registered final-five selection window.

All four original GPU trainers, both candidate controllers, and historical
Miss evaluator4138772 remained live. GPU0..3 temperatures were83/80/78/76C,
power caps250W, memory21607/21575/22710/22710MiB, utilization100/96/100/100%.
No additional GPU job was started. Current CPU throughput supports the
next combined observation near20:11UTC; final heldout checking may extend
completion to approximately20:20UTC. Neither convergence nor the benchmark
targets are established.

## Completed Result

PID61899 exited normally after192 optimizer updates in2584.97 seconds.
Development histories for epochs1..6 are:

| Epoch | Updates | Mean missing-state task loss |
|---:|---:|---:|
| 1 | 32 | 4.051320 |
| 2 | 64 | 3.967925 |
| 3 | 96 | 3.912870 |
| 4 | 128 | 4.015621 |
| 5 | 160 | 3.957024 |
| 6 | 192 | 3.917524 |

The registered final-five window was epochs2..6. Epoch3 was selected,
with development score3.912870 versus its private initial development
reference4.115627. Selected branch SHA256:
`6730fc3ceacddd2316de175d1cc2e60724ce23a43fa441215b9bbd678fcfdc8d`.
Its file is1,597,391 bytes. Exactly one branch checkpoint remains;
the other four final-window snapshots have per-file size/SHA cleanup
receipts and are absent. Baseline state remained unchanged, baseline
gradients were absent, and sampled complete11 outputs were bitwise
identical.

After the registered development selection, the32 fixed val97 heldout
tuples showed task-loss deltas11/01/10 of0/-0.188141/-0.363297. These are
not PR/SR or closed-loop metrics; historical E0 exposure to val97 also
remains uncertified. Do not use them to tune the branch or claim a
benchmark gain. The fixed CPU tracking control is now running on
train882 development sequences and retains real template availability
history.

Authoritative artifacts:

- `reports/server28-protected-sixepoch-20261030/report.json`
- `reports/server28-protected-sixepoch-20261030/development_history.json`
- `reports/server28-protected-sixepoch-20261030/completed.json`
- `reports/server28-protected-sixepoch-20261030/repair_epoch03.pth`
