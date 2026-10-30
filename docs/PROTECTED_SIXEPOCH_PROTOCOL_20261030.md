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
