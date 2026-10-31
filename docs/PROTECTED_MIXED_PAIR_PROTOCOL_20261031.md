# Matched Dynamic-Template Coverage Pilot

The historical CPU LocalRepair pilot used complete templates, while the
fixed8 tracking audit measured missing dynamic templates on24.94% of
observed frames and5.74% of complete-search frames. The four-tuple
history diagnostic found mixed complete-search changes. Full epoch5
candidate results also do not establish improvement in restored11
outage penalties. These are reasons to test template coverage, not
evidence of template harm or a novel module.

## Registered Comparison

One sequential CPU process with four Torch threads, no CUDA allocation.
Use the existing unchanged Python3.7/Torch1.12 environment. Freeze the
E0 baseline, including all buffers/BN in eval mode. Train only the
existing398466 LocalRepair parameters at layer6, width128. No new module,
loss, tracker rule, dependency, or frozen-runtime edit.

Two freshly initialized arms have the same seed2027, same E0 checkpoint,
same initial repair parameters, same training tuples and optimization:

- `complete_templates`: both templates are always complete.
- `mixed_templates`: the initial template remains complete; dynamic
  template state11/01/10 is sampled independently per tuple using the
  existing formal-training probabilities0.4/0.3/0.3.

The mask generator uses a local seed50000+epoch so sampling masks cannot
change the shared data RNG. Both search outage directions see exactly the
same effective batch and dynamic masks before moving to the next batch.
No spatial patch dropout. Complete-search/missing-template cells are
measured, not optimized with an extra objective.

Replay the prior six-epoch input manifests exactly:705 training sequences,
177 sequence-disjoint train882 development sequences,97 untouched val
sequence names. Development uses16 fixed cached tuples, seed27182.
Each epoch samples64 train tuples, seed20270+epoch; their image/box hashes
must match the previous pilot. Microbatch2, accumulation2, effective4,
AdamW LR1e-4, decay1e-4, norm clipping0.1,32updates/epoch,192/arm.
This is384 updates total, not a full GPU training campaign.

## Selection And Checks

Evaluate all nine dynamic-template11/01/10 x search11/01/10 cells on
the same16 development tuples before training and after each epoch.
For BOTH arms select the smallest equally averaged six-cell missing-search
task loss among epochs2..6; exact ties choose earlier. This avoids
comparing differently selected historical and new checkpoints.

The three complete-search cells are descriptive outcomes; they do not
change training or selection. Verify equal initial results across arms,
unchanged baseline tensors, absent baseline gradients, finite losses/norms,
and bitwise full-history11_11 outputs. These checks do not assert
complete-search equality after missing-template histories.

Retain the selected small branch file per arm and remove exactly its
other four final-window snapshots only after selected-score replay and
baseline checks. Store file hashes/sizes and cleanup receipts. No existing
checkpoints or running GPU jobs enter this cleanup.

Do not evaluate val97/official metrics or select parameters from previous
val97 failures. A small paired task-loss result still needs closed-loop
tracking before deciding whether full protected training is justified.
One seed and reused development tuples cannot establish generalization
or supply any of the three claimed paper contributions.

## Timing And Artifacts

Prior192-update CPU pilot took43.1minutes with simpler development
measurement. This paired run evaluates nine states each epoch; budget
roughly100-160minutes total, to be revised from observed update and
validation times. Initial nine-cell references may take several minutes.
Check once that the actual process is running and producing updates;
then inspect near the first arm completion and final completion, with
180-300second polling near those nodes.

Runner: `probe_protected_mixed_pair28.py`.
Launcher: `launch_protected_mixed_pair28.py`.
Output: `/home/gaob/Miss-RGBT/reports/server28-protected-mixed-pair-20261031`.
Log: sibling `.log`. Exclusive output/log creation prevents overwrite.
Final evidence requires `comparison.json` AND `completed.json`, plus
per-arm selection and cleanup receipts. Do not restart after an SSH timeout.

## Launch

PID869785 started2026-10-31T03:44:23Z (11:44:23 UTC+08).
At11:44:50 the same process was live at347%CPU, RSS1.16GiB and
performing its initial nine-cell development measurements. Runtime source
and E0 checkpoint hashes matched the original pilot. No optimizer updates
were claimed at this initial observation. Four original GPU trainers and
their three controllers were verified live before launch.

Provisional first-arm completion12:35-13:05 UTC+08, both arms13:25-14:25,
pending actual update and nine-cell validation throughput.
Launch receipt: `reports/protected-mixed-pair-launch-20261031.json`.

At11:49:44 UTC+08 the same PID869785 was live at393%CPU, RSS3.22GiB.
The complete-template arm had finished10 real optimizer updates in epoch1;
losses and gradient norms were finite. Update10 loss4.892799616,
gradient norm0.190860614. Recent update intervals7.7-9.1seconds.
The exact prior epoch1 image/box hashes passed before these updates.
Initial nine-cell measurement time explains the pre-update log interval.
This verifies execution, not training benefit. Next useful observation
remains the first-arm completion window, rather than another immediate poll.
