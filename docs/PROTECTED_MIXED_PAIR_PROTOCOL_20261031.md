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

## Independent Receipt Analysis

`analyze_protected_mixed_pair28.py` runs only after both arms finish.
It checks the final comparison hash, unchanged source/input manifest hashes,
all six replayed training manifests per arm, finite16-tuple cell losses,
nine-cell means and six-cell selection scores, exact final-five selection,
retained weight hashes, and removal of exactly the other four snapshots.
It also checks equal initial results and the reported full-history invariant.
Baseline tensor equality is a training receipt, not an independent tensor
re-execution by this analyzer.

The analysis reports each selected arm's nine cell losses and mixed-minus-
complete deltas. Selection reused these development tuples, so these numbers
are descriptive selected development outcomes, not unbiased tracking effects.
It creates `analysis.json` exclusively after all final receipts pass.
At12:36 UTC+08 the analyzer's measurement checks passed on actual initial
and available complete-template epoch records. The full final audit has
not yet run. At12:37:29 the original CPU PID869785 remained live after
update192, performing final development/selection work.

## Paired Progress, Not Final Results

At13:06:20 UTC+08 both initial.json objects were independently compared
and were exactly equal, including baseline/repair hashes and initial
nine-cell results. Both arms have completed epochs1/2 on identical inputs.
The complete-template arm is already finished; mixed-template training
remains active, with epoch3/update88 observed at13:05:32.

At epoch1 the six-cell selection loss is4.506485711 complete-template
versus4.503826366 mixed-template (difference-0.002659345).
At epoch2 it is4.427691171 versus4.432939143 (+0.005247972).
Per-cell changes have mixed signs. Complete-history11_11 remains exactly
equal; missing-template/complete-search cell differences at epoch2 are
+0.000251725 and+0.000318335. These are small descriptive task-loss
changes on reused development tuples, not tracking evidence or final
checkpoint comparisons. They do not establish consistent improvement
or harm, and do not justify promoting this control to a full GPU campaign.

Receipt: reports/protected-template-paired-progress-20261031T0506.json.
The existing six-epoch protocol and final-five selection remain unchanged.
Analysis controller1020925 will run the independent final paired audit
after CPU869785 exits and completion artifacts exist.

## Completed Result

The earlier live entries above are historical. Both arms completed192
updates, selectedepoch6 within the registered final five epochs2..6, replayed
that selection, retained one small branch checkpoint and removed the other
four with receipts. The separate analyzer verified384total updates,
common initialization, replayed input hashes, all nine-cell loss means,
exact selection scores, retained hashes and deleted-file absence.

| Training template coverage | Selected six-cell mean task loss |
|---|---:|
| Complete | 4.363575024840732 |
| Mixed11/01/10 | 4.355297873883198 |

Mixed minus complete is-0.008277150957534, approximately-0.19%.
Only3of6missing-search cells improve;3worsen. Fully complete11_11
outputs remain identical. Selected development tuples are reused in these
comparisons; the numbers do not prove independent generalization, tracking
improvement, convergence or a new module. Do not promote this small mixed
effect alone into a larger GPU training campaign.

| Dynamic/search | Complete-template training loss | Mixed-template training loss | Mixed minus complete |
|---|---:|---:|---:|
| 11/11 | 2.988172 | 2.988172 | 0 |
| 11/01 | 4.534665 | 4.564351 | +0.029686 |
| 11/10 | 3.300384 | 3.292627 | -0.007758 |
| 01/11 | 3.328026 | 3.326445 | -0.001581 |
| 01/01 | 4.603475 | 4.612228 | +0.008753 |
| 01/10 | 4.024025 | 4.043722 | +0.019697 |
| 10/11 | 3.734240 | 3.737079 | +0.002839 |
| 10/01 | 6.409793 | 6.315915 | -0.093879 |
| 10/10 | 3.309108 | 3.302945 | -0.006163 |

The largest reduction is the opposite-modality10/01cell. The sign changes
across other cells rule out a uniform benefit from this selected comparison.

Retained branch hashes:
complete `fe1bc6fb533ae81b2bc8e4e2cf1fc4005c10ee0b165fd70ff016b300838ea7ad`;
mixed `15cad3fbd973ec1c674a76565961fbe42d39679a6affe01e7849dec78e23a434`.
Published comparison SHA256
`f4f28cb3a39fb2edd54c09be26804feb79da05c20601b1a84b7a4575f3184132`;
analysis SHA256
`d83418a53bd3287412c7dd7b34732b901086d2c159540acc8a93e6792ae6108f`.
Both hashes were rechecked against server artifacts at19:26:20 UTC+08.

The formal A training sampler already keeps initialization complete and
samples dynamic-template and search availability independently among11/01/10.
This pilot concerns a simplified frozen-branch control; it does not identify
a missing-template omission in the four running formal campaigns.
