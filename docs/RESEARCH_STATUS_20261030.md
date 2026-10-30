# Miss-RGBT evidence and remaining targets

## Latest Update, 16:31 UTC

All four training jobs continue with finite recorded key losses. Fixed20
GPU0/1 utility6951/random7429 of50000; adaptive batch10 GPU2/3 utility1561/
random1601 of2000 first-epoch updates. First boundaries estimated16:50:20/
16:48:08Z before validation/save. No candidate epoch validation yet.
User explicitly authorized single-GPU concurrent evaluation/data collection;
candidate val97 and official evaluators already use4workers/GPU after training.

Frozen selector-objective control completed452.37s: regression stopped45/
selected45, pairwise stopped30/selected26, other4 final heads/objective
cleaned with receipts. On4 heldout tuples/state, pairwise beats refit regression
in raw mean joint loss (3.767578 vs3.774773), but wins only5/ties1/loses6
against historical native selection. Raw mean slightly improves versus native
3.769790 while mean per-tuple relative benefit worsens (0.0074% vs0.0681%).
State10 raw loss worsens; pairwise ranking accuracy is only50.49%.
No consistent improvement or module promotion. Selected heads and raw
evidence retained. See docs/SELECTOR_OBJECTIVE_CONTROL_20261030.md.
Overall targets and three validated contributions remain incomplete.

Date: 2026-10-30. Authoritative source: SSH2028, /home/gaob/Miss-RGBT.

## Recorded Complete-input Results

These are historical results copied from server25, not completed server28 runs.

| Model | LasHeR PR | LasHeR NPR | LasHeR SR | RGBT234 MPR | RGBT234 MSR |
|---|---:|---:|---:|---:|---:|
| E0 baseline | 75.636503 | 71.772837 | 60.125714 | 90.677134 | 66.845228 |
| A utility | 75.100912 | 71.359539 | 60.004465 | 91.450712 | 66.197862 |
| A random | 73.388636 | 70.127102 | 58.778041 | 90.297337 | 65.145811 |
| A utility minus E0, pp | -0.535591 | -0.413298 | -0.121249 | +0.773578 | -0.647367 |

Sources: reports/e0_complete/metrics.json and artifacts/server25-data2/results/a_{utility,random}_full20_complete_20261020/metrics.json.

Utility beats its same-budget random selection control on all five complete-input metrics. This does not establish the requested improvement over E0. Historical A Miss metrics are absent from the migrated A result directories.

## Recorded Miss Results

| Fixed epoch20 model | LasHeR-Miss PR | LasHeR-Miss SR | RGBT234-Miss MPR | RGBT234-Miss MSR |
|---|---:|---:|---:|---:|
| E3-allBN | 69.337978 | 55.175799 | 85.063210 | 60.325720 |
| E4-allBN | 68.422283 | 54.499880 | 83.975517 | 60.033254 |
| Strict user threshold | >69.3 | >54.4 | >88.1 | >64.3 |

Sources: reports/e3_allbn_full20_miss/metrics.json and reports/e4_allbn_full20_miss/metrics.json.

E3-allBN exceeds both LasHeR-Miss thresholds. Its RGBT234-Miss shortfalls to the threshold values are3.036790 and3.974280 percentage points; crossing a strict threshold requires a larger improvement. No combined target attainment is established.

## Module Evidence

| Candidate | Existing evidence | Still required |
|---|---|---|
| A, task utility selection and local compensation | Implemented; real training; utility beats random on historical complete metrics | Small fixed validation audit finds state-dependent ranking signal; 11/01 unreliable,10 preliminary only. Complete/Miss paired results, larger independent ranking audit, selection/capacity controls, repeatability and baseline benefit remain |
| B, evidence-driven recovery | Prior fixed-trajectory diagnostics identify search crops that lose target support; U0/U1/U2 template controls already completed | A deployable recovery mechanism and matched controls separating localization, search expansion and template effects |
| C, protection of complete-task learning | Prior real-train CPU probe demonstrated reliable-gate activation, gradients and AdamW update differences | A mechanism that exceeds ordinary KD/optimization controls, full training, complete/Miss gains |

Existing mask handling, response KD, residual MLP and native template updates cannot simply be renamed as three validated contributions. Test failure analysis has informed the direction; test data must not be used for hidden parameter selection. RGBT234 has no independently established development split.

Sources: docs/MODULE_EVIDENCE_PROTOCOL.md, docs/RGBT234_DEVELOPMENT_BOUNDARY_20261018.md, reports/cpu_reverse_probe_ep01_seed2027_v1 and reports/template_controls_20261018_v1.

## Current Execution

Server28 trains existing A utility/random configs on GPU0/1, fixed seed2027 and final epoch20. This is a same-configuration run on the migrated system, not a new-method improvement or an additional seed.

A separate evaluation controller, PID3463984, waits for both successful training exit receipts. It checks live training-controller presence every300 seconds while waiting, verifies completion records and frozen source hashes, and then evaluates utility complete/Miss followed by random complete/Miss using GPU0/1. Each condition must finish479 sequences/337353 frames. Failures remain in original logs and stop the queue; there is no automatic restart.

Queue: /home/gaob/Miss-RGBT/logs/server28_a_evaluation_20261030.
Results: /home/gaob/Miss-RGBT/results/server28_20261030.

Training-data audit: train882/459435 frames and val97/54646 frames, no count mismatches or split overlap. There are1314/260 annotations with nonpositive width or height; existing dataset code excludes these from visible-frame sampling. The audit checks file counts and annotation shapes, not image decoding or original E0 training exposure.

Audit: reports/server28-training-data-audit.json.

The overall research goal remains incomplete.

New user instructions allow GPU2/3 and adaptive candidate budgets. A separate
batch10/LR1.25e-4 utility/random pair is now training there with fixed cached
validation tuples, post-drop plateau stopping and rolling final5 checkpoints.
All5 will receive full val97 paired tracking; the same validation-selected
checkpoint then receives official complete/Miss evaluation. At15:24:15Z both
had24 finite updates and22710MiB GPU occupancy. This tests parameters for
existing A, not three new modules. See docs/CANDIDATE_TRAINING_LAUNCH_20261030.md.
The GPU0/1 fixed20 campaign and its evaluation queue remain separate.

Independent-anchor evidence was subsequently checked on CPU. Separately
encoded foreground-minus-background cosine still loses to same-frame fixed
background candidates on21.6% of observed outage rows in corrected37-stride
sampling (102rows; correlated, fixed8 cohort). State01/10 target-top1 is only
68.2/73.9%. A32-training-sequence frozen-descriptor MLP pilot also failed to
give consistent heldout ranking improvement:11/01/10 correct18/16/17 of22
groups versus cosine18/17/16. The selected small head is retained but is not
promoted into recovery tracking. No calibrated acceptance or recovery benefit
is demonstrated. See docs/INDEPENDENT_ANCHOR_AUDIT_20261030.md.

The subsequent fixed9-crop proposal experiment completed35 sparse early
observations/315 proposals. Independent cosine/pilot-head selection harms
useful center candidates (7/6 drops>0.1IoU); no recovery intervention is
promoted. All-frame geometry shows that9 native-size crops restore overlap
on only150/1505 already-unsupported observed frames; median grid cost to
include the target center is169 crops. Severe drift requires a different
coverage/selection strategy or earlier detection. This is fixed8 development
diagnostic evidence, not a deployed module or benchmark gain. See
docs/WIDER_SEARCH_AUDIT_20261030.md.

A CPU branch-only control now verifies bitwise complete11 output equality
and frozen baseline state across8 real LocalRepair updates. Val8 missing
loss changes are -0.002850(01) and+0.000217(10), so no stable benefit or
convergence is claimed. Existing absolute utility targets are also much
smaller than selector predictions on fixed4 audit tuples; calibration
and ranking remain distinct issues. Ordinary RankNet is recorded as a
future matched baseline, not a contribution. See
docs/PROTECTED_ADAPTATION_AUDIT_20261030.md.

## Verified Training Snapshot

Observed2026-10-30T16:05:12Z (12:05 America/New_York). Live process command lines were checked, including the waiting evaluation controller.

| Model | Updates | Recent seconds/update | Estimated finish, America/New_York |
|---|---:|---:|---|
| utility | 6306/50000 | 2.431 | October31,17:35 |
| random | 6745/50000 | 2.287 | October31,15:34 |

All recorded loss, task-C, task-M and accumulated mean-loss values were finite. These are training health observations, not evaluation scores. Estimates use the latest100 recorded updates and may change.

Snapshot: reports/server28-training-progress-20261030.json.

Development outage inputs and a standalone GPU recovery-diagnostic runner are prepared; see docs/VAL97_RECOVERY_PROTOCOL_20261030.md. Actual crop and IoU helper comparisons passed on CPU. A separate explicit CPU tracker has now completed real neural tracking on the predetermined first257 frames of the first val97 sequence across four conditions. All execution checks passed, including32 holds/224 calls for alternating outages. No observed crop escape occurred in that prefix. This does not establish full-val97 results, GPU numerical equivalence or module efficacy; the GPU runner remains unexecuted. See docs/VAL97_CPU_PREFIX_AUDIT_20261030.md. The frozen formal training/evaluation source remains unchanged.

The bounded source review of KeepTrack and GlobalTrack is saved in docs/RECOVERY_SOURCE_REVIEW_20261030.md. It provides a prospective search/acceptance experiment and matched controls, not a transplanted implementation or demonstrated recovery benefit.

A real CPU AdamW intervention is complete: three paired single effective-batch8 updates at the historical A epoch20, with all BN frozen and no utility auxiliary term. Additional task gradients were not negatively aligned with the complete gradient in this case, yet complete-training loss improvement was smaller with the missing task/KD. Changes on eight additional train882 samples were tiny and unreplicated. This does not establish a benchmark-regression cause or justify adding ordinary gradient projection to the frozen run. See docs/COMPLETE_TASK_UPDATE_AUDIT_20261030.md and reports/server28-task-interference-20261030/report.json.

## Fixed8 CPU Trajectory Cohort, Complete

At2026-10-30T14:39:41Z the fixed first8 val97 sequences finished paired
complete/alternating-outage tracks. Across3924 observed valid frame pairs,
836 had outage-only crop-support loss,670 had both crops miss target support,
and6 had complete-only crop-support loss. Paired support-loss rates were
38.38% outage versus17.23% complete; frame-weighted paired observed IoU
delta was-0.119095. All6 inverse frames are in `ab_pingpongball`, whose
complete/outage mean IoU is only0.071520/0.070928 over87 valid labels;
26 invalid-size annotation frames were retained without geometry labels.

Outage-only support loss occurred in just2/8 sequences:
`leftdrillmasterstanding`770/1839 observed pairs (with717 complete
zero-overlap crops across its full valid trajectory) and
`leftgirlunderthelamp`66/225. The other six had none. `midboyplayingphone`
had a small positive paired observed IoU delta+0.033628. This heterogeneity
rules out a universal failure assumption; these CPU geometry diagnostics
are not official Miss metrics, identity-recovery proof or causal module
gain. The frame-weighted aggregate is dominated by the long
`leftdrillmasterstanding` sequence.

Low Hann peak score predicted zero GT/crop intersection with AUC0.837 in
alternating outage but0.553 in complete tracking; raw peak AUC was
0.847/0.559. Outage positive examples occurred in only2 sequences, so
six per-sequence AUCs are undefined. The score may be a useful outage cue
but is not a universal/calibrated trigger. B should test independent
target/template consistency alongside response evidence, with any trigger
calibrated on train882 rather than this diagnostic. Details and full traces:
docs/VAL97_CPU_COHORT8_LAUNCH_20261030.md and reports/
server28-val97-cpu-cohort8-20261030.

## A Selector Diagnostic, 2026-10-30

A frozen CPU counterfactual audit tested16 regions on four first-search val97
tuples in each availability state. Spearman correlation of predicted
utility logit with actual single-region task-loss reduction was -0.058 in
11, -0.136 in01, and+0.456 in10. Mean native top4 normalized loss changes
were -0.020%, +0.082%, and+0.356%, versus seeded random +0.094%, +0.110%,
and+0.156%. Between-tuple variation is large and n=4/state; do not infer
reliable improvement. This preliminary evidence does not support describing
the selector as generally task-optimal. Do not change the running training
or tune on this probe. Details and limitations:
docs/UTILITY_RANKING_AUDIT_20261030.md.

Training-log per-batch-mean correlation was also near zero, but is only a
weak aggregate diagnostic, not individual-region calibration evidence:
reports/server28-utility-training-audit-20261030.json.
