# Miss-RGBT evidence and remaining targets

## Latest Evidence, 2026-10-31 14:07 UTC+08

The576-call fixed-crop CPU probe completed in303.75seconds. A post-hoc3x3
current/prior-state factorial reuses its saved outputs without new calls.
At equal lag1, same-state prior predictions have the lowest macro velocity
error for each current state11/01/10. Current01/10 same-minus-complete-prior
velocity errors are-2.912099/-1.035978px/frame with video intervals below0.
This rules out complete-prior quality as the sole explanation in that fixed
context, but is not closed-loop, official or novel-module evidence.
All64 triplets have valid labels; uncertainty resamples16 videos.
Lag2 also independently reduces velocity error under all three prior states.
See docs/MOTION_IDENTIFIABILITY_20261031.md and published factorial reports.

Fresh factorial replication runner1129716 waits on actual adjacent reference
controller1090400, then repeats576 calls and both analyses without tuning.
At14:07:46 both adjacent GPU workers had32/32tracks; final parent checks
were pending. Availability-history replication and all four main training
jobs remain separate, with unchanged GPU execution sources.

The historical A utility20 Miss evaluation is already complete, verified
against current server metrics and479-sequence/337353-frame completion:

| Model | LasHeR-Miss PR | SR | RGBT234-Miss MPR | MSR |
|---|---:|---:|---:|---:|
| Historical A utility20 | 68.565174 | 54.621376 | 84.964353 | 60.511286 |
| Strict target | >69.3 | >54.4 | >88.1 | >64.3 |

Only LasHeR-Miss SR clears its threshold. These historical weights are
not the currently training candidates. Earlier live-evaluation entries
below are dated history. No combined target attainment or three validated
paper modules has been established.

## Latest Evidence, 2026-10-31 13:56 UTC+08

Both protected-template arms completed192updates, selectedepoch6, retained
one branch weight each and removed4 with independently verified receipts.
Mixed template loss is4.355298 vs complete4.363575 (-0.19%) on reused
selection tuples, with3/6 missing-search cells improving. This does not
justify a larger GPU campaign or a module claim.

Completed16-video motion history contrast gives PR/SR+1.806145/+1.659243pp,
with paired-video intervals above0 and bitwise equal controls. Crucially,
an exact1290-blind-frame audit per study proves that availability matching
is identical to ordinary lag2 velocity under this schedule. Fresh-video
replication1090396 is running, but cannot identify a sensor-state mechanism.
An equal-lag, fixed-crop CPU probe1109528/1109529 is separately running.
See docs/DEVELOPMENT_CONTROLS_FINAL_20261031.md and
docs/MOTION_IDENTIFIABILITY_20261031.md for evidence and limitations.
Four main trainers remain live; no complete/Miss target attainment or
three substantively validated modules has been established.

## Native-Scale Spatial Audit, Complete

CPU worker4006613 completed46points/2346candidates in1314.66s; terminal
verified17:37UTC. Same51fixed proposals, no currentGT generates proposals.
Across45validpoints, native/oracle/actualHann meanIoU:
0.564517/0.647675/0.485004; IoU>=.5 counts32/33/28, actual5harms>.1.
Among12nativeunsupported points,9regain center coverage but only1has an
oracle successful localization, in complete11. Missing01/10 oracle success
is0/3and0/4. Ranking alone cannot fix the8center-covered points without a
successful candidate. Do not promote this fixed-grid selector or infer a
closed-loop gain. See docs/NATIVE_GRID_AUDIT_20261030.md.

Separate CPU reference worker4024185 completed exact128cached candidate
validation inputs for both models in1114.01s, terminal verified17:42UTC.
Mean E0/historicalA20 task loss5.408533/4.995538 (7.636% reduction).
E0 states11/01/10=4.412175/6.750802/5.062622; historicalA20=
4.398898/5.788143/4.799574. Complete improvement is small. This is a
development task-loss reference, not PR/SR or CPU/GPU numerical equivalence.
No stopping protocol changed. See docs/VALIDATION_REFERENCE_20261030.md.

CPU worker4063222 completed46points/2346candidates in1355.57s. Same
Hann-decoded boxes as the earlier run (max difference0px). Hann/raw peak
selection meanIoU .485004/.431677 vs native .564517; harms>0.1 are5/11.
Oracle ceilings .647675/.656347; unsupported-state oracle success1/12 for
both. Hann removal does not fix candidate ranking or localization. No live
tracker change or module promotion. See docs/NATIVE_GRID_DECODE_20261030.md.

Historical A full official Miss scoring was attempted concurrently on
GPU1 but failed during four-worker model loading with CUDA OOM (only
5.75/11.75MiB free at the failed allocation); the later cuDNN error is
secondary. A measured single inference worker used500MB peak allocated and
completed one565-frame sequence in52.04s on GPU1 alongside training.
Evaluator now supports configurable workers and sequence-limited smoke.
Two-GPU smoke completed2long official sequences/22546frames: ``blkboy`shead``
12862frames on GPU1 and `boyshead9684`9684frames on GPU0. Both receipts
contain finite boxes. The first controller stopped on an incorrect
aggregate-receipt assumption; corrected controller PID4138770 verified the
two per-sequence receipts and launched the complete historical A Miss
evaluation, PID4138772, one worker per GPU, output
results/server28_historical_a_20261030/miss_single_worker_per_gpu.
At18:29UTC the evaluator was alive with0 full-run sequence receipts; no
metrics yet. Smoke throughput suggests roughly4h for this run. The initial
four-worker OOM output is preserved.

## Training Update, 18:42 UTC

Candidate GPU2/3 utility/random are at epoch2 update4053 and epoch3
update4176; recent speeds3.453/3.115seconds per update project boundaries
20:34/20:17UTC. Both fixed128 task-loss means worsened from epoch1 to epoch2:
utility5.041177->5.119562 (+1.55%), random4.980912->5.094764 (+2.29%).
Epoch2 state losses utility11/01/10=4.397491/5.935633/5.025561;
random=4.346727/5.995276/4.942290. Inputs and first update hashes match.
This is a non-monotonic early proxy, not a convergence or tracking result.
Registered post-LR-drop stopping has not activated; continue without
changing training.

Fixed20 GPU0/1 utility/random are epoch5 at10032/10435 updates, finite.
Latest100 rates2.977/3.142s/update project completion Nov1 03:46/05:15UTC,
before queued full evaluation.

Historical A Miss evaluation PID4138772 remains live with2/479 sequence
receipts at18:42UTC; it is the separate historical reference, not a new
candidate. One worker per GPU0/1; the failed4-worker attempt is preserved.

## Oracle Crop Diagnostic, 17:07 UTC

CPU GT-guided diagnostic completed45points/135crops in73.88s. Of12
native unsupported points, GTcenter+previous size restoresIoU>=.5 at6,
GTcenter+true size at10, GTcenter+coarse tile size at0. Existing raw global
oracle also0/12. Target median shortside searchpixels30.14/43.98/10.67.
Center/scale/context/Hann effects interact; no deployable recovery/novelty
or official gain. This narrows a concrete bottleneck before further recovery
design. See docs/ORACLE_CROP_AUDIT_20261030.md. Existing four GPU trainings
unchanged. Source and development evidence are attached to draft GitHub PR1,
branch codex/server28-research; not merged into default main.

## Latest Update, 16:55 UTC

Both candidates completed epoch1 and continued epoch2 at2062/2117 updates.
Fixed128-sample mean validation task loss utility/random:5.041177/4.980912.
States11/01/10 utility:4.379604/5.839105/4.904822; random:
4.317632/5.787514/4.837589. Identical cached validation inputs, finite losses.
First validation receipts arrived5342.69/5177.07s after training start
(89.04/86.28min, before checkpoint saving). Epoch01 checkpoints exist.
This is a task-loss proxy, not PR/SR or convergence evidence.
Recent throughput at250W suggests epoch2 boundaries18:46/18:33Z,
still subject to changing speed. Temperatures84/81/79/76C at16:55Z.
Earlier snapshots below are retained as dated history.

### Selector Objective Control

See the detailed report below: pairwise loss showed a small
aggregation-sensitive result, insufficient for promotion.

### Recovery Proposal Audit

Fixed CPU audit of46 points/11 proposals each on first8 val sequences.
Whole-frame/coarse proposals restored crop-target overlap at12/12 native
no-support points, but GT-oracle best IoU stayed below0.5 at all12 and had
mean0.601 overall vs native0.565. Hann peak selection mean IoU0.547;
raw peak without Hann0.422, with12 drops>0.1. Coverage alone does not
recover identity/localization; no recovery module promoted. See
docs/GLOBAL_PROPOSAL_AUDIT_20261030.md and raw reports.

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

At16:48Z, all four GPU power limits were250W; temperatures0-3 were
84/81/80/75C. This is a live hardware snapshot, not proof of a persistent
setting. Candidate utility/random were1937/1988 updates, epoch1, finite
logged losses and estimated first boundaries16:51:50/16:49:08Z; completed
epoch validation was still empty. Fixed utility/random remained6951/7429
at last inspector sample16:31Z; refresh before relying on those counts.

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
