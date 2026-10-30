# Adaptive larger-batch parameter pair

Date: 2026-10-30. SSH2028 /home/gaob/Miss-RGBT.
This pair tests training parameters for existing A utility/random, not three
new validated modules. Target metrics remain unachieved.

## Actual Launch

| Selector | Physical GPU | Controller PID | Training PID |
|---|---:|---:|---:|
| Utility | 2 | 3816049 | 3816056 |
| Random | 3 | 3816050 | 3816057 |

Both use microbatch5, accumulation2, effective batch10, LR1.25e-4,
backbone multiplier0.1, seed2027,20000 samples per epoch, maximum20 epochs.
Source/model/objectives/tracker of the fixed GPU0/1 campaign were not edited.
The new trainer and evaluators live outside that frozen runtime.

At2026-10-30T15:24:15Z both candidates completed24 actual updates with finite
recorded key losses. Observed occupancy was22710/24576MiB per GPU, utilization
100%. Peak allocated memory was21118406144 bytes; occupancy and allocated
memory are distinct measurements.

Both cached exactly the same128 validation tuples:
`13105d2512b095da153e0ec8cd38cf6264c0971db7d40b036ddfc10a71455391`.
Their first optimizer-update input hashes also matched each other and the
corresponding native batch5 profile. The fixed tuple cache is sampled from
the val split once, then reused, with Python/NumPy/Torch RNG isolated during
construction. It is a small sampled convergence proxy and is not guaranteed
to cover all97 sequences. Current search states11/01/10 are averaged equally;
templates are complete, no patch dropout is applied. It does not measure
long-term drift, hold/recovery behavior or official benchmark metrics.

## Selection and Cleanup

Each epoch evaluates the fixed cached tuples with eval/no_grad and restores
the selector RNG buffer; training mode and frozen BN policy are restored.
The early-stop reference resets at epoch11, the first epoch trained with the
post-drop LR. Improvement requires loss below reference*(1-0.002). Three
consecutive non-improving post-drop epochs stop training, earliest epoch14.
Twenty epochs is a budget limit, not evidence of convergence.

The trainer retains a rolling final5 epoch window. Once training exits
successfully, all5 weights receive full97-sequence validation in complete
and fixed alternating-outage conditions. The same epoch maximizes their mean
macro-sequence SR; ties choose the earlier epoch. Validation SR uses21 IoU
thresholds0..1, IoU>threshold, excluding initialization/nonpositive-size GT
and including valid both-missing held frames. PR20 is also recorded.
These are development scores, not official LasHeR-Miss metrics.

This selects the best of the final5, not the best of all training history.
After all10 validation passes succeed, records and checkpoint hashes are
verified, the selected weight is retained and other4 candidate weights are
deleted with receipts. Older rolling checkpoints are also recorded before
removal. Historical/formal weights are outside this cleanup scope.
Only then does the same selected checkpoint receive official complete and
Miss evaluations,479 sequences/337353 frames each, on its assigned GPU.
Official test results do not select epoch or alter the completed run.
The standalone official evaluator reuses the frozen evaluation code with
only its runtime path and GPU2/3 CLI binding adapted.

## Verification and Timing

First epoch validation completed on both models: utility mean task loss
5.041177, random4.980912 over the same cached128 tuples/three states.
Receipts at5342.69/5177.07s from start (89.04/86.28min, before save).
At16:55:34Z both epoch01 weights existed and training continued in epoch2,
2062/2117updates. No convergence claim is possible from one epoch.
All GPUs were limited to250W, with temperatures84/81/79/76C; recent speed
predicts next boundaries18:46/18:33Z. Initial timing estimates below predate
the power cap and should not be treated as final completion times.

- Native batch5 profiles:2 and8 actual updates per selector, all finite;
  eight-update probe views C/M/C/M, peak21118406144 bytes.
- Candidate trainer smoke:6 short epochs/12 updates with real optimization
  and fixed validation;70.44s, rolling retention left epochs2..6 exactly.
- Native validation evaluator smoke:1 sequence565 video frames,35.30s,
  finite trajectory and explicit PR20/SR output.
- Python3.7 compilation passed for all new entry points.
- Smoke checks do not establish convergence, generalization or method benefit.

Short-profile post-startup time2.41-2.43s/update suggests about80-81min per
2000-update epoch, before epoch validation/save overhead. Initial scheduling
budget is80-90min to first epoch, roughly19-29h for14-20 epochs. Full last5
validation needs additional hours; the previous single-worker cohort suggests
6.25-7.5h per candidate, while the new4-worker evaluator still needs full-run
timing. Two candidates operate in parallel. Official evaluation adds time.
Use actual first-epoch and validation receipts to replace these estimates.
Controller polls are300 seconds; no automatic failure restart.

## Evidence

- reports/server28-candidates-20261030/verified-launch.json
- reports/server28-candidates-20261030/batch10_{utility,random}_seed2027
- runs/server28/batch10_{utility,random}_seed2027
- reports/server28-batch5-profile-summary-20261030.json
- reports/server28-candidate-pipeline-smoke-20261030
- results/server28_candidates_20261030 (future selected-checkpoint official runs)

The separate epoch-validation-plan document is a proposed read-only review;
this launch document describes the actual implemented boundary.

## Live State, 2026-10-31 05:14 UTC

The batch10 utility/random trainers on GPU2/3 were both live at epoch4,
updates6925/7125. Their step losses were finite and peak allocated memory
was21118406144bytes on each GPU. The epoch4 validation boundary had not
completed; based on observed throughput, utility was roughly40minutes from
that boundary. No convergence judgment is supported before the registered
post-LR-drop stopping window.

Historical A utility20 official Miss evaluation PID4138772 was live with
175/479 receipts:103 LasHeR-Miss and72 RGBT234-Miss. No aggregate metrics
receipt existed yet. GPU0/1 fixed20 utility/random training was also live.
No jobs were restarted or added.

At05:46UTC both candidate trainers and the historical evaluator remained
live. Batch10 utility/random had reached epoch4 updates7525/7750, about
475/250updates from their next epoch boundaries. Progress was slower than
the earlier estimate, so the next check is scheduled near the random
boundary. Historical Miss receipts reached353/479 (179 LasHeR-Miss,
174 RGBT234-Miss), with no final metrics receipt.

## Epoch 2 Observation, 2026-10-30 18:42 UTC

Both runs used the same hashed128 validation tuples. Fixed task-loss means:

| Candidate | Epoch1 | Epoch2 | Epoch2 states11/01/10 |
|---|---:|---:|---|
| Utility | 5.041177 | 5.119562 | 4.397491 / 5.935633 / 5.025561 |
| Random | 4.980912 | 5.094764 | 4.346727 / 5.995276 / 4.942290 |

Both means worsened versus epoch1 (utility+1.55%, random+2.29%). Random
remains numerically lower by0.024798, but this small cached task loss is
not a tracking metric and CPU/GPU loss equivalence is not assumed. This is
no evidence of convergence or utility-selector benefit. Do not stop early:
the registered stopping reference starts after the epoch10 LR drop, with
earliest stopping at epoch14 after3 stale epochs. No full val97 checkpoint
has yet been selected.

At18:42:11UTC utility/random were at epoch2 update4053 and epoch3 update4176;
recorded losses were finite. Recent speed3.453/3.115s/update predicted next
boundaries20:34/20:17UTC. GPU0/1 fixed20 utility/random were at epoch5,
10032/10435 updates, all logged loss fields finite; 100-update speed
2.977/3.142s estimated completion Nov1 03:46/05:15UTC before evaluation.
Historical A Miss fill-in evaluation is separately running with one worker
on each of GPU0/1; at18:42UTC it had2/479 sequence receipts. These times and
task losses are progress indicators only, not final benchmark evidence.

## Historical A Miss result, 2026-10-31 05:56 UTC

The historical utility20 checkpoint completed all479 sequences/337353
frames in12528.71seconds. Its Miss metrics are LasHeR PR/SR68.565174/
54.621376 and RGBT234 MPR/MSR84.964353/60.511286. E0 deltas are
+5.092667/+4.038228 and+2.169539/+0.936850pp. LasHeR-Miss SR passes its
threshold by0.221376pp; LasHeR-Miss PR is0.734826pp short, RGBT234-Miss
MPR3.135647pp short and MSR3.788714pp short. This is not a complete-input
evaluation or the active batch10 candidate. All official sequence receipts
and curves are archived under `reports/historical-a-miss-20261031/`.
