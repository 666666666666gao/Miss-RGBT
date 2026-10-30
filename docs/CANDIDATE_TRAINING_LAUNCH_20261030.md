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
