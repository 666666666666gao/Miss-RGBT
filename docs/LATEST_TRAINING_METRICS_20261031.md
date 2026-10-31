# Recent Training Metrics

Verified server state: 2026-10-31 18:44:59 UTC+08 (06:44:59 EDT).
Training steps and validation histories were read directly from all four
recovery run directories. All four trainer processes were verified alive.

## Live Recovery Training

Two server reboots interrupted the original jobs. All four restored trainers
and their controllers are alive. Model and optimizer were restored from
verified checkpoints, but original RNG state was unavailable; this is not
bitwise uninterrupted continuation.

Progress snapshot at 18:44:59 UTC+08:

| GPU | Run | Current epoch | Total updates | Completed checkpoint epoch |
|---|---|---:|---:|---:|
| 0 | Fixed Utility, effective batch8 | 11 | 27499 | 10 |
| 1 | Fixed Random, effective batch8 | 12 | 27593 | 11 |
| 2 | Candidate Utility, effective batch10 | 11 | 20740 | 10 |
| 3 | Candidate Random, effective batch10 | 12 | 22791 | 11 |

Means over the latest100 logged records, collected at18:44:59.
Logging is sparse; these are not necessarily100 consecutive updates.
task_c/task_m/KD columns are logged microbatch terms; update_loss_mean is
the mean update objective. These columns are not an additive decomposition.
All five recorded loss fields were finite over every recovered logged update.

| Run | update_loss_mean | task_c | task_m | KD forward | KD reverse |
|---|---:|---:|---:|---:|---:|
| Fixed Utility | 5.881230 | 2.576569 | 3.425462 | 0.607230 | 0.064264 |
| Fixed Random | 5.884656 | 2.459552 | 3.474240 | 0.629053 | 0.047347 |
| Candidate Utility | 5.713097 | 2.380517 | 3.167563 | 0.601034 | 0.070413 |
| Candidate Random | 5.786517 | 2.297556 | 3.193426 | 0.645160 | 0.073039 |

These training losses are not tracking precision, nor sufficient evidence of
convergence. All four trainer processes were verified alive at18:44:21.
Recent seconds/update were2.2771/2.4980/2.6014/2.5634 respectively.

Fixed batch8 Random has also completed epoch11 cached validation:
11/01/10 losses4.222112/5.883677/4.789905, mean4.965231
(128 tuples, update27500). This belongs to a different batch configuration
from the candidate history below.

At19:07:02 UTC+08 both fixed batch8 models have completed epoch11
validation on128tuples:

| Model | Both available | TIR only | RGB only | Mean |
|---|---:|---:|---:|---:|
| Fixed Utility epoch11 | 4.155397 | 5.794388 | 4.734656 | 4.894814 |
| Fixed Random epoch11 | 4.222112 | 5.883677 | 4.789905 | 4.965231 |

Utility is lower on all three proxy states; this does not replace tracking
evaluation or establish convergence. Recovered histories currently contain
one validation entry each, not eleven new entries.
Full downloaded/server SHA256 matches:
Utility `47a29672de633aad726b4ed8072a037d37405b9ad28e28bc074f5b64349fd4ef`;
Random `4df5c6adfbaa9eff8bf9b2a1b3138cf8d9b4196064ef4dd6f2331f999cc72d41`.
At19:07:02 actual live trainers recorded fixedUtility12/update28072,
fixedRandom12/28169, candidateUtility11/21249, candidateRandom12/23306.
The earlier training-loss average table remains timestamped18:44:59.
All four independent final verifiers and original cleanup waiter were
also confirmed alive from their actual process command lines.

## Current Campaign: Completed Epoch5 Development Tracking

All97 validation videos,54646 frames per condition,54289 scored frames.
These are macro PR20/SR development metrics, not official LasHeR245 or
RGBT234 scores. Candidate weights are batch10 epoch05/update10000.
E0 is the packaged availability baseline, not a matched five-epoch
fine-tuning control; Random is the relevant selector control.

| Condition | Model | PR20 (%) | SR (%) |
|---|---|---:|---:|
| Complete | E0 | 75.879164 | 61.654017 |
| Complete | Utility epoch5 | 79.596578 | 64.472777 |
| Complete | Random epoch5 | 80.667388 | 65.321675 |
| Synthetic outage | E0 | 64.528129 | 52.346049 |
| Synthetic outage | Utility epoch5 | 70.868974 | 56.749288 |
| Synthetic outage | Random epoch5 | 71.457491 | 57.152398 |

Utility minus E0: complete +3.717415PR/+2.818759SR points;
outage +6.340845PR/+4.403239SR points.
Utility minus Random: complete -1.070809PR/-0.848899SR;
outage -0.588517PR/-0.403110SR. Both Utility-Random PR/SR paired-video
95% intervals include zero. This does not establish selector superiority.
No final-five checkpoint selection follows from this fixed epoch5 check.

Evidence: `work/publish-miss-rgbt/reports/epoch05-full-paired-analysis-20261031.json`.
Server and local SHA256 both
`7034c378efa3dd1430dd32dcd94e968aaa11832a3e333f8a95b922aa106f7c31`;
server analysis-completion receipt matches. The completed independent
analyzer verified582 trajectories, hashes, metrics and scored frame counts.

## Latest Completed Cached Validation

128 fixed tuples; complete templates, search availability varied.
Task loss is lower-is-better and is not a tracking precision metric.

| State | Utility epoch10 loss | Random epoch10 loss | Random epoch11 loss |
|---|---:|---:|---:|
| Both available | 4.387996 | 4.239481 | 4.348452 |
| TIR only | 5.847460 | 5.984103 | 5.879302 |
| RGB only | 4.843250 | 4.767208 | 4.870980 |
| Mean | 5.026235 | 4.996931 | 5.032911 |

These proxy losses do not establish convergence or measure closed-loop
missing-template history. Utility10 is worse than Utility9; Random11 is
worse than Random10. Utility10 predates its first post-drop training epoch;
Random11 is its first post-drop reference. The registered patience resets
at11, so earliest candidate plateau stop remains14, not11 or12.
Recovered histories preserve the old entries; the new `seconds` fields
restart at recovery and must not be interpreted as total campaign time.

## Full Candidate Validation History

All entries use the same fixed128 validation tuples.
11=both available;01=TIR only;10=RGB only.

| Epoch | Utility11 | Utility01 | Utility10 | Utility mean | Random11 | Random01 | Random10 | Random mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 4.379604 | 5.839105 | 4.904822 | 5.041177 | 4.317632 | 5.787514 | 4.837589 | 4.980912 |
| 2 | 4.397491 | 5.935633 | 5.025561 | 5.119562 | 4.346727 | 5.995276 | 4.942290 | 5.094764 |
| 3 | 4.294023 | 5.835956 | 4.873298 | 5.001092 | 4.137216 | 5.800659 | 4.754964 | 4.897613 |
| 4 | 4.365547 | 5.877756 | 4.907265 | 5.050189 | 4.252729 | 5.851748 | 4.946566 | 5.017014 |
| 5 | 4.444279 | 5.805996 | 5.036729 | 5.095668 | 4.433926 | 5.900509 | 5.054249 | 5.129561 |
| 6 | 4.393904 | 5.730497 | 4.782562 | 4.968988 | 4.250734 | 5.757653 | 4.880781 | 4.963056 |
| 7 | 4.344943 | 5.822408 | 5.042096 | 5.069816 | 4.415956 | 5.907963 | 5.084766 | 5.136228 |
| 8 | 4.285330 | 5.838099 | 4.768391 | 4.963940 | 4.340837 | 5.815869 | 4.829668 | 4.995458 |
| 9 | 4.343521 | 5.634853 | 4.961537 | 4.979970 | 4.345498 | 5.683127 | 4.888636 | 4.972420 |
| 10 | 4.387996 | 5.847460 | 4.843250 | 5.026235 | 4.239481 | 5.984103 | 4.767208 | 4.996931 |
| 11 | pending | pending | pending | pending | 4.348452 | 5.879302 | 4.870980 | 5.032911 |

The lowest recorded Utility mean is epoch8,4.963940; Random is
epoch3,4.897613. These are proxy minima, not final selected checkpoints.
The campaign selects one weight using full tracking evaluations of its final
five checkpoints.

Recovered-history receipts:
`reports/candidate-utility-recovered-epoch10-validation-20261031.json`,
SHA256 `a54d7580b25ecf0ae8710927b783b65917c56032d7bfc4be1e75d1c699e51175`;
`reports/candidate-random-recovered-epoch11-validation-20261031.json`,
SHA256 `c3222577181e8f2a1763c4c5a15092b347a30efbfe675340986e4f126025beab`.
Server and downloaded file hashes match.

## Historical Official Reference

This is a separate migrated Utility epoch20 weight, SHA256
`c4aabefb495344b6d5e6e17655ac20fd51c4e2dec575843133096b319e047db3`.
It is not the current server28 campaign's score.

| Dataset | PR/MPR (%) | NPR (%) | SR/MSR (%) |
|---|---:|---:|---:|
| LasHeR-Miss245 | 68.565174 | 63.576593 | 54.621376 |
| RGBT234-Miss234 | 84.964353 | N/A | 60.511286 |

Only the requested LasHeR-Miss SR threshold is exceeded.
Current-campaign official complete/Miss scores do not yet exist, as verified
from the server results directory. Epoch5 development evaluation reports
PR20/SR only; no epoch5 development NPR or FPS was produced in this report.
The previous user-facing report omitted the completed epoch5 development
tracking table; that omission is corrected here.
## New Validation, 2026-10-31 19:43 UTC+08

| Batch10 candidate | Completed epoch | Update | Complete11 loss | TIR01 loss | RGB10 loss | Mean task loss |
|---|---:|---:|---:|---:|---:|---:|
| Utility | 11 | 22000 | 4.355199903 | 5.744167686 | 4.855334401 | 4.984900663 |
| Random | 12 | 24000 | 4.340420678 | 5.873122118 | 4.900210366 | 5.037917721 |

Utility changes from epoch10 mean5.026235372 to4.984900663;
Random changes from epoch11 mean5.032911457 to5.037917721.
Different completed epochs must not be treated as a matched comparison.
These are128cached-tuple validation losses, not tracking PR/SR or convergence.
Utility11 is the post-drop stopping-reference epoch; earliest stop remains14.
All four trainer commands were actually live at19:39:32; fixedU/R12
updates28833/28938 and candidateU11/22000,R13/24065 at19:40:06.
All recorded loss/task/update means were finite.

Raw histories:
`reports/candidate-utility-recovered-epoch11-validation-20261031.json`,
SHA256 `7e9e91dd6fb48c93ca3a23d47d8dd67a58575bae11ba023936d93f860542f3ab`;
`reports/candidate-random-recovered-epoch12-validation-20261031.json`,
SHA256 `34d5f0099aff3dd834ce72a0e51fc9582b3013048542460e0ca84fb20fd416bf`.

## New Fixed-Campaign Validation, 2026-10-31 20:44 UTC+08

| Fixed campaign | Completed epoch | Update | Complete11 loss | TIR01 loss | RGB10 loss | Mean task loss |
|---|---:|---:|---:|---:|---:|---:|
| Utility | 12 | 30000 | 4.160528 | 5.795622 | 4.741132 | 4.899094 |
| Random | 12 | 30000 | 4.222112 | 5.883676 | 4.825316 | 4.977035 |

Relative to epoch11, mean proxy loss rises+0.004280 for Utility and
+0.011804 for Random. These are small non-monotonic proxy changes, not
tracking gains. Fixed campaigns continue to the registered20-epoch budget.

At20:44:51 actual trainer command lines were live. Fixed U/R were epoch13,
updates30164/30376; batch10 U was epoch12/update23228 and R epoch13/update
25317. Candidate histories still ended at Utility11/4.984901 and Random12/
5.037918 in that exact snapshot. Do not treat the current in-progress epoch
as a completed validation. Candidate first post-drop reference epoch11;
earliest stop14, maximum20.

Raw fixed histories:
`reports/fixed-utility-recovered-validation-history-20261031.json`,
SHA256 `5af4a9862c0da019ae70e7ee0bbcb5fa0d57271564c17b0bb24b1894dcfe4d2a`;
`reports/fixed-random-recovered-validation-history-20261031.json`,
SHA256 `cdd57589a563eb24dc29e49ab7eaece9857f2a5eb40a6229b111c8a02d93df16`.
These fixed128-sample cached losses remain proxies, not PR/SR or convergence.

## Live Training Snapshot, 2026-10-31 21:56 UTC+08

Server process command lines were verified live at21:55:22+08. Latest
training-log records at21:56:25+08:

| Campaign | Epoch/update | Latest completed cached validation |
|---|---:|---|
| Fixed Utility | 13 / 31525 | Epoch12, mean4.899094 |
| Fixed Random | 13 / 31825 | Epoch12, mean4.977035 |
| Batch10 Utility | 13 / 24575 | Epoch12, mean5.004183 |
| Batch10 Random | 14 / 26700 | Epoch13, mean5.001639 |

Batch10 Random epoch13 mean improved by0.036279 from epoch12 and reset
stale-epoch count to0; epoch14 remains in progress. The other three rows
show the latest update, not a completed validation at that update. These
cached128-tuple task-loss values are not PR/SR and do not establish
convergence. The fixed runs retain their20-epoch budget; candidate
early-stop and final-five full-val97 selection protocols remain active.
No current-campaign official complete/Miss score or common selected weight
exists yet. The four training processes remain active; do not clean up
weights before final-five validation and selection complete.

## Live Training Snapshot, 2026-10-31 22:25 UTC+08

At22:25:21+08 all four trainer processes, four final-validation waiters,
and the original-weight cleanup waiter were live. Latest update positions:

| Campaign | Current epoch/update | Latest completed cached validation |
|---|---:|---|
| Fixed Utility | 13 / 32225 | Epoch12, 4.899094 |
| Fixed Random | 13 / 32500 | Epoch13, 4.945365 |
| Batch10 Utility | 13 / 25125 | Epoch12, 5.004183 |
| Batch10 Random | 14 / 27250 | Epoch13, 5.001639 |

These values are mean task losses on128 fixed cached tuples, not PR/SR or
convergence evidence. Fixed Random epoch13 improved0.031669 versus epoch12;
Batch10 Random epoch13 improved0.036279. Utility log tails had not advanced
since22:20, but both processes remained live at about101% CPU and GPUs0/2
were100% utilized; this does not indicate a stalled or failed run.
Current-campaign official complete/Miss scores and final-five selected
weights remain pending.

GPU telemetry at22:25:21 showed temperatures83/78/78/76C and power limits
of250W on GPUs0-3. This does not verify a configured85C target. Preserve
registered weights until the final-five evaluation and cleanup gates finish.

## Live Training Snapshot, 2026-10-31 22:37 UTC+08

At22:37:03+08 all four trainers, four final-validation waiters, and the
original-weight cleanup waiter remained live. Current updates were Fixed
Utility epoch14/32525, Fixed Random epoch14/32800, Batch10 Utility
epoch13/25350, and Batch10 Random epoch14/27475.

Fixed Utility completed cached epoch13 validation with losses11/01/10
4.259495/5.816765/4.654164, mean4.910141, stale_epochs2, stop=false.
This is0.011047 higher than epoch12, and is not PR/SR. Latest completed
validations for Fixed Random, Batch10 Utility and Batch10 Random remain
epoch13/4.945365, epoch12/5.004183, and epoch13/5.001639. All are mean
task losses on128 fixed tuples. Current-campaign official complete/Miss
scores and final-five selected checkpoint remain pending.

GPU telemetry at22:37:03 showed temperatures83/80/78/75C, power limits
250W on all GPUs, and utilization100/100/94/100%. This does not verify a
configured85C temperature target. Preserve registered weights until the
existing evaluation and cleanup gates complete.

## Live Training Snapshot, 2026-10-31 23:04 UTC+08

At23:04:51+08 all four trainers, four final-validation waiters, and the
original-weight cleanup waiter remained live. Current updates were Fixed
Utility epoch14/33200, Fixed Random epoch14/33475, Batch10 Utility
epoch13/25875, and Batch10 Random epoch15/28025.

Batch10 Random completed epoch14 cached validation (128 tuples): losses
11/01/10=4.316951/5.781932/4.812072, mean4.970318, stale_epochs0,
stop=false. Its mean improved0.031321 from epoch13/5.001639. Latest
completed validations for Fixed Utility, Fixed Random, and Batch10 Utility
remain epoch13/4.910141, epoch13/4.945365, and epoch12/5.004183.
These are task-loss proxies, not PR/SR or convergence evidence. No current-
campaign official complete/Miss scores or final-five selected checkpoint
exist yet.

GPU0-3 telemetry at23:04:51 was82/79/78/74C, power limits250W on each,
and utilization100/100/100/72%. This does not verify an85C target.
Preserve registered weights until evaluation and cleanup gates complete.
