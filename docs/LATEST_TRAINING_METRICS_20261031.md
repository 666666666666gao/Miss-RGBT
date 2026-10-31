# Recent Training Metrics

Verified server state: 2026-10-31 17:59:33 UTC+08 (05:59:33 EDT).

## Live Recovery Training

Two server reboots interrupted the original jobs. All four restored trainers
and their controllers are alive. Model and optimizer were restored from
verified checkpoints, but original RNG state was unavailable; this is not
bitwise uninterrupted continuation.

Progress snapshot at 17:59:33 UTC+08:

| GPU | Run | Current epoch | Total updates | Completed checkpoint epoch |
|---|---|---:|---:|---:|
| 0 | Fixed Utility, effective batch8 | 11 | 26598 | 10 |
| 1 | Fixed Random, effective batch8 | 11 | 26749 | 10 |
| 2 | Candidate Utility, effective batch10 | 10 | 19687 | 9 |
| 3 | Candidate Random, effective batch10 | 11 | 21720 | 10 |

Means over the latest100 logged updates, collected at17:59:33:
task_c/task_m/KD columns are logged microbatch terms; update_loss_mean is
the mean update objective. These columns are not an additive decomposition.
All five recorded loss fields were finite over every recovered logged update.

| Run | update_loss_mean | task_c | task_m | KD forward | KD reverse |
|---|---:|---:|---:|---:|---:|
| Fixed Utility | 5.785116 | 2.532332 | 3.362975 | 0.598257 | 0.067988 |
| Fixed Random | 5.753347 | 2.409846 | 3.260267 | 0.603756 | 0.057157 |
| Candidate Utility | 6.020692 | 2.532831 | 3.416879 | 0.681238 | 0.069590 |
| Candidate Random | 5.737769 | 2.346651 | 3.282969 | 0.645740 | 0.079484 |

These training losses are not tracking precision, nor sufficient evidence of
convergence. All four trainer processes were verified alive at17:58:41.
Recent seconds/update were2.5272/2.5601/2.5588/2.5065 respectively.

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

| State | Utility epoch9 loss | Random epoch9 loss | Random epoch10 loss |
|---|---:|---:|---:|
| Both available | 4.343521 | 4.345498 | 4.239481 |
| TIR only | 5.634853 | 5.683127 | 5.984103 |
| RGB only | 4.961537 | 4.888636 | 4.767208 |
| Mean | 4.979970 | 4.972420 | 4.996931 |

These proxy losses do not establish convergence or measure closed-loop
missing-template history. Random epoch10 mean is worse than epoch9.
Utility epoch10 validation is not yet complete.

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
| 10 | pending | pending | pending | pending | 4.239481 | 5.984103 | 4.767208 | 4.996931 |

The lowest recorded Utility mean is epoch8,4.963940; Random is
epoch3,4.897613. These are proxy minima, not final selected checkpoints.
The campaign selects one weight using full tracking evaluations of its final
five checkpoints.

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
