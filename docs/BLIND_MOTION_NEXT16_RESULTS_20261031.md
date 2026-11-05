# Fixed CV On16 Full Development Videos

Both GPU workers finished normally in1132.04/1116.14seconds. Independent
worker analyses verified64 trajectories, all box/trace/GT hashes and
recomputed metrics. The complete outputs and pre-blind prefixes agree
bitwise between hold and velocity; baseline tensors remained unchanged.
Full videos contain7776frames;7742 valid frames are scored per condition.

| Condition | Hold PR20 / SR | Fixed CV PR20 / SR | Difference |
|---|---|---|---|
| Complete | 84.909989 / 69.541361 | 84.909989 / 69.541361 | 0 / 0 |
| Alternating32 | 69.109374 / 55.843400 | 70.264971 / 56.690159 | +1.155597 / +0.846760 |

Paired16-video bootstrap95% intervals are[-0.915736,3.678837] PR20
and[-0.891109,2.836747] SR. Both cross zero. SR improves on6 videos,
ties on6 and declines on4. PR improves on7, ties7 and declines2.
Five videos have no blind frames under this32-frame schedule.

Blind-frame effects average+6.325758/+5.264475 PR20/SR across11 videos
and1075 valid frames. Observed-after-first-blind effects average
+2.362047/+1.535023 across10 videos/4369frames; both-observed recovery
effects are+0.813136/+0.495921 over10videos/2651frames.
These groups have different membership and cannot be added.

## Identified Recovery Failure

The worst full-video SR change is `blkskirtwoman` -7.114032pp.
Its second blind interval, zero-based frames352..383, improves blind SR
by10.565476pp (hold9.077381, CV19.642857). Yet the following48 observed
frames have SR85.119048 hold versus18.750000 CV, a66.369048pp decline.

Reusing the actual crop geometry with configured SEARCH_FACTOR4.0:
at the first recovered frame384, the target overlaps the hold crop and
its center is inside, but it has no overlap with the CV crop.
The resulting first-frame box IoU is0.941611 hold versus0 CV.
Across the48 recovered frames, hold has0 unsupported crops, whereas
CV has34 unsupported crops; target-center exclusion counts also0/34.
This locates an actual crop-support failure after blind extrapolation.
It does not prove identity confusion, confidence miscalibration, or that
any proposed recovery intervention would fix it.

Across35 blind intervals, other regressions also occur during blind
prediction itself. For example `hyalinepaperfrontclothes` frames1312..1343
has blind SR23.363095 hold versus0.744048 CV. Thus accepting CV
unconditionally is not supported by this control.

These findings motivate testing recovery crop placement separately from
missing-frame prediction. Do not fit a trigger or motion cap to these
retrospective cases. Retain hold/CV as ordinary matched controls.

## Short-Gap Run

The registered block1 comparison launched at12:54:45 UTC+08 after the
long-gap final report and independent analyses passed. The continuation
archived old evaluator/controller/analyzer files under the long-gap
output's `verified_source/` before deploying staged files.

Controller1006350, GPU0 worker1006355 and GPU1 worker1006356 were verified
live at12:55:22. Each evaluator allocated2266MiB beside19290MiB training.
Output: `reports/server28-blind-motion-shortgap16-20261031`.
Estimate20-30minutes from12:54:45; inspect near13:14-13:25, then300s
completion polls. All four original trainers remain in place.

This is train882 development evidence, not official benchmark scores,
a validated novel module, or proof that the requested targets are reached.
Published official gaps are much shorter than32 frames; see the mask audit.

## Artifacts

Report SHA256:
`d0f7e15e2633924a2738661af1f383122af3ca2ee6e91856fe862021335bb933`.
Archived report/completed, both worker analyses, interval geometry analysis
and short-gap launch are in publication `reports/` and local `outputs/`.
Analyzer: `analyze_motion_intervals28.py`; geometry analysis uses an
exclusive separate output name and records the YAML hash/search factor.

At the latest training check, batch10 random enteredepoch9 afterepoch8
validation mean4.995457771. Utility remainedepoch8/update15524. This is
not convergence: original LR drop10 and post-drop reference11 remain.

## Velocity-Entry Audit

`audit_motion_estimates28.py` verifies the saved CV endpoint against the
last-two-observation formula for all35 blind intervals. Every long-gap
velocity used two complete11 observations. No cross-sensor switch occurs
between those two input observations, so such a switch cannot explain
these particular velocity estimates. Earlier template/trajectory history
can still differ; this does not isolate a neural failure mechanism.

For the `blkskirtwoman` interval352..383, the final observed center error
is5.203668px. Estimated velocity is(-12.493881,-1.535645)px/frame, while
the previous GT displacement is(-11.5,-10)px/frame. By the last blind frame,
CV center error is479.839829px. Holding the same last observed box would
have90.852464px error at that endpoint. This local comparison uses the
same CV history, not the separate closed-loop hold arm. It illustrates
the limits of extrapolating a recent displacement through a long gap.

For `hyalinepaperfrontclothes` interval928..959, recent GT displacement
is(0,0), but estimated velocity is(-19.464630,-9.828613)px/frame. The last
observed center already has173.483649px error; endpoint error grows to
701.996890px, versus173.061759px holding the same last box. Thus erroneous
observed trajectories are also a concrete source of motion-estimate error.
No confidence threshold or motion cap is fitted from these examples.

The one-frame schedule uses input states10 then11 before each00 gap,
unlike the long schedule's11 then11. Besides duration and switching
frequency, the observation-state pair used for velocity also differs.
The queued short-gap estimate audit will record the actual pairs; the
cross-duration result must not be attributed solely to gap duration.

At13:02:55 UTC+08 analysis controller1020925 was verified live waiting on
actual short-gap controller1006350 and CPU process869785. It checks
process exit every300seconds and then requires completion artifacts.
It runs the cross-duration contrast and velocity-entry audit for short
gaps, and independently verifies the final paired CPU result after that
process exits. Each task is processed as soon as it finishes.
Source hashes of its analyzer scripts are fixed at launch. Log:
`reports/server28-development-controls-analysis-20261031.log`.
Final/partial receipt:
`reports/server28-development-controls-analysis-20261031.json`.
No short-gap result or final CPU-pair analysis existed at this observation.

## Privileged Past-Velocity Control

`audit_past_gt_motion28.py` replaces the measured last-two-box velocity
with the last-two-GT velocity, anchored at the same saved CV last box.
It uses only past GT to construct predictions, then scores all valid blind
frames using later GT. It does not recompute the tracker or recovery.
The local hold control uses that same CV last box; it is not the separate
closed-loop hold arm.

Of35 long gaps,34 are eligible, covering1047 valid frames in11 videos.
The interval beginning160 in `blkskirtwoman` is excluded because its
past GT geometry is invalid. This changes the scoring subset from the
main closed-loop experiment; do not directly compare the tables.

| Offline predictor on this subset | Blind video-macro PR20 | SR |
|---|---:|---:|
| Same last CV box held | 22.897727 | 19.192145 |
| Measured velocity | 27.642045 | 23.395788 |
| Privileged past-GT velocity | 34.668561 | 25.305961 |

Past-GT velocity improves mean SR over measured velocity by1.910173pp,
with6 video wins,1 tie and4 losses. Relative to same-last hold it improves
6.113817pp, with5wins,2ties,4losses. Exact past displacement therefore
helps on average here but is not uniformly better and is not a performance
bound. Even accurate recent displacement can be a poor long-gap predictor.

In `blkskirtwoman` gap352, measured velocity gives blind SR19.642857,
while past-GT velocity gives15.327381 and same-last hold8.928571.
No method/threshold is selected using this privileged diagnostic.
Its role is to distinguish input trajectory error from the limitations of
extrapolating recent velocity; it does not identify their causal shares.

Report: `reports/blind-motion-next16-past-gt-audit-20261031.json`.
At13:08:27 UTC+08 short-gap workers had finished30/32 and24/32 trajectories.
Both remained live, as did post-analysis1020925, CPU869785 and fourtrainers.
CPU mixed-template training had completed two epoch validations.
