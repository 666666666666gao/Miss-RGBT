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
