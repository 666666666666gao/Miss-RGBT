# Prospective Calibration Gap Control

The official-mask audit shows a coverage gap between development block1
and prolonged absence of complete observations. This experiment tests the
unchanged ordinary calibration algorithm on new development videos.
It does not train weights, tune an expiry threshold or establish novelty.

New standalone entry point `run_calibration_gap28.py` copies the existing
four-arm execution/verification structure. Existing queued/running sources
and frozen training runtime remain unchanged.

## Registered Design

The next sequential development indices60..63 are used without duration
or outcome filtering:

| Index | Video | Frames |
|---|---|---:|
| 60 | boyaftertworunboys | 458 |
| 61 | twopeople_0109 | 450 |
| 62 | 10rightblackboy | 333 |
| 63 | leftbottle | 1143 |

Total2384frames. Every video has four arms (native, calibration sham,
center-only, center-plus-size) and four conditions: complete, block1,
block32, block128. Total64trajectories. Each video remains on the same
physical GPU across all arms/conditions; load balancing gives1143/1241.

For block length B, frame t uses
`[[1,1],[0,1],[1,1],[1,0],[1,1],[0,0]][(t//B)%6]`.
Initialization is always complete and does not calibrate. Block1 is the
existing immediate-refresh control, block32 is an existing development
schedule, and block128 is a fixed fourfold stress extension. No official
score selects among these conditions or determines method parameters.

This is a within-video condition comparison. Longer blocks also change
which images are observed, template writes and motion histories, not just
cache age. Do not call an interaction an isolated causal age effect.
Some short videos end before all block128 states occur; keep every video
and report exposure rather than filtering it. In particular, the333frame
video has no RGB-only block128 interval. Only the1143frame video reaches
the first blind block128 interval.

## Analysis And Controls

For each condition, report macro PR20/SR for every arm and paired-video
bootstrap intervals for center-native, geometry-native and geometry-center.
Also report each block32/128 contrast minus its block1 contrast with paired
video intervals. These four correlated videos are not independent seed or
official benchmark evidence; intervals remain descriptive.

The independent analysis recomputes all64trajectory metrics and checks
input/source/checkpoint/trajectory hashes, native-sham equality, complete
equality, identity priors, past-only cache provenance, scale/offset
reconstruction, native clipping, model-call counts and blind holds.
It records per-video calibrated/uncalibrated single-frame exposure and
cache-age percentiles in video time. No new gate/decay/cap is introduced.

The seven-frame CPU witness checks the actual shared tracker body and
native/sham/previous-center equality with frozen E0. For block32/128 those
seven frames are complete, so it does not exercise long-duration cache use.
Only the full queued trajectories and final analyzer can verify that path.
Neither this witness nor source review proves CUDA/full-run success.

## Execution

Run `conda-envs/miss_rgbt/bin/python run_calibration_gap28.py --mode witness`,
then `conda-envs/miss_rgbt/bin/python launch_calibration_gap28.py`
from `/home/gaob/Miss-RGBT`.

The launch helper targets actual existing geometry controller1256793.
The new controller registers before results, waits for its actual exit,
requires the predecessor's final hash/bitwise/past-only verification, and
checks at least3000MiB free on each of GPU0/1 before allocating workers.
GPU2/3 remain occupied by their candidate trainers. Poll interval300seconds.

Output `reports/server28-calibration-gap4-20261031`, sibling `.log`;
launch helper receipt `reports/server28-calibration-gap-launch-20261031.json`.
Estimate60-120minutes after predecessor release, plus up to300seconds
queue delay; actual throughput may change this estimate.

Registration, witness result and actual process state must be recorded
separately after they exist. A launch or witness is not a completed result.

## Verified Entry And Registration

The CPU witness exited0 and passed sham/complete/previous-center equality
and frozen-network checks. Report `reports/calibration-gap-entry-20261031.json`,
SHA256 `050a54f4f770ce7a638b9adca160a0c7e1e3c4077e80c9bd9964a63e22a3dd7e`.
This retains the seven-frame scope and limitations above.

Controller1320176 registered2026-10-31 16:00:19 UTC+08 and was actually
observed alive16:00:45 waiting for1256793. Registration
`reports/calibration-gap-registration-20261031.json`, SHA256
`e9260c3a032c2bc3395904d974066fb479639318020707a18aeee1b150f7a3b7`.
Entry point SHA256
`9c1ae50b7a9efbad10d83480fe2e8db786d6d1a7512280c4a55ec5b53a56c06e`.
Server/download hashes match. No new GPU worker has launched.
All four trainers, center16 workers and preceding geometry queue remained
actually alive16:00:45. Preserve this new source and its registered shared
dependencies until final analysis; no experiment result is available yet.

## Completed Recovered Run

The recovery chain completed all64registered trajectories in
`reports/server28-calibration-gap4-20261031-recovered`. At18:46:36 UTC+08
the controller and both workers were absent, with final report/completion
present. The analyzer recomputed64trajectory metrics and verified source,
GT, checkpoint, trajectory and trace hashes; sham/complete bitwise equality;
past-only offset/scale reconstruction; clipping; calls and blind holds.
Downloaded report SHA256 matches the completion receipt:
`2ff1cb19cb250aeb4cc6f98d6b440de1b6f32633c246206cfb23375617c61ebd`.

| Condition | Native PR/SR | Center PR/SR | Geometry PR/SR |
|---|---|---|---|
| Complete | 85.409489 / 64.130333 | 85.409489 / 64.130333 | 85.409489 / 64.130333 |
| Block1 | 76.140126 / 56.606728 | 75.210148 / 56.922971 | 79.610563 / 60.690643 |
| Block32 | 62.353447 / 45.415382 | 56.979892 / 42.563832 | 63.272790 / 46.365210 |
| Block128 | 80.463083 / 57.892618 | 78.143502 / 56.187878 | 72.068849 / 53.086645 |

Sham equals native in every condition. All scores are macro development
PR20/SR percentages, not official benchmark results.

| Condition | Center-native SR delta [95% interval] | Geometry-native SR delta [95% interval] | Geometry-center SR delta [95% interval] |
|---|---|---|---|
| Block1 | +0.316243 [-0.497026, 0.896028] | +4.083916 [-1.532030, 11.896272] | +3.767672 [-1.117270, 11.428809] |
| Block32 | -2.851550 [-6.392466, -0.357359] | +0.949828 [0.493879, 1.251953] | +3.801378 [0.898792, 7.501759] |
| Block128 | -1.704740 [-4.886544, 0.396096] | -4.805974 [-12.499397, 2.405965] | -3.101234 [-11.813837, 2.167675] |

Geometry-native SR wins/losses are3/1,4/0,2/2 for block1/32/128.
Its block1 improvement is concentrated in leftbottle (+15.374031SR);
boyaftertworunboys worsens(-2.813379SR). The geometry-native SR effect
change relative to block1 is-3.134087[-10.845697,2.641323] at block32
and-8.889889[-24.395669,0.518459] at block128.

All videos remain included. Calibrated single-frame cache ages reach
1/32/128video frames in their respective schedules. The333frame video
has no RGB-only block128 exposure; only leftbottle reaches blind block128.
Four videos and their correlated conditions provide limited uncertainty
estimates. Schedule changes alter images, template writes and crop history,
so the interaction does not isolate cache staleness as the cause.

Decision: reject an unconditional center/scale correction as a generally
supported method. Block32 gains warrant mechanism analysis, while block128
regression is an observed problem that must be addressed and replicated
on fresh data before any module claim. No expiry, decay or gate parameter
is selected from this cohort. The current training campaigns are unchanged.

## Saved-Trajectory Mechanism Audit

New `audit_gap_mechanism28.py` executed successfully on all64saved
trajectories with zero new model calls. It rechecks registered sources,
GT and all worker/trace/bbox hashes, recomputes full metrics, and reconstructs
the geometry-arm raw, center-only and center-plus-scale boxes with native
clipping. The three boxes share one prediction and the geometry arm's
prior history. They are not separately rerun tracker trajectories.
Report SHA256:
`ae238e608c5f05739012810f542a529161d9e1375e388645a48e3960f359c8d3`.

| Block/state | Scored frames / exposed videos | Immediate center-raw SR delta [95%] | Immediate geometry-center SR delta [95%] |
|---|---|---|---|
| 1/TIR | 399 / 4 | -0.232392 [-0.895898, 0.191043] | +2.656282 [1.115904, 5.181252] |
| 1/RGB | 396 / 4 | +2.174622 [0.256892, 4.092352] | +1.816587 [0.513784, 3.091820] |
| 32/TIR | 448 / 4 | +0.117808 [-0.043403, 0.248016] | +1.475694 [-0.136409, 4.439484] |
| 32/RGB | 384 / 4 | -2.027530 [-9.375000, 2.585565] | +0.024802 [-0.186012, 0.241815] |
| 128/TIR | 640 / 4 | -1.046317 [-2.790179, 0.055804] | +2.264695 [0.120908, 4.724702] |
| 128/RGB | 268 / 3 | -0.313900 [-0.721501, 0.037202] | +0.591450 [-0.064350, 1.875902] |

RGB/block128 lists the unexposed333frame video outside that state slice;
it remains in whole-video aggregate scores. The scale addition helps
immediate TIR SR even at block128, while center translation has a negative
average. Whole-trajectory regression cannot therefore be reduced to a claim
that scale staleness is the sole problem. Prior crop/template history remains
involved; template-write changes do not isolate a causal template contribution.
These post-hoc comparisons select no parameter or gate and prove no module.
