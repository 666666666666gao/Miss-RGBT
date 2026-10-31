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
