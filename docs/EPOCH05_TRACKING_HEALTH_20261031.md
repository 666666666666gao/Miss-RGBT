# Matched Epoch Five Tracking Health Check

The fixed128-tuple task loss fluctuates through epoch5 for both candidates.
It cannot establish tracking quality or support stopping from PR/SR.
This predeclared evaluation measures actual full validation trajectories
while all original training processes continue unchanged.

## Fixed Scope

- Use all97 LasHeR validation sequences,54646 video frames per condition,
  from the existing hashed recovery_val97 protocol.
- Models: original E0 availability checkpoint epoch20, and the already
  completed epoch5/update10000 batch10 utility/random checkpoints.
- Complete and fixed alternating-outage inputs for every model.
- Native MissingTracker, with dynamic template availability and updates,
  both-missing holds and original masks. Reuse the existing validation
  evaluator's `initialize` and `sequence` functions without modifying it.
- Macro sequence PR20/SR, excluding initialization/nonpositive-size GT,
  identical to candidate final-five validation. These development metrics
  differ from official benchmark metrics.
- GPU0 queue: E0 complete, utility complete, utility alternating outage.
  GPU1 queue: E0 alternating outage, random complete, random alternating
  outage. Exactly one evaluator per GPU; no new training.
- Existing historical official evaluation completed with this concurrency
  alongside the same GPU0/1 trainers. One worker previously profiled at
  approximately625MB Torch reservation; four workers caused a real OOM.
  Require at least3000MiB free at launch. Current free memory is about5GB
  per GPU. Do not increase concurrency or automatically retry failures.
- Record source/protocol/mask/checkpoint hashes, worker PIDs, each full
  sequence's bbox hash and metric receipt, plus97-sequence completion.

This check is fixed at epoch5 rather than selected from observed tracking
results. It does not select the final checkpoint, replace the existing
convergence rule, or use official test results for tuning. Final-five full
validation and common-weight official evaluation remain mandatory.
Observed E0 pretraining exposure to val97 remains uncertified.

## Runtime And Artifacts

Previous free-GPU eight-sequence throughput was about26-29frames/second.
Shared training slowed historical evaluation to approximately13frames/sec
per worker. Budget about60-90minutes per54646-frame condition, three
sequential conditions per GPU (roughly3-4.5hours, not yet measured here).
Re-estimate from completed sequence wall times. Inspect near condition
boundaries at180-300second intervals; no frequent update polling.

Runner `evaluate_epoch05_health28.py`, launcher `launch_epoch05_health28.py`.
Output `/home/gaob/Miss-RGBT/reports/server28-epoch05-tracking-health-20261031`,
with sibling log. Exclusive output/log creation prevents overwriting runs.
Frozen runtime and candidate final-five controllers are unchanged.

## Verified Launch

Controller570241 launched at2026-10-31T00:11:12.902742Z
(08:11:12+08:00). At08:11:45+08:00 it and both children were live:
GPU0 worker570254 and GPU1 worker570255. Each evaluator occupied2266MiB
according to the GPU process query, alongside the existing19290MiB trainer.
GPU2/3 original training processes remained allocated22702MiB each.
The E0 complete/outage output directories and launch receipts exist.
No sequence or aggregate result was yet complete at that observation.

The controller manifest is archived as
`reports/epoch05-tracking-health-launch-20261031.json`, SHA256
`a8843b4b21a2068d6e76e3b9123ec3836df49396dee5d75fa84138a1fa5b48ed`.
Runner SHA256 is
`937a6b5d0e65e25069cfd02d22cfc9047c570a9e629c6111205af09e64765e77`.
Check initial longest-sequence progress around08:23+08:00, then estimate
condition completion from actual speed. Launch checks prove execution
and bounded concurrency, not tracking correctness or improved metrics.

At08:17:23+08:00 both original evaluator PIDs remained live with substantial
CPU and read I/O, with no completed sequence or error. The longest-first
inventory starts with `ab_bolster`8406frames, followed by
`rightof2cupsattached`4857frames. At the prior13frames/sec estimate,
the first receipt takes about11minutes; an empty sequence log after6minutes
does not indicate failure. Do not restart the workers.

## Paired Analysis

`analyze_epoch05_health28.py` is prepared and syntax-checked in the server's
Python3.7 environment. It has not been run on complete results yet.
After all six conditions finish, invoke:

```bash
/home/gaob/Miss-RGBT/conda-envs/miss_rgbt/bin/python \
  /home/gaob/Miss-RGBT/analyze_epoch05_health28.py
```

It requires the complete six-cell receipt and all97 sequence receipts per
cell; verifies582 trajectory hashes, GT hashes, frame counts, protocol and
common source/weight identities; and recomputes every sequence's PR20/SR
from saved boxes before verifying aggregate metrics. Paired comparisons
are utility-minus-E0, random-minus-E0 and utility-minus-random in both
conditions. Report sequence wins/ties/losses, five largest gains/regressions,
and paired-sequence percentile bootstrap95% intervals using10000 draws
and fixed seed2027. All comparisons use the same resampling indices.

Intervals describe variation across these97 videos for one training seed,
not training-seed robustness or official performance. Do not select
checkpoints, tune thresholds or declare the research goal complete from
these development comparisons. The analyzer creates `paired_analysis.json`
exclusively and does not modify model weights or running evaluations.

## First Complete Sequence Verified

At2026-10-31T08:25:39+08:00 both E0 workers had completed `ab_bolster`:
8406 video frames,8393 scored frames, approximately733seconds per
condition (11.46frames/sec). All original training and evaluation PIDs
remained live; GPU0/1 total memory was21603/21571MiB.

| Single sequence | PR20 | SR |
|---|---:|---:|
| E0 complete | 65.864411 | 61.055982 |
| E0 alternating outage | 38.782319 | 42.084390 |

Both saved trajectory hashes match their receipts, and independently
reading the saved boxes through the prepared metric function reproduces
PR20/SR and scored-frame counts exactly. The first32 common complete-input
frames have bitwise equal boxes across the two GPU runs (max difference0).
This verifies the initial sequence's execution, not full validation quality
or a candidate gain. The utility/random epoch5 queues have not begun yet.

At this first-sequence rate, a full54646-frame condition takes about
79.5minutes. Remaining46240frames imply roughly67minutes after the first
receipt; plan the next combined observation around09:25-09:35+08:00,
near the first condition completion and candidate epoch6 validation.
Three sequential conditions per GPU project completion around12:10+08:00,
with substantial uncertainty from sequence resolution, I/O and model cost.
Use actual subsequent receipt rates before treating that estimate as firm.

Evidence: `reports/epoch05-health-initial-verification-20261031.json`,
SHA256 `7c3bc003f8c44fd56d26049280f1bfd497be008e434db378928913306487899a`.
