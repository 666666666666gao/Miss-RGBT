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
Check initial longest-sequence progress around08:20+08:00, then estimate
condition completion from actual speed. Launch checks prove execution
and bounded concurrency, not tracking correctness or improved metrics.
