# Frozen E0 / Selected LocalRepair Tracking Control

This follows the fixed six-epoch CPU branch pilot. It tests the selected
branch in closed-loop tracking rather than assuming sampled task loss
predicts tracking quality. It is a control using existing LocalRepair,
not a new module or evidence of novelty.

## Fixed Before Execution

- CPU only, four threads, no modification to frozen formal runtime.
- Wait for existing pilot PID61899 at300-second intervals. No restart.
  Continue only after its successful192-update completion, all six
  development records, final-five selection, unchanged complete outputs,
  unchanged baseline, selected branch SHA and four-weight cleanup checks.
- First four names in the pilot's177-sequence development partition, in
  stored order. Evaluate the first257 video frames, or the entire sequence
  if shorter. These sequences are disjoint from the705 pilot-training and97
  heldout sequences. The development partition selected the branch epoch;
  these are descriptive development results, not an unseen test.
- E0 availability-aware wrapper versus the same frozen E0 plus selected
  LocalRepair. Restore raw E0 strictly for each model; only the selected
  branch's state differs. No composite weight export is required.
- Complete observations versus fixed32-frame blocks
  `11,01,11,10,11,00`, repeating. Both-missing frames hold the previous
  box without a network call or template update.
- Reuse the existing CPU tracker crop, Hann decoding and native dynamic
  template policy. Do not inject GT after first-frame initialization.
- Record every box and observed crop-support diagnostic. Check complete
  boxes match E0 bitwise across each evaluated prefix, both models match
  for the first32 complete frames of the outage schedule, and baseline
  parameter/buffer hashes remain unchanged.
- Development macro PR20 and SR use the existing metric definitions:
  center error <=20px; mean success at21 thresholds0..1 withIoU>threshold;
  exclude initialization and nonpositive-size GT. Include both-missing
  holds in scored frames. Also report meanIoU and lost-support counts.
- No tracking-based epoch selection, parameter adjustment or checkpoint
  cleanup. The six-epoch pilot remains the sole selection authority.

## Scope and Timing

At most4112 video frames across four model/condition combinations.
Based on the earlier fixed8 CPU cohort, estimate30-45 minutes after
the pilot finishes, depending on actual selected sequence lengths.
Check near the estimated completion, with180-300 second observations.

This is neither full-sequence/full-development validation, official
LasHeR/RGBT234/Miss evaluation, GPU equivalence nor proof of convergence.
Complete-output equality is tested on complete histories; mixed histories
may diverge after outage and are part of the tracking comparison.

Runner: `evaluate_protected_development28.py`.
Launcher: `launch_protected_tracking28.py`.
Output: `/home/gaob/Miss-RGBT/reports/server28-protected-tracking-development-20261030`.
Log: the sibling `.log` file.

## Verified Queue and Actual Cohort

PID96513 started2026-10-30T19:51:26Z. At19:51:34Z its live command
line and launch record were checked; it was sleeping while the original
pilot61899 remained live. Both files compiled with the server Python3.7;
deployed runner/launcher SHA256 matched local sources. No tracking result
exists yet, and successful syntax/queue checks do not establish numerical
tracking correctness.

The fixed first-four development sequence selection produced:

| Sequence | Total frames | Evaluated frames |
|---|---:|---:|
| 2up | 176 | 176 |
| comingboy'shead | 256 | 256 |
| wanderingly_1202 | 256 | 256 |
| bike2north | 150 | 150 |
| Total per model/condition | 838 | 838 |

All four happen to fit within the predeclared257-frame budget, so the
actual experiment covers these four sequences in full,3352 video frames
across two models and two conditions. The launch scope's conservative
"not full sequences" wording describes the budgeted general protocol;
the actual frame counts above establish full coverage of this small cohort,
not full177-sequence development or97-sequence validation. Revised runtime
estimate25-40 minutes after the pilot ends. The selected repair itself
has not yet been loaded or evaluated.

## Numerical Entry Verification

`verify_protected_tracking_entry28.py` completed successfully on server
Python3.7/Torch1.12.1 in73.00 seconds. On the first33 frames of2up,
including the first missing-search frame, E0 and E0+zero-initialized repair
produced bitwise identical boxes in both conditions. All four tracks had
32 neural calls, finite boxes, and unchanged baseline state hashes.
The script asserts equality of the entire box arrays, not just matching
aggregate metrics. No selected trained branch was used.

Receipt: `reports/server28-protected-tracking-entry-smoke-20261030.json`;
local copy: `outputs/protected-tracking-entry-smoke-20261030.json`.
Runner SHA256:
b340708f391ef41bf47ee3b03a4933af5c293248951a3d18180776c5ad4f5f32.
This proves the bounded zero-branch entry comparison, not GPU parity,
trained-branch benefit, later missing-template behavior or benchmark gain.

## Selected-Branch Run In Progress

The branch pilot completed normally at2026-10-30T20:17:09Z. Its preregistered
development selection chose epoch3 of192updates; report SHA256
6730fc3ceacddd2316de175d1cc2e60724ce23a43fa441215b9bbd678fcfdc8d.
Queue PID96513 verified the six histories, selected-file SHA, four cleanup
receipts, unchanged baseline, and bitwise complete-output checks before
loading it.

At20:19:59Z the same queue process remained live. It had completed E0 on
2up for complete (PR20 97.1429, SR62.4762) and alternating outage
(PR20 96.5714, SR60.2449); 16 both-missing holds and159 tracker calls in
the outage condition. These are single-sequence CPU development observations,
not a reported paired gain; selected-branch comparison remains pending.
The next observed sequence was not yet complete at that check. Do not
extrapolate these values to the four-sequence cohort or official metrics.

At20:27:29Z, the queue remained live and had completed E0 on the first two
sequences in both conditions and the third sequence in complete condition.
Observed runtimes ranged from86.59 to544.99seconds per sequence-condition;
the initial25-40minute post-pilot estimate was optimistic. Re-estimate after
the E0 pass from measured outage and bike2north durations. Keep the single
CPU process and do not raise concurrency while the GPU campaigns use host
resources. The selected-branch comparison remains pending.

At20:31:55Z the E0 pass had completed all four sequences in both conditions.
Their unweighted four-sequence macro PR20/SR are:

| Condition | PR20 | SR |
|---|---:|---:|
| Complete | 98.991597 | 78.587389 |
| Alternating outage | 89.414603 | 71.045377 |

These are from four short train882 development videos, not a benchmark or
LasHeR-Miss estimate. The selected branch then completed its first full
sequence condition, E0/repair complete on2up, with identical bbox SHA
`b8489ffa21598d6dad8d783c1c199d49cb22589a0c44acc80a9fb699c576556e`.
This confirms equality on that one complete trajectory; remaining selected
conditions and the paired aggregate are still running.

At20:31:55Z PID96513 had run40:30 total. Selected-branch tracking on the
same CPU is materially slower than the E0 pass (about15.1 minutes for
complete2up versus1.4 minutes for E0/2up), so the initial completion
estimate no longer applies. Continue checking only near observed completion
milestones, with180-300 second intervals; do not add a competing CPU/GPU
job. Progress receipt: `tracking_progress.json`.

By20:39:48Z the selected branch had completed both conditions on2up and
comingboy'shead, plus complete onwanderingly_1202. All three complete
trajectories match E0 bbox SHA exactly. Paired alternating-outage differences
for the first two are heterogeneous:

| Sequence | E0 PR20/SR | Selected repair PR20/SR | Difference, pp |
|---|---:|---:|---:|
| 2up | 96.5714 / 60.2449 | 94.2857 / 61.7959 | -2.2857 / +1.5510 |
| comingboy'shead | 88.6275 / 72.0822 | 88.6275 / 72.6050 | 0 / +0.5229 |

Only two of four outage pairs are complete. The sequence-specific results
are mixed and not evidence of a general gain. Repair arm is slower: roughly
15-22 minutes per completed sequence-condition in the observed samples.
