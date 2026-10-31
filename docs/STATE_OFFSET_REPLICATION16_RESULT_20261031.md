# Sixteen-Video Offset Replication Result

All96 registered full-video trajectories completed, including the6158-frame
leftconergirl sequence. The original independent analyzer verified source,
registration, worker/report/trajectory and GT hashes, recalculated per-video
PR/SR, checked every saved past-only correction, and verified exact sham and
complete-condition equality. No sequence was dropped after the reboot.

The report SHA256 is
`246af77cac7b3272aa5adff9599839e75612250b9d5eed263f4203c3bf233fab`;
the downloaded report matches the server and completion receipt.
These are16 fresh registered development videos, not full val97 or official
LasHeR-Miss/RGBT234-Miss results.

## Full-Trajectory Result

| Condition | Arm | PR20 (%) | SR (%) |
|---|---|---:|---:|
| Complete | Native/sham/corrected, all identical | 90.656936 | 73.839150 |
| Synthetic outage, block1 | Native/sham, identical | 86.550590 | 71.114156 |
| Synthetic outage, block1 | Offset corrected | 87.301072 | 71.472439 |

| Outage Effect | Mean Delta, pp | Paired-Video 95% Interval | Wins/Ties/Losses |
|---|---:|---|---|
| PR20 | +0.750482 | [-0.140864,1.691871] | 9/3/4 |
| SR | +0.358282 | [-0.326380,1.071918] | 11/0/5 |

The small positive average does not demonstrate a stable full-trajectory
benefit. Both intervals include zero. Worst SR changes are rightof2boys953
-2.285914 and left4thgirlwithwhitepants -2.134381pp; best is
rightwhitegirlleftpink +3.764252pp. Neither failures nor the long video
were excluded from the result.

## Saved Same-History Mechanism Audit

The unchanged `audit_offset_mechanism28.py` then analyzed the completed16
videos without additional model calls. Its report SHA256 is
`a5f42c576ccf5afec3bbdf35f7aee7f071d3efd0aa6107dc5a09f59befd2e748`,
also matched after download. Source/worker/trajectory/GT hashes and full
per-video scored-frame counts/PR/SR were rechecked.

Immediate comparisons use the corrected tracker's same saved raw prediction
and history, comparing its clipped uncorrected and clipped corrected box.
They do not compare two independently evolving trackers.

| State | Immediate Center Error Delta, px [95% Interval] | PR Delta, pp [95% Interval] | SR Delta, pp [95% Interval] |
|---|---|---|---|
| TIR-only01 | -1.929761 [-5.125844,0.139422] | +1.312371 [-0.562840,3.387505] | +1.256158 [0.063631,2.555410] |
| RGB-only10 | -0.865772 [-3.174648,0.622491] | +2.160315 [0.142191,5.306046] | +0.162236 [-0.521907,1.017431] |

Immediate TIR-only SR improves11 videos and worsens5. RGB-only PR improves7,
ties7 and worsens2, but RGB-only SR improves7 and worsens9.
Error, threshold precision and overlap do not necessarily improve together.
These post-hoc strata and intervals are descriptive, not an independent
prospective state-specific gating experiment.

Closed-loop native-versus-corrected outage SR slices are:

| Availability | SR Delta, pp | Paired-Video 95% Interval |
|---|---:|---|
| 11 | +0.257426 | [-0.387166,1.026359] |
| 01 | +1.246706 | [-0.326824,2.798687] |
| 10 | -0.144857 | [-0.981264,0.737872] |
| 00, inherited held boxes | +0.236168 | [-0.437850,0.996357] |

Template-write decisions change in14/16 videos. Equal decisions in the other
two do not establish equal template contents because crops can differ.
This audit cannot isolate a template mechanism from crop/state recurrence.
The entirely complete condition remaining bitwise equal is consistent with
different11 frames inside an outage trajectory, where histories diverge.

## Next Executed Stage And Decision

The existing recovery controller72348 launched registered geometry workers
151970/151971 at approximately17:49:44 UTC+08, on GPU0/1. Both were actually
alive17:51:25 and had completed native full-video tracks with saved receipts.
Four main trainers and the CPU compensation pair remained alive17:52:16.

Geometry uses the unchanged fresh four-video registration,32 trajectories;
after independent verification, the existing chain runs the registered
four-video long-gap64-trajectory control. Provisional geometry completion
18:00-18:15 UTC+08, then up to300seconds for controller wakeup.
Do not alter registered sources, choose a TIR-only gate or promote center
calibration into a paper module from these results. Generic calibration
remains an ordinary control, and longer-gap efficacy remains unverified.

Published artifacts:

- `reports/state-offset-replication16-recovered-20261031.json`
- `reports/state-offset-replication16-recovered-completed-20261031.json`
- `reports/state-offset-mechanism16-recovered-20261031.json`
- `reports/spatial-calibration4-recovered-launch-20261031.json`
- `reports/spatial-calibration4-recovered-children-20261031.json`
