# Motion History Identifiability

## Exact Saved-Trajectory Audit

`audit_motion_identifiability28.py` verifies both completed block1 studies,
including every blind frame regardless of annotation validity.
Each of the adjacent and availability studies has1290 blind frames across
the same16 videos. Every blind frame has:

| History | Earlier observation | Latest observation | Time interval |
|---|---|---|---:|
| Adjacent | 10 | 11 | 1 |
| Availability matched | 11 | 11 | 2 |

The matched earlier frame is always exactly `last-2`. The matched velocity
is exactly the ordinary two-frame difference on every saved history.
Reconstructed blind boxes agree with saved outputs with maximum error0.
The audit uses no GT or new model inference.

This strengthens the earlier confounding caveat: the registered schedule
makes state matching and lag2 perfectly coupled. Even a positive fresh-video
replication under this schedule cannot establish a sensor-state mechanism
or distinguish the predictor from ordinary lag2 velocity. Do not add a
third identical lag2 GPU arm or count this as a novel module. The current
fresh replication still tests whether this ordinary rule generalizes
across videos; its fixed protocol remains unchanged.

Evidence: `reports/motion-identifiability-20261031.json`.
The reported1290 includes3 invalid-GT blind frames omitted from the
earlier1287-frame accuracy diagnostic.

## Equal-Lag Prior-Availability Probe

CPU runner `run_fixed_crop_motion28.py` executes a bounded factorial probe
and then `analyze_fixed_crop_motion28.py`. It makes576 model calls:
16 reused videos,4 early blind targets(5,11,17,23),3 preceding observation
times and3 availability states(11,01,10).

For each observation time, every availability state uses exactly the
saved complete-track previous box as crop center/size. The first-frame
complete templates remain fixed. The probe resets the tracker call index
before each forward to prevent template updates and asserts template
identity. The E0 model is frozen and its state digest must remain unchanged.
Unavailable image channels follow native substitution and availability
gating, while crop geometry remains fixed.

Forecast comparisons hold the latest observation at11. At lag1 and lag2
separately, vary the earlier observation state11/01/10. The probe records
actual decoded boxes and velocities. Offline GT yields velocity errors
against the matching true time interval and next-frame center errors.
No current/future GT generates a crop or prediction.

The independent analyzer reconstructs velocities and GT metrics, verifies
source/completion receipts, and reports paired-video bootstrap intervals.
Four triplets/video are averaged before resampling16 videos.
The384 metric rows are not independent video samples.

This can test whether earlier availability affects velocity errors at a
fixed lag in this context. It cannot by itself distinguish state consistency
from the greater information in a complete observation: the latest state
is always11. Fixed initial templates, early frames, reused videos and
GPU-derived reference crop histories further restrict transfer. No CPU/GPU
equivalence, closed-loop recovery, official score or novelty claim follows.
Comparing different lags may use different valid-GT subsets.

Started05:55:35Z, runner1109528/worker1109529. First video36 calls took
22.98seconds; estimate6-8minutes overall, inspect around14:02-14:04 UTC+08.
Output `reports/server28-fixed-crop-motion-factorial-20261031`;
log sibling `.log`. No extra GPU allocation or main-training change.

## Completed Equal-Lag Probe

All576 model calls and384 diagnostic rows completed in303.75seconds.
Independent analysis reconstructs velocities and GT errors and verifies
source hashes and unchanged model state. Report SHA256:
`5dd9c71105a49ac525ad6bfd1c75be630a38e6108cfc30206d190bbcb27ff596`.

With latest state11, same11 versus prior01 lowers velocity error by
4.001169px/frame at lag1 and1.778563 at lag2; respective paired-video95%
intervals are[-7.963059,-1.518081] and[-3.716948,-0.488352].
Comparisons with prior10 cross zero at both lags. Independently, lag2
reduces velocity error for all three fixed prior states on all16 videos;
this confirms a temporal-lag effect in the fixed-crop diagnostic.
Forecast-center error does not improve consistently for every contrast.

## Reused 3x3 State Factorial

The same saved outputs contain all three states at frames2/3,8/9,14/15,
20/21. Reuse these pairs with forecast frames4/10/16/22, one frame earlier
than the original targets. Vary both current and prior states at identical
lag1. This adds no model calls and is an explicitly post-hoc diagnostic.
All64 triplets have valid GT; all nine cells use the same triplets.

Video-macro velocity error in pixels/frame (row=current, column=prior):

| Current / Prior | 11 | 01 | 10 |
|---|---:|---:|---:|
| 11 | 2.274888 | 6.434540 | 3.246100 |
| 01 | 6.346440 | 3.434340 | 7.472402 |
| 10 | 3.625227 | 7.643225 | 2.589249 |

The same-state cell has the lowest macro velocity error in each row.
For current01, same01 minus prior11 is-2.912099px/frame
CI[-6.467046,-0.771541],15/16 videos lower error.
For current10, same10 minus prior11 is-1.035978
CI[-2.190000,-0.214201],12/16 lower error. Thus complete prior observation
quality alone does not explain this fixed-context pattern. All six
same-versus-different velocity contrasts have descriptive95% intervals
below zero; they are not independent tests or seed uncertainty.

The corresponding same-minus-complete forecast-center error differences
are-4.026711px (current01,CI[-8.058217,-1.547097]) and-1.162301px
(current10,CI[-2.486863,-0.183217]). Current11 versus prior10 forecast
contrast crosses zero. This is not a general guarantee or closed-loop gain.

`analyze_motion_state_factorial28.py` verifies the parent/completion/analysis
receipts before reconstructing the saved pairs. Published
`reports/fixed-crop-motion-state-factorial-20261031.json` contains all rows,
per-video counts, metrics and contrasts.

## Registered Fresh Replication

At14:07:20 UTC+08 runner1129716 started a300-second wait on the actual
adjacent fresh-video controller1090400. Both workers had completed32/32
tracks at14:07:46; controller final checks were still pending.
The runner requires a hash-valid completion, then repeats the same576-call
probe and both independent analyses on the registered development20..35.
It verifies all queued source hashes before execution. Finished original
CPU sources were archived/hash-checked before adding explicit path arguments.
No pending GPU evaluator/controller sources changed.

New output `reports/server28-fixed-crop-motion-replication16-20261031`,
log sibling `.log`. Expected CPU start by14:12-14:17 and finish14:18-14:25,
subject to actual reference completion. This cohort is fresh to the
factorial diagnostic, while its motion study remains development evidence.
No parameter fitting or official-score selection is introduced.

At14:13:40, fresh CPU worker1138583 and runner1129716 were verified live;
108/576 calls completed in65.01seconds. Automatic receipt and3x3 analyses
remain queued. Inspect around14:18-14:22.

## Fresh Adjacent Motion Arm Completed

At14:12:13 the reference controller had exited with a valid64-trajectory
completion. Complete hold/velocity scores match at PR20/SR83.865751/
69.620978. Full outage hold79.572046/65.009895 versus adjacent velocity
90.016246/71.278588 gives+10.444200/+6.268693pp. Paired-video95%
intervals[-0.283139,25.678533]/[-0.240559,15.576421] still cross zero.
SR improves10 videos and declines6. Two large gains dominate:
`motocomeinrain`+52.380952pp SR and `Athe3rdboybesidescar`+46.652864.
`redcupatleft` declines3.791436pp SR. Do not treat this variable mean
as stable generalization or use per-video outcomes to alter the cohort.

Reference report SHA256:
`b77bf568199a27aded01559eccc54c22fd7c10879eab4b838ca6d734965783de`.
The availability/lag2 arm started at about14:08:47; controller1132262 and
workers1132272/1132273 were verified live14:12:13. Its final paired
comparison is still pending. Active GPU execution sources remain unchanged.
