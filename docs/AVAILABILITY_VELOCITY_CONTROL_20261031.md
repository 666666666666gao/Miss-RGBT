# Availability-Matched Velocity Diagnostic

The fixed32-frame schedule estimates velocity from two11 observations.
The fixed1-frame schedule instead estimates from10 followed by11.
Long/short comparisons therefore also change sensor-state history.
This follows directly from the registered schedule, not benchmark scores.

Before examining the final short-gap result, prepare one offline control:
anchor all forecasts at the same saved adjacent-CV last observed box.
Compare holding that box, velocity from the last two observations, and
velocity from the last two observations with the same availability state
as the last observation. Divide both velocities by actual video-time
separation. Preserve the same box dimensions and predict only the
registered one-frame blind intervals.

No current or future GT enters these forecasts. Valid blind GT scores
them afterwards. Past GT is used only to report velocity-estimation error.
There is no confidence threshold, fitted window, motion cap, learned
parameter or data-selected option. The control uses all16 registered
development videos and all their valid blind frames.

The analyzer verifies completion/report/trajectory/GT/trace hashes,
the actual10->11 input pair, and reproduction of the saved adjacent-CV
blind predictions. It uses paired16-video bootstrap10000 draws, seed2027.
Same-state matching also lengthens the observation interval, so any
difference cannot be attributed solely to removing sensor-state bias.

This is offline reuse of one trajectory history. It cannot establish
closed-loop recovery behavior, crop/template updates, official scores,
generalization, or a novel paper contribution. Only a subsequent matched
closed-loop run could test the effects of changing the predictor.

Analyzer: `audit_availability_matched_velocity28.py`.
Output: `reports/server28-blind-motion-shortgap16-20261031/availability_velocity_audit.json`.
Exclusive output creation preserves the first result.
At13:14:05 the short-gap workers had exited and the existing controller
was still awaiting its scheduled300-second completion check/analyses.
No final short-gap or availability-matched result existed then.

## Verified Short-Gap Results

The completed short-gap run verified64 trajectories on all16 videos,
7776 total frames per arm/condition and7742 valid scored frames.
Complete outputs remain84.909989 PR20 /69.541361 SR for both arms.
Outage hold is80.858060/65.136455; adjacent-CV is81.693045/66.032080.
The paired gain is+0.834986/+0.895625pp. Bootstrap95% intervals
[-0.679692,2.966112]/[-0.177558,2.483089] both include zero.
SR improves11videos and declines5; PR improves8,ties2,declines6.

The independently verified1-frame-minus32-frame CV effect contrast is
-0.320611PR20/+0.048866SR, with95% intervals
[-2.888904,2.045985]/[-1.982117,1.833271]. Neither establishes a
duration-specific advantage. Complete trajectories agree bitwise across
the two studies, with identical video/GPU/checkpoint assignments.
These development schedules are not the official published masks.

## Offline Availability-History Result

On1287 valid blind frames across16 videos, all anchored to the same
saved adjacent-CV last observed states:

| Offline predictor | Blind PR20 | SR |
|---|---:|---:|
| Same-last hold | 78.757178 | 62.689511 |
| Adjacent observations | 80.809015 | 65.654678 |
| Same-availability observations | 81.473186 | 66.201395 |

Matched-minus-adjacent is+0.664170/+0.546717pp. Its video bootstrap
intervals are[-0.262309,1.649594] PR20 and[0.170715,1.007952] SR.
SR improves12,ties1,declines3. This is a positive blind-only development
signal for a closed-loop control, not evidence of a novel contribution.
Matched observations are two frames apart here; adjacent are one apart,
so the state alignment and temporal averaging effects are confounded.

Short-gap report SHA:
`9a076d9b77cd9c1b0c81a9ecec9e649f2b6e72a6e77cc1850a987900b29649a9`.
Availability-audit SHA:
`88e9120f0263215662e6bb7a6307919a74ce7c584354cac92ab932253402456d`.
Cross-gap contrast SHA:
`53b44c70a1830974b30f0bf422bb6c152ed32841106dafaae68c831715b4c952`.

## Closed-Loop Follow-Up

Keep the same16 videos, physical GPU assignment, checkpoint, block1 cycle,
hold control, complete conditions, box dimensions, neural tracker and
template rules. For the velocity arm, retain the last two observations per
availability state and use the history matching the last observed state.
Use actual video-frame separation. This registered schedule supplies two
matching11 observations before every blind frame; assert that condition.
No fallback for unrelated schedules or general official-mask deployment is
introduced. Adjacent-CV remains the default evaluator mode.

The original short-gap controller had exited and all final receipts existed
before its evaluator/controller/analyzer sources were archived under
`server28-blind-motion-shortgap16-20261031/verified_source/`, checked against
launch hashes. The new evaluator adds `--velocity-history availability`.
It and the controller compiled with server Python before launch.

At13:17:23 UTC+08 controller1044705 and GPU workers1044714/1044715 were
verified live, each evaluator allocating2266MiB beside the existing trainers.
Started about13:17; budget20-30minutes, inspect near13:37-13:47.
Output `reports/server28-state-motion-shortgap16-20261031`.

Comparison process1049221 was verified live at13:19:32, waiting on the
actual controller every300seconds. `analyze_state_motion_control28.py`
requires both completed64-track studies and worker analyses, equal
cohort/GPU/checkpoint assignments, bitwise equal complete trajectories and
bitwise equal hold-outage trajectories. It compares the two velocity arms
with paired-video intervals for the full outage and blind/recovery groups.
Log `reports/server28-state-motion-comparison-20261031.log`.
Final result is `comparison.json` in the new run directory.

The same videos motivated this follow-up, so a favorable result still needs
fresh development replication and eventual official evaluation without
official score-based tuning. No closed-loop result existed at registration.
