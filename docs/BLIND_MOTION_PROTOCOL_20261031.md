# Offline Blind-Interval Motion Control

Full-val97 state00 has the largest observed outage penalty. Existing
tracking holds the last box without a model call when both sensors are
missing. Before training or deploying a motion module, test the simplest
motion control on already saved train882 development trajectories.

Reuse the four registered protected-tracking development sequences and
both E0/selected-repair trajectories. Three videos contain blind intervals;
the150-frame bike video ends before the first00 block. Verify trajectory,
trace and GT hashes and native constant-box holds.

For each blind interval compare five fixed controls:

- Native last-box hold.
- Constant center velocity from the two immediately preceding tracker
  observations; retain the last observed size.
- The same tracker anchor but velocity from the preceding two GT boxes.
- Privileged preceding GT box held fixed.
- Privileged preceding GT box and GT center velocity.

No future GT enters a forecast. No clipping, velocity fitting, smoothing,
thresholds or alternative window search. GT forecasts are diagnostic
references, not deployable predictions or mathematical performance bounds.
Score only valid blind frames, giving per-video and macro/frame-weighted
PR20/SR. The last two observations are available before each actual gap.

This is an offline intervention. Observed-frame boxes are not recomputed,
so any benefit cannot be presented as closed-loop tracking gain or official
Miss performance. Changed gap positions would change later crops in a real
tracker. Three short intervals are insufficient for broad motion claims.
Keep normal inference unchanged until evidence warrants a closed-loop
development test with train/development separation.

Runner `audit_blind_motion28.py`, CPU NumPy only; no neural inference.
Output `/home/gaob/Miss-RGBT/reports/server28-blind-motion-20261031`.
Exclusive creation prevents overwriting the diagnostic.

## Offline Result And Closed-Loop Plan

Offline E0 blind-frame macro PR20/SR improved from28.125/27.926587
to64.583333/54.861111 with the fixed tracker-velocity forecast.
Selected-repair blind-frame scores improved from23.958333/27.480159
to56.25/46.577381. Each covers the same80 valid blind frames across
three videos; the two models are not independent replicates.
SR improved in all three videos for both models.

The privileged last-step GT velocity was much less effective and GT
anchor+velocity could be worse than a GT hold. A two-frame velocity is
no general motion oracle; annotation jitter and non-constant motion remain
possible explanations, not measured causal conclusions.

This positive fixed-control observation warrants one matched closed-loop
development run. Use GPU0, one evaluator, with at least3000MiB free.
Run E0 hold/velocity x complete/alternating-outage on the same four videos,
838frames per arm/condition,3352frames total. GPU0/1 have previously
supported one2266MiB evaluator alongside existing training; do not add
workers on GPUs2/3 or more than one onGPU0.

Initialize the native GPU tracker with the verified E0 availability package,
which certifies unchanged baseline tensors and zero optimizer updates.
Recompute all observed-frame predictions/crops/templates for each arm.
When both sensors are absent, the velocity arm uses the last two observed
video-frame boxes, normalizing by their true frame separation, retaining
size and leaving native model-call/template counters unchanged.
The hold arm remains native. No GT enters the predictor.

Verify same-GPU complete trajectories and the no-blind video bitwise;
outage prefixes must match before the first blind frame. Verify unchanged
network tensors and native hashes. No ordinary CV control is promoted to
a new paper contribution. No parameter search or official test evaluation.

Runner `evaluate_blind_motion28.py`, launcher `launch_blind_motion28.py`.
Output `reports/server28-blind-motion-closed-loop-20261031`, siblinglog.
Estimate5-10minutes under shared training; refine from actual trajectory
receipts. Check actual PID/allocation once, then inspect at300seconds.
Final evidence requires report.json AND completed.json. An SSH timeout
does not authorize restarting the evaluator.

## Verified Closed-Loop Result

PID925753 finished normally in237.57seconds. The independent analyzer
verified all16 trajectory/trace hashes, GT hashes, per-video and aggregate
scores. All complete-condition boxes, no-blind video boxes, and pre-blind
prefixes are bitwise identical between arms on the same GPU. Model
tensors remained unchanged. The evaluator released GPU0.

| Condition/group | Videos | Frames | Hold PR20/SR | Velocity PR20/SR | Delta, pp |
|---|---:|---:|---|---|---|
| Complete | 4 | 834 scored | 98.848739/78.647520 | 98.848739/78.647520 | 0/0 |
| Outage, all valid | 4 | 834 | 89.324968/70.910351 | 91.134491/72.246885 | +1.809524/+1.336535 |
| Blind only | 3 | 80 | 23.958333/26.785714 | 43.750000/40.873016 | +19.791667/+14.087302 |
| Observed after blind | 2 | 128 | 99.218750/83.296131 | 99.218750/83.482143 | 0/+0.186012 |
| Both observed after blind | 2 | 64 | 100/88.616071 | 100/88.839286 | 0/+0.223214 |

All three blind-containing videos improve in full-outage SR:
2up+0.136054, comingboy'shead+2.278245, wanderingly_1202+2.931839pp.
The no-blind bike video is unchanged. Most aggregate benefit is on blind
frames; the observed recovery gain is small. Group macros have different
video membership and must not be added.

The GPU control's predictions differ from the earlier CPU trajectories;
the offline CPU gain is not its expected numerical gain or an independent
replication. Causal arm comparisons use the freshly matched same-GPU
trajectories. This is still only four reused development videos, not an
official result, generalization claim, or a paper contribution.

Report `reports/blind-motion-closed-loop-report-20261031.json`, SHA256
`faead2d14bc4580fb2efba9caac3365a422245058ff234bb3c276ee7b80ea4a5`.
Independent analysis `reports/blind-motion-closed-loop-analysis-20261031.json`,
SHA256 `07e20f91cb1d79b2f7069536942c53382094afb46ce0e72ed132a8ca19086794`.
Offline report SHA256
`ad87a72040a3254661da7e30eeb88491e0203bcb2eb296493fbd0bf38a9f225b`.

## Existing Motion Reference

Inspected `noahcao/OC_SORT` commit
`8462e7e729a93ccd3bd995c0a79a890336cb3a0b` (API date2026-04-21).
Its `trackers/ocsort_tracker/ocsort.py:70-152` supplies a constant-velocity
Kalman box state and separate observed history. In
`kalmanfilter.py:419-516`, recovery uses the newly arrived measurement to
replay a virtual path and update the filter. That recovery operation
cannot provide a forecast before the measurement arrives.

This is an existing multi-object tracking baseline to compare against,
not a new contribution. Our single-target pipeline still needs a reliable
visual measurement after an outage; a motion filter does not verify target
identity. The pinned root LICENSE reads MIT. No code/dependency was
integrated. Inspection files are outside the publication checkout.

Pinned base:
`https://github.com/noahcao/OC_SORT/tree/8462e7e729a93ccd3bd995c0a79a890336cb3a0b`

Downloaded source SHA256:
`ocsort.py`:122cc1c27d3321eac0debdf3ba782b76d189e1ac878256b8291ffeb4314fa979;
`kalmanfilter.py`:165a2d5243f2b874e37d3a3ae8112402f5c53ddfb6ebe6e2cddda3311c3a2cded.
