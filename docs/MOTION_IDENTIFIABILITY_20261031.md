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
