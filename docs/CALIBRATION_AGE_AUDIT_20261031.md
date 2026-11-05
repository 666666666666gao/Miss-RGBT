# Calibration Age Coverage Gap

CPU-only mask audit, no GT access, neural calls, GPU allocation, parameters
fitted or running-source changes. Script: `audit_calibration_age28.py`.
Report: `reports/calibration-age-20261031.json`.
SHA256: `4cda9fa314e59d52ddc2264c7bd181fddb97b979d6ab6499d2caf4122da8a8d5`.
Both official mask hashes match the frozen E0 inventory. Download matches
the server report hash. A separate vectorized maximum-accumulate calculation
reproduced counts, uncalibrated counts, maxima and all four percentiles over
147248 calibrated single-modality frames.

## Definition

The current offset/geometry tracker initializes identity calibration at
frame0; initialization does not perform the11/01/10 calibration forwards.
The next actual11 inference refreshes the cache. On later01/10 frames,
age is current video index minus the last calibration video index.
Blind00 frames age the cache while making no model call.
The report separately measures observed-call age to avoid conflating clocks.

The stored first-frame mask is not always11: LasHeR245 has34 non11 initial
masks, RGBT234 has31. The actual evaluator still initializes from both images
and skips mask-driven inference on frame0. The audit follows that behavior;
it does not assume stored mask0 proves a full calibration occurred.

## Observed Distributions

These are frame-pooled descriptive distributions, not video-independent
confidence estimates. Long videos can dominate; per-video counts are saved.
States01/10 mean TIR-only/RGB-only.

| Dataset/state | Calibrated single frames | Median age | p95 age | Maximum | Age>1 | Before first calibration |
|---|---:|---:|---:|---:|---:|---:|
| LasHeR-Miss/01 | 55051 | 493 | 7273.5 | 8715 | 94.99% | 954 |
| LasHeR-Miss/10 | 42678 | 279 | 3628 | 7716 | 93.88% | 699 |
| RGBT234-Miss/01 | 24694 | 141 | 916 | 2482 | 93.97% | 96 |
| RGBT234-Miss/10 | 24825 | 128 | 1065 | 2066 | 93.83% | 815 |

Applying the current development block1 schedule to exactly the same video
lengths produces age1 on every calibrated single frame. There is one
uncalibrated first01 frame per video, and no uncalibrated10 frame.
Thus this is a schedule coverage difference, not a video-length explanation.

Observed-call age medians are405/211 for LasHeR01/10 and110/99 for
RGBT23401/10. Skipping blind frames does not remove the coverage gap.

## Consequence

The running center16 and queued geometry4 experiments must finish unchanged.
Even positive results would establish only the tested short-age control.
They cannot show that cached offsets or size ratios remain useful hundreds
of frames after calibration, or when calibration has never occurred.

Before promotion, register a separate closed-loop long-gap development
control with matched native/sham/center/geometry compute, full trajectory
analysis, past-only caches and unchanged identity priors. Prefer an existing
development outage schedule over choosing durations from official scores.
No decay, expiry, gate or clipping cap is justified by this distribution
alone. It identifies an untested regime, not a demonstrated stale-cache
failure or a successful fix. Do not change the registered current experiment.

This finding changes the next promotion requirement; no official score,
novel module or convergence claim is made.
