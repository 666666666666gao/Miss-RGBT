# Prospective Scale Component Control

The completed gap control shows geometry-native SR+0.949828points at
block32 but-4.805974at block128. Saved geometry-history arithmetic shows
block128 TIR center translation-1.046317SR and scale addition+2.264695SR.
These are distinct comparisons: the immediate boxes share geometry history,
whereas the whole trajectories include earlier crop/template differences.
They motivate a scale-only component control, not a fitted expiry gate.

## Registered Design

Use sequential development entries64..67 of the protected manifest, without
outcome or duration filtering, disjoint from its train705 and heldout97
populations and all earlier development entries. Each video gets native,
sham, scale-only and center-plus-scale arms in complete/block32/block128:
48full trajectories. E0 weights remain frozen. Scale-only sets
apply_offset=False and apply_scale=True in the existing isolated
SpatialCalibrationTracker. The algorithms, coefficients, native confidence
and template rules are otherwise unchanged; whole trajectories may diverge.

At complete frames, calibration uses the same present image with synthetic
single-modal forwards. During missing frames, apply only the last past
complete-frame width/height ratios, centered on the current prediction.
Both-missing frames hold the prior state without a network call. No GT
chooses corrections; GT initializes and scores tracking.

Use the same fixed six-state mask pattern as the prior control. Longer
blocks change all input histories; this does not isolate cache age.
Keep short videos and record unexposed states. Evaluate PR20/SR with paired
video intervals for scale-native, geometry-native and scale-geometry.
Report every condition and block128 effect minus block32 effect; no selection
of the best condition as a benchmark gain.

## Execution And Verification

New standalone run_scale_only28.py follows the existing gap-control
entrypoint without modifying registered predecessor or frozen runtime files.
The CPU seven-frame actual-body witness exercises complete and block1
states, checks exact sham/complete equality, zero scale-only offset,
center-preserving unclipped scaling, nonidentity scale use and frozen state.
This is not CUDA constructor or long-duration trajectory verification.

The controller requires the completed gap report hash and checks at least
3000MiB free on GPU0/1 before workers allocate. One evaluator per GPU runs
beside the fixed batch8 trainers; GPU2/3 remain with batch10 candidates.
The observed available headroom before implementation was4779MiB per
GPU0/1; prior same-tracker evaluation used approximately2.2GiB per worker.
No environment installation, trainer restart or runtime edit is involved.

The final analyzer recomputes all48trajectory metrics and checks original
input/source/checkpoint/trace/bbox hashes, actual scale-only zero offset,
past-only scales, complete/sham equality, clipping, blind holds and model
call counts. Until those checks finish there is no completed result.
This ordinary component control is not a validated novel paper module.

Output reports/server28-scale-only4-20261031; sibling .log.
Run CPU witness, then launch_scale_only28.py. Poll300seconds near the
estimated completion, using actual assigned frames and observed throughput.
Preserve the new source and shared dependencies until analysis completes.

## Cohort Scope

| Index | Video | Frames |
|---|---|---:|
| 64 | rightgirlbike | 190 |
| 65 | whitegirlcrossingroad | 154 |
| 66 | boybesidescarwithouthat | 318 |
| 67 | rightblkboybesidesred | 219 |

Total881frames, assigned472/409frames to GPU0/1 by duration balancing.
Every video ends before frame384, so block128 contains no RGB-only
interval. None reaches the first blind block128 interval. This cohort
tests only TIR missing-state calibration at block128 and must not support
a claim of long-gap RGB or blind robustness. Every video remains included
in all three whole-video conditions. The cohort was not filtered to achieve
this duration profile.

Initial expected run duration10-25minutes after workers launch, plus up to
300seconds controller observation delay; refine from actual completed tracks.
There is no trained weight or checkpoint cleanup in this diagnostic.

## Actual Launch

The entrypoint and launcher compiled under Python3.7. The CPU actual-body
witness completed successfully: sham and complete equality, zero scale-only
offset, frozen network, and2nonidentity scale frames. It retains the limited
seven-frame scope described above.
Witness SHA256:
`0d3fb33841f505c8b1340340c82336e3be75f5cad9cb48af08dfbc2cd1ec6d05`.
Entrypoint SHA256:
`acf6a56e84cbfdfa406ef7904664627ea52e899f3933e37e25afe2de2e182086`.
Registration SHA256:
`c8335c3a96ba57e2ad3fbc3a96c3f764e15f64304095af0f55079c358a51ed10`.

Controller268017 launched at19:18:14 UTC+08. At19:18:33 it was actually
alive; GPU worker268025/268026 appeared in nvidia-smi with2124/2080MiB,
alongside all four existing trainers. Initial full-track receipts were not
yet present. This is a running experiment, not a completed benefit result.
Provisional final estimate19:30-19:45 UTC+08; check near that window,
then refine using completed calls. Do not restart on an observer timeout.

## Completed Result, 2026-10-31 19:34 UTC+08

All48 trajectories finished; the controller log ends COMPLETE and its
three process handles were absent at19:34:26. The final analyzer verifies
metrics, source/GT/checkpoint/trace/bbox hashes, past-only calibration,
zero scale-only offset, clipping, calls, holds and exact control equality.
The frozen-state check is a worker witness, not independent network replay.
Report SHA256:
`df6a9be1f1ce6ec5dad1b81b980357b1b639f29be7c3fc3d5de0d8ca064b9cf7`.
Downloaded report, server report and completion receipt agree.

| Condition | Native/sham PR20 / SR (%) | Scale-only PR20 / SR (%) | Center+scale PR20 / SR (%) |
|---|---:|---:|---:|
| Complete | 75.146496 / 57.352125 | 75.146496 / 57.352125 | 75.146496 / 57.352125 |
| Block32 | 57.257201 / 44.689179 | 52.145743 / 39.656658 | 58.213389 / 44.977624 |
| Block128 | 57.049005 / 44.075706 | 57.443327 / 43.987086 | 57.891059 / 45.397069 |

Four-video paired bootstrap intervals are descriptive:

| Condition | Contrast | PR difference pp [95% interval] | SR difference pp [95% interval] | SR wins/losses |
|---|---|---:|---:|---:|
| Block32 | Scale-native | -5.111459 [-11.622615, 0.717014] | -5.032522 [-10.156655, 0.091612] | 1/3 |
| Block32 | Geometry-native | 0.956188 [-8.974263, 10.179439] | 0.288445 [-5.080820, 5.455225] | 3/1 |
| Block32 | Scale-geometry | -6.067646 [-15.669265, -0.529101] | -5.320967 [-14.410780, -0.334195] | 1/3 |
| Block128 | Scale-native | 0.394322 [0, 1.182965] | -0.088620 [-0.542623, 0.365383] | 2/2 |
| Block128 | Geometry-native | 0.842054 [0, 2.129338] | 1.321363 [0.345323, 2.330831] | 3/1 |
| Block128 | Scale-geometry | -0.447733 [-0.946372, 0] | -1.409983 [-2.312355, -0.507610] | 0/4 |

The scale-native SR effect changes by+4.943902pp from block32 to block128,
interval[-0.106565, 9.994368]. A negative effect becoming approximately
neutral does not establish a long-gap gain. Block128 covers TIR-only
missing exposure on this cohort, with no RGB-only or blind exposure.

Reject unconditional removal of center translation. The earlier same-history
instantaneous component diagnostic did not predict independent closed-loop
benefit. The geometry gain on these short TIR exposures does not resolve
its prior regression on the longer cohort. No gate or parameter was fitted;
this is neither official benchmark attainment nor a validated paper module.
