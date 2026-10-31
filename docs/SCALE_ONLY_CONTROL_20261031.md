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
