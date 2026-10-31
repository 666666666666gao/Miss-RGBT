# Spatial Calibration Development Control

## Evidence And Limits

The fixed-crop geometry audit reuses two previously inspected16-video
development cohorts and performs no new neural calls. Each state/cohort
has64 early frame pairs. Same saved crop geometry and first-frame templates
are used for all availability states.
Report `reports/state-offset-geometry-20261031.json`, SHA256
`9f556ba28a3258aa412c7ec3eab324262dbe8b694b680d035eabfe703967345f`.
Server/download hashes match; fixed-crop and parent report receipts,
duplicate saved boxes and video GT hashes are checked.

Replacing only the current missing prediction's size with its current
complete prediction's size changes TIR-only SR by+5.059524pp on both
cohorts, with video intervals[1.190476,10.788690] and[1.785714,9.302455].
These complete predictions require the missing sensor and are diagnostic
only. They indicate a size component, not a deployable correction.

An additional saved-box calculation uses only the previous frame's
complete/missing predictions:

`offset_s = center_s(previous) - center_11(previous)`

`scale_s = size_11(previous) / size_s(previous)`

`corrected_center = current_center_s - offset_s`

`corrected_size = current_size_s * scale_s`

Compared with the same past-center-only correction:

| Cohort / State | SR delta, pp | Video95% interval | Wins / Ties / Losses |
|---|---:|---|---|
| Original16 / TIR-only01 | +6.324405 | [2.008929,12.276786] | 13 / 0 / 3 |
| Fresh16 / TIR-only01 | +7.589286 | [2.827381,13.467262] | 14 / 0 / 2 |
| Original16 / RGB-only10 | +1.413690 | [.520833,2.306548] | 8 / 7 / 1 |
| Fresh16 / RGB-only10 | +.148810 | [-.446429,.818452] | 4 / 8 / 4 |

Both cohorts were inspected before devising this scale extension: neither
is prospective for it. Saved synthetic boxes are not clipped and do not
change future crops/templates, so these gains cannot be promoted to
closed-loop or official results. RGB-only scale benefit does not replicate
consistently. GT evaluates boxes but never constructs them.

## New Closed-Loop Comparison

`spatial_calibration_tracker28.py` extends the existing center-offset
control. During a complete frame, the same previous tracker state and
template history produce11/01/10 predictions. Cache clipped-center
differences and positive width/height ratios from those predictions.
During a single-modality frame, use only its observed sensor and cached
past calibration. Apply scale about the raw predicted center, subtract the
past center offset, then use native clipping and template-update logic.
Before first calibration, center offset is zero and size ratios are one.
Both-missing frames hold state with no model call or template write.

Four arms share frozen E0, physical GPU assignment and all inputs:

- Native tracker.
- Calibration sham: compute all offsets/ratios and apply neither.
- Center only: same calibration calls, apply center offsets.
- Center and size: same calls, apply both corrections.

Each arm runs complete and block1 alternating-outage conditions.
The three calibration arms use3 forwards per complete frame and1 per
single-modality frame, so geometry-versus-center has equal model-call cost.
The new center arm must match the previous center implementation; all
complete-condition trajectories and native/sham trajectories must be
bitwise equal. Ratios apply to both missing states with no learned
coefficient, state-specific gate, cap, fitted threshold or weight update.
Ordinary online geometry calibration is a control, not a novelty claim.

The CPU witness uses seven frames of previously used development video36,
the explicit CPU E0 port and shared frozen weights. It compares actual new
track-body outputs with native and the existing center tracker.
This does not test the CUDA constructor or full trajectories.
Fresh registration uses sequential development indices56..59, excluding
all prior diagnostic cohorts, pilot train705 and heldout97. E0 pretraining
exposure is not independently certified.

`run_spatial_calibration28.py` separates witness, controller, worker and
analysis modes. The controller must wait for the actual running center-only
16-video parent1212195 and require its final receipt and verified controls
before allocating either GPU. It checks free memory>=3000MiB per GPU.
Only the two new source files are introduced; the active center16 sources
and frozen training runtime are unchanged.

The final analyzer independently recomputes every trajectory's metrics,
verifies registration/worker/input/checkpoint/source receipts and equality
controls, traces past-only center offsets and size ratios including identity
priors, reconstructs correction/clipping, counts calls and checks blind
holds. Geometry-minus-center is the primary new size contrast; geometry-
minus-native and center-minus-native remain reported in both conditions.
Uncertainty is paired-video bootstrap for this one frozen weight.

Output `reports/server28-spatial-calibration4-20261031`,
log sibling `.log`. Controller polls300seconds. Registration and actual
process/launch status are recorded in the handoff only after verification.
No official goal, convergence or validated paper module is established.

## Verified Registration

The CPU witness completed successfully. All four equality/frozen-state
checks passed. Missing-frame scales include identity1/1 before calibration,
RGB-only .981922/.991887 at frame3, and TIR-only .853643/1.047972 at frame7:
the new nonidentity path was actually executed. Witness report
`reports/spatial-calibration-entry-20261031.json` has SHA256
`9fc2f23a9c7405556b10babc31236cc0675ff2cc27db23df047593753a7f7987`.
Its source hashes match the local/server files.

Controller1256793 registered at15:21:59 UTC+08 and was observed alive
15:22:52, waiting for1212195. No geometry GPU worker has launched yet.
Registration hash:
`2127191883035a6c3721a4de4a55651c8c8d5e6a3ea8e178c825035a0fd93251`.

| New video | Frames | GPU after prior completion |
|---|---:|---:|
| rightbhindbike-gai | 265 | 0 |
| leftwhitebike | 199 | 1 |
| whitegirlwithumbrella | 131 | 0 |
| manbikecoming | 241 | 1 |

Total836frames,32trajectories, loads396/440. No duration/outcome filtering
was applied. Expected10-20minutes after both predecessor workers release
their resources, plus up to300seconds queue delay. Current16-video
predecessor remains live and is still expected16:30-17:00 UTC+08.
Do not modify either new source or the shared offset tracker while queued.
All four original training processes remain alive at15:22:52.
