# Fixed-Budget Native-Scale Proposal Audit

The GT-centered intervention suggests center and scale/resolution deserve
separate checks, but supplies no deployable recovery rule. This follow-up
generates all proposals without current-frame GT.

## Fixed Protocol

Reuse all46 observations from the earlier global audit. At each point:

- Keep the previous-box native crop.
- Place25 centers at the centers of a fixed5x5 image grid, using previous
  predicted box width/height.
- Use the same25 centers with initialization-frame box width/height.

Every crop uses native search factor and native output resolution. The
51-crop budget is fixed before execution and is not an image-coverage
guarantee. It is not tuned using GT, scores or an oracle radius. The first
frame's GT is ordinary tracker initialization; later GT only labels crop
support and predicted IoU after inference. One point has invalid-size GT
and remains in the raw predictions without geometric labels.

Use historical A utility epoch20, initialization templates only, current
modality availability, native Hann window and box head. Process crops in
CPU batches of8 with four threads. Compare the native candidate with the
previous audit's native prediction, tolerance rtol1e-4/atol0.05pixels.

Report separate coverage and center coverage for previously unsupported
points, GT-oracle IoU over each scale's candidates, and actual peak-response
selection. Oracle performance is only the finite candidate-set ceiling,
not a practical selection rule. Fixed trajectories, no learned acceptance
or trigger, no dynamic-template equivalence and no closed-loop/PR/SR gain.
Do not promote solely from oracle improvement.

## Execution

CPU worker4006613 launched on SSH2028 at approximately17:12Z;
Python3.7 syntax check passed. Verified live at17:13:15Z, CPU363%,
RSS about1GB, first observation completed29.35s after model initialization.
Estimated full neural audit20-25min, around17:35-17:38Z. Check near that
milestone with180-300s spacing; do not restart merely from elapsed time.
CUDA is disabled. All four GPU training workers remain allocated unchanged.
Source audit_native_grid28.py and launcher launch_native_grid28.py.
Evidence destination reports/server28-native-grid-20261030 with sibling log.
Launch receipt records script/checkpoint/protocol/source-report hashes.
Runtime estimate will use the first completed observation, then check near
the expected end with180-300s spacing.

Required completion evidence: report.json, analysis.json, completed.json
and a normal terminal process state. Results pending.

## Full-Trajectory Geometry

The exact same5x5 rules were evaluated geometrically over3924 observed
valid frames in the existing fixed8 GPU outage cohort:

| Scale rule | Restored overlap among1505 unsupported frames | Restored target center |
|---|---:|---:|
| Previous predicted size,25 proposals | 888 | 728 |
| Initialization size,25 proposals | 694 | 503 |
| Union of both,50 proposals | 909 | 744 |

The union restores overlap in60.4% of unsupported frames and target-center
coverage in49.4%; this is not proposal quality or recovery success.
The initialization scale adds only21 overlap frames and16 center frames
beyond previous scale in this cohort, at twice the grid proposal count.
The running51-crop neural audit remains unchanged; do not tune candidate
budgets from this val97 geometry. Results are dominated by the long
leftdrillmasterstanding sequence:1378/1505 unsupported frames, union restores
782 overlaps and617 centers there.

Geometry uses first-image dimensions per sequence, assuming dimensions
remain constant; the analysis has not decoded every image to verify that.
No neural or closed-loop result follows from geometric coverage.
Evidence reports/server28-native-grid-coverage-20261030.json and
analyze_native_grid_coverage28.py; local JSON copied to outputs/.
