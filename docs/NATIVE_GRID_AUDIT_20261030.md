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

Completed normally after1314.66seconds (21.91minutes),46observations,
2346candidates. Worker4006613 was absent at17:37Z; final log reports COMPLETE,
and completed.json/report.json/analysis.json exist. All native candidate
boxes matched the previous audit exactly (maximum bbox difference0pixels).

| Rule | Mean IoU,45 valid points | IoU>=0.5 | Drops>0.1 versus native |
|---|---:|---:|---:|
| Native | 0.564517 | 32 | 0 |
| GT-oracle across51 candidates | 0.647675 | 33 | 0 |
| Actual maximum Hann peak | 0.485004 | 28 | 5 |

Of12native unsupported points, the union restores overlap and center
coverage at9, but only1has any candidate reachingIoU>=0.5. That success is
in complete11; neither missing state01 nor10 has an oracle success among
their3and4unsupported points. Actual selection also succeeds at that same
single complete point. Of9points with center coverage,8still lack a
successful localization. A better ranker alone cannot produce IoU>=0.5
at those8points from the present candidate boxes.

Across all45points,5have an oracle success but actual selection fails.
Coverage, candidate localization and ranking all have distinct limitations;
these counts do not isolate their causes. Fixed-grid center offsets, scale,
context and the native center prior remain confounded. This is not proof
that localization can never recover outside the native crop. No grid
budget, selector or threshold was tuned from these results.

Do not integrate the51-crop peak selector as a recovery module: it harms
average localization in this fixed diagnostic and provides no successful
missing-state recovery at the selected unsupported points. Prior GTcenter
diagnostics provide an upper bound only; current GT cannot generate a
deployable proposal. Next recovery changes need better localized candidates
and preservation of native tracking before an acceptance-only experiment.

Raw evidence copied to outputs/server28-native-grid-20261030. Additional
selection_analysis.json separates coverage, candidate ceiling and actual
selection overall and per availability. Its analyzer is
analyze_native_grid_selection28.py, run after completion, no neural changes.
Raw report SHA256:
8c80fa876762e8a12e8258b0a59ae4f579a48907b5457f3b96e001a98d6540c2.

## Raw-Score Versus Hann Decode Control

The paired rerun completed46/46 observations and2346candidates in
1355.57seconds. Before comparing raw selection, the analyzer verified the
same observation order and crop boxes and checked every valid Hann-decoded
bbox against the prior report: maximum coordinate difference0pixels.
The raw rerun report SHA256 is
5e894005819fc257c86666c03b365cf9d68ba5047e8060a4c804a28eb694ce19.

| Decode and selector | Mean IoU | IoU>=0.5 | Drops>0.1 vs native |
|---|---:|---:|---:|
| Native Hann | 0.564517 | 32/45 | 0 |
| Candidate peak selected with Hann | 0.485004 | 28/45 | 5 |
| Candidate peak selected without Hann | 0.431677 | 23/45 | 11 |
| Hann GT-oracle ceiling | 0.647675 | 33/45 | 0 |
| Raw-score GT-oracle ceiling | 0.656347 | 33/45 | 0 |

Removing Hann changes candidate ranking but does not improve it: raw peak
selection is worse by0.053326 mean IoU versus Hann selection and0.132840
versus native. The raw oracle ceiling is only0.008672 above Hann oracle;
the unsupported-state raw oracle still succeeds at only1/12 points, no
better than Hann. Therefore the window is not the sole cause. The remaining
gap is shared by candidate coverage/localization and unreliable response
ranking; these one-step fixed points do not isolate a causal mechanism.
Do not remove Hann or use raw peak as a recovery selector from this audit.

The exact same score/size/offset outputs were decoded both ways. This
controlled diagnostic establishes that the observed failures are not
explained by Hann weighting alone, but it does not show that any particular
representation, regression head or candidate strategy is causal.
Full comparison is in
outputs/server28-native-grid-decode-20261030/selection_analysis.json.

### Launch History

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
valid frames in the existing fixed8 CPU outage cohort:

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
