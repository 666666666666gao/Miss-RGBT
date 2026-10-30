# Search coverage and candidate selection audit

Date: 2026-10-30. Historical A utility epoch20 on SSH2028.
CPU-only diagnostic. Four live training jobs and frozen runtime preserved.

## Executed Proposal Experiment

Fixed first8 val97 entries, alternating-outage trajectory, every37th frame:
take the first2 observed frames per availability state per sequence without
GT/score selection. This gives35 observations and315 proposals. CPU4threads
completed in207.81s. The sampling favors early sequence frames and does not
represent the full distribution of later severe drift.

Each observation uses its prerecorded previous box, the fixed initialization
templates and9 native-size search crops. Crop centers are the original plus
8 offsets on a3x3 grid at half native crop side. Spatial resolution, target
scale and inverse crop mapping remain native. No current GT enters proposals.
Dynamic templates are intentionally excluded; this is not an intervention
that reproduces native dynamic-template outputs.

For every observation, center batch prediction was compared with an actual
single-crop CPU tracker call. Maximum bbox difference was0.00003052pixels;
all35 allclose checks passed. This checks this CPU mapping/implementation,
not equivalence to GPU trajectories.

## Proposal Selection

The independently encoded cosine and already selected pilot ranker were
evaluated on exactly the same proposals, without retuning either scorer.
The ranker checkpoint hash remains
11110ac9c04de0d234e1e23fef51aa087e57146a28d7d769dbdfe7453e117169.

| Rule | Mean IoU | IoU>=0.5 count, out of35 | Harm vs center >0.1 |
|---|---:|---:|---:|
| Original center crop | 0.708827 | 31 | reference |
| GT oracle best of9 | 0.723872 | 32 | oracle includes center |
| Hann top1 | 0.722725 | 31 | 0 |
| Raw response top1 | 0.688270 | 30 | 2 |
| Independent cosine top1 | 0.575025 | 25 | 7 |
| Fixed pilot ranker top1 | 0.605296 | 26 | 6 |

GT oracle is diagnostic candidate quality, not an inference rule. The
apparent mean-IoU increase is dominated by one `ab_pingpongball` observation;
neither paired test performance nor a recovery improvement follows.
Only2 sampled center crops lost all GT overlap, and none of their9 crops
restored support. All selection rules had zero IoU for those2 observations.
Cosine/ranker selection harms otherwise useful candidates, strengthening
the decision not to promote either into the tracker.

## Full-Cohort Geometry

A separate analytical audit covers all3924 observed valid frames of the
fixed8 GPU alternating trajectories. It applies the same9-crop geometry
at each recorded previous box, with GT used only for retrospective labels.

- Native crop has no positive-area GT overlap on1505 frames.
- The9-crop union restores any GT overlap on150/1505 (9.97%).
- It contains the GT center on116/1505 (7.71%).
- Median required center coverage is6.132 times the native crop half-width.
- At half-side grid spacing, median minimum grid radius is6:13x13=169 crops.
- Maximum required expansion is9.793; radius9 gives19x19=361 crops.

These are oracle geometric cost estimates, not an adaptive search rule.
The radius formula ignores subpixel rounding; the9-crop support test uses
native rounded crop bounds. Background padding cannot add true image evidence.
The long `leftdrillmasterstanding` sequence supplies1378/1505 support-loss
frames and dominates these counts. This cohort is not a benchmark sample.

## Decision

The current evidence does not support deploying naive9-crop expansion or
the pilot identity scorer as the proposed B module. More coverage is
insufficient without target-conditioned localization and reliable selection;
local expansion also becomes expensive after large drift. The next useful
comparison should separate early failure detection from coarse global
proposals and acceptance, using train882 calibration and independent
validation. This is a direction, not a new module or novelty claim.

Formal20epoch and adaptive batch10 campaigns continue. Target metrics and
three validated modules remain incomplete.

## Evidence

- reports/server28-wider-candidates-20261030 (35frames/315proposals)
- reports/server28-wider-proposal-ranking-20261030 (fixed scorer outputs)
- reports/server28-search-coverage-20261030.json (all-frame geometry)
- outputs/wider-candidates-analysis-20261030.json
- outputs/wider-proposal-ranking-analysis-20261030.json

All raw traces, labels, candidates and scorer outputs are retained locally
and on server28. Source/checkpoint/protocol/trace hashes are in receipts.
