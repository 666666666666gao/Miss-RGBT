# Past-Frame State Offset Control

## Independent Factorial Replication

The fresh16-video fixed-crop probe completed576 model calls in338.26seconds.
Completion and both analyses exist. Report SHA256:
`cf6046d5edd800a5694da9675b5d062088b4ea20586d096ff11426fcbc894c99`.

Equal-lag1 velocity errors, pixels/frame (current state by prior state):

| Current / Prior | 11 | 01 | 10 |
|---|---:|---:|---:|
| 11 | 2.001192 | 5.766073 | 3.442066 |
| 01 | 9.011773 | 5.546387 | 10.030667 |
| 10 | 2.828380 | 6.245085 | 2.724069 |

Current01 same01-minus-prior11 velocity error is-3.465386,
paired-video95%CI[-5.944977,-1.744047],15/16 videos improve.
Current10 same10-minus-prior11 is-0.104311,
CI[-0.717432,0.481283],9 improve and7 worsen: this contrast does not
replicate a stable velocity advantage. Five of six same-state velocity
contrasts have descriptive intervals below zero. These are early fixed-crop
development observations, not whole-tracker gains or seed uncertainty.

## Fresh Closed-Loop Lag2 Counterexample

The registered fresh motion comparison completed and verified64 tracks per
history, equal complete/sham controls and common cohort/GPU/checkpoint.
Availability-history equals ordinary lag2 for this schedule. Relative to
adjacent velocity, fresh lag2 PR/SR changes are-2.712765/-1.301565pp,
95% video intervals[-10.293802,1.598041]/[-5.519468,1.159146].
SR improves10,ties1,worsens5; `fallenbikeitself` alone loses31.014006pp SR
and56.470588pp PR. This counters a uniform lag2-generalization claim.
Do not discard the video, tune a cap from this result or promote the
earlier positive16-video mean as a validated module.

## Past-Only Offset Counterfactual

Use the saved states at consecutive frames to estimate
`offset_s(t-1) = center_s(t-1) - center_11(t-1)`.
Correct the current missing-state center with
`center_s(t) - offset_s(t-1)`. Current full predictions enter only an
offline offset-drift diagnostic, never correction or forecasting.
No new model calls or fitted coefficients are used in this saved-data audit.

| Current state / Cohort | Raw current center error | Corrected | Delta |
|---|---:|---:|---:|
| 01 / Original16 | 6.802768 | 4.403591 | -2.399177 |
| 01 / Fresh16 | 9.518039 | 6.313314 | -3.204725 |
| 10 / Original16 | 4.702243 | 3.691054 | -1.011189 |
| 10 / Fresh16 | 3.530234 | 3.433573 | -0.096662 |

Values are equal-video mean pixels, not PR/SR. Fresh01 delta CI is
[-5.730900,-1.380145]; fresh10[-0.702118,0.461060] crosses zero.
Fresh01 forecast error also drops6.874234px relative to raw cross-state
velocity and2.696090px relative to same-state velocity.
Fresh10 forecast comparisons remain uncertain; versus same-state motion
the mean changes+0.239415px, CI[-0.326405,0.882329].

The correction was devised after inspecting both factorials. Neither cohort
is a prospective test of this new correction. No closed-loop, official or
novel-module claim follows. Audit:
`reports/causal-state-offset-audit-20261031.json`.

## Prospective Closed-Loop Control

New sequential development indices36..39 were registered before collecting
their offset-control outcomes:

| Video | Frames | GPU |
|---|---:|---:|
| 10phone_boy | 435 | 1 |
| mototaking2boys | 111 | 1 |
| leftredflag-lsz | 1334 | 0 |
| bikefromnorth | 275 | 1 |

The2155 frames are disjoint from705 pilot training,97 heldout and preceding
36 development videos. Freshness is relative to these diagnostic comparisons;
E0 pretraining exposure is not independently certified.

Three arms use identical E0 weights, physical GPU and complete/block1 outage
conditions,24 full trajectories total:

- Native tracker.
- Calibration sham: calculate the same counterfactual offsets but apply none.
- Offset correction: apply cached missing-state center offsets before native
  clipping and dynamic-template extraction.

On a complete frame, evaluate11 and simulated01/10 on the same previous
tracker state and template history; store clipped center differences.
Counterfactual sensor images are constructed only when both inputs are
available. On a missing frame, infer only from the observed sensor and
subtract its past offset. Zero offsets are the initial prior until the
first complete observation. Both-missing frames hold state and perform no
model call. This experiment adds no blind velocity, coefficient fitting,
threshold, confidence gate or neural weight update.

The calibration policy is fixed for both missing states despite the
inconsistent RGB-only diagnostic; do not select per-video rules afterward.
The untrusted/stale calibration risk and extra forward-call cost remain
empirical questions. Raw confidence and native template schedule are retained.
Three model calls per complete frame are required in both sham and correction;
native uses one. Per-condition model-call counts are recorded.

## Verification and Execution

Seven-frame CPU entry witness on the first registered video checks actual
new track-body computations, shared frozen E0 weights, bitwise native/sham
equality in both conditions, and bitwise complete/corrected equality.
It uses CPU-initialized attributes and does not verify the GPU constructor,
CUDA initialization or full trajectories. Report
`reports/state-offset-entry-20261031.json` records the limited scope.

The new standalone GPU evaluator leaves the frozen runtime and active
training/evaluation sources untouched. Full-run controls must verify
bitwise native/sham equality and complete-condition equality on every video.
Independent analysis recomputes trajectory metrics and checks that every
applied offset comes from the last complete calibration, including initial
zero offsets, and that coordinate correction/clipping matches saved boxes.
It also verifies model-call counts, blind holds, source/checkpoint/input
receipts and unchanged model state.

Controller1173352 started14:33:00 UTC+08 after checking the completed
preceding paired study. Workers1173436/1173438 were verified live14:34:55,
each2266MiB beside the original19290MiB trainers. GPU2/3 training continues.
GPU assignments contain1334/821 frames. Controller polls300seconds.
Estimate20-40minutes plus final checks, inspect near14:55-15:15.

Output `reports/server28-state-offset-control4-20261031`, log sibling `.log`.
`registration.json` is immutable; worker specs reference its hash.
Do not modify the queued four source files before independent analysis.
Original targets and three validated paper modules remain unachieved.

## Completed Four-Video Result And Independent Replication

The24 trajectories and independent analyzer completed by14:53 UTC+08.
Downloaded report hash matches the server and completion receipt:
`c522e6a898ebfc2f88a145bc41c9c317afc1c380efbddf397f1c36f35dccf32f`.
Archived original four sources match every registered source hash.
Both downloaded worker reports match their final report receipts.

| Outage arm | PR20 | SR |
|---|---:|---:|
| Native | 76.109299 | 60.194491 |
| Calibration sham | 76.109299 | 60.194491 |
| Corrected | 76.889868 | 60.844880 |

Paired PR/SR changes are+0.780568/+0.650389pp.
Video bootstrap95% intervals[-1.951681,4.105839]/
[-1.360381,3.278505] both cross zero. SR wins2/loses2:
10phone_boy-.076805, mototaking2boys-1.861472,
leftredflag-lsz+.142893, bikefromnorth+4.396941.
Complete PR/SR75.596413/62.662731 is bitwise identical across all arms.
All native/sham trajectories are bitwise identical, all offsets have verified
past-only provenance, and the analyzer checks clipping, holds, call counts,
input/checkpoint/source hashes and unchanged network state.
This small heterogeneous result is insufficient to claim stable efficacy.

Without changing the tracker or correction parameters, registered fresh
development indices40..55 for16 videos13951 frames,96 trajectories.
The same three arms and two conditions run on physical GPU0/1 beside their
trainers, loads6968/6983 frames. The6158-frame leftconergirl is retained;
cohort selection is sequential and independent of outcomes.
Same video and physical GPU are used in every arm.
Original sources were archived before adding cohort/output arguments to the
controller/analyzer; evaluator changed only its descriptive scope string.
The offset-tracker hash remains078973206eb4b070cbd2cbfce14118018b5748003fdb3826f745d23aac8d94c3.
The revised entrypoints compile under the actual server Python.

Controller1212195 launched14:55:51, workers1212202/1212203 were verified
running14:56:10 with actual GPU memory allocation. Four training workers
3453478/3453479/3816056/3816057 remain present in GPU process accounting.
Immutable registration hash:
`68d5a2fad5fb69663ddc03185a1061e7edd3f6aac3aecda412d795deab29f1f2`.
Output `reports/server28-state-offset-replication16-20261031`,
log sibling `.log`. Expected completion16:30-17:00 UTC+08 based on the
completed four-video runtime; controller polls300seconds.
Preserve the new four source hashes until its analyzer finishes.
No replication outcomes or official benchmark improvements are established.

The pinned DSRTrack review in `DSRTRACK_SOURCE_REVIEW_20261031.md`
documents existing anchor-guided template repair, volatility-based queue
selection, identity/licensing limitations and a coordinate-contract concern.
It provides bounded source evidence, not a novelty certificate or adopted
upstream code.
