# Published Mask Structure And Motion Control Scope

The standalone audit reads the frozen E0 inventory and the two published
availability JSON files, verifying their existing inventory hashes. It does
not read images, ground truth, predictions or benchmark scores. Frame zero
is forced complete to match the actual evaluator initialization.

| Property | LasHeR-Miss | RGBT234-Miss |
|---|---:|---:|
| Videos / frames | 245 / 220703 | 234 / 116650 |
| Raw both-missing frames | 33743 | 18766 |
| Effective both-missing frames | 33737 | 18756 |
| Effective both-missing percentage | 15.286154 | 16.078868 |
| Videos containing both-missing frames | 147 | 140 |
| Contiguous both-missing intervals | 19048 | 10708 |
| Duration median / mean, frames | 1 / 1.771157 | 1 / 1.751588 |
| Duration p95 / maximum, frames | 4 / 15 | 4 / 12 |
| Recovery to RGB absent, TIR present | 8057 | 4663 |
| Recovery to RGB present, TIR absent | 8296 | 4556 |
| Recovery to both present | 2686 | 1480 |
| Intervals ending at video end | 9 | 9 |

Availability labels are RGB,TIR; one means observed. There are34/31 first
frames with a noncomplete published state, overridden by initialization.
The effective counts include initialized frames, matching the evaluator
protocol. Initial frames cannot be blind intervals in this audit.

## Consequence

Both-missing prediction is relevant to the requested official benchmarks.
However, the registered development schedule has32 consecutive blind
frames. That is longer than even the maximum official gap in either
published mask. The first four-video gain and pending next16 result
therefore describe a long-interruption stress test. They cannot establish
that fixed CV improves the official short-gap regime.

The next comparison must isolate shorter gaps before committing a new
motion module or claiming benchmark relevance. It must also examine
recovery with only one sensor: most published blind intervals return to
one observed sensor, rather than both sensors.

## Registered Short-Gap Follow-Up

Keep exactly the same16 full development videos, indices4..19, all7776
frames, frozen E0, unchanged hold/velocity rules and paired GPU assignment.
Use the same availability cycle11/01/11/10/11/00 with block size1 instead
of32. This gives isolated one-frame blind gaps. It is a controlled duration
comparison, not an exact reproduction of official joint state frequencies,
interval distributions or sensor-specific persistence.

Run both arms and complete/outage conditions,64 trajectories,31104frames.
Require the same complete/prefix equality, frozen-state and independent
trajectory verification, paired-video bootstrap and recovery group analysis.
Do not select videos, thresholds or velocity parameters based on the
pending long-gap outcomes or any official tracking results.

Do not edit the currently running evaluator/controller/analyzer sources:
their recorded hashes must remain verifiable through completion. After
their final analysis finishes, archive receipts and add an explicit
frame-block argument for a separate output directory. No short-gap GPU
run has been launched as of this registration.

## Evidence

Script: `audit_official_mask_structure28.py`.
Full report is6.2MB and remains in server reports and local outputs rather
than the publication repository. It includes every interval and transitions.

Server report: `reports/server28-official-mask-structure-20261031.json`.
Local report: `outputs/official-mask-structure-20261031.json`.
Report SHA256:
`4be7ac472affdd0c6ee023b3882556431cd74c4d7db7b0595f2f994d0d3d9c04`.

This is a schedule-structure finding, not a tracking score, training result,
parameter fit or independently validated paper contribution.

## Short-Gap Execution Prepared

At12:49 UTC+08 continuation PID997655 was launched and verified live,
waiting on actual long-gap controller967085 at300-second intervals.
The long-gap workers967093/967094 were still live at12:49:56.
The short-gap GPU run is therefore queued, not yet started.

`continue_blind_motion_shortgap28.py` first waits for that controller to exit,
then requires its64-trajectory completion and both independent worker
analyses. It archives the exact three long-gap source files, checks their
launch hashes, deploys the compiled staged short-gap files, and invokes
`launch_blind_motion_shortgap28.py`. Any failed assertion stops the chain.

The staged evaluator adds only an explicit1/32 frame-block choice. The
controller sends that argument to both workers and uses a separate
`reports/server28-blind-motion-shortgap16-20261031` directory for block1.
The files in GitHub describe this staged next execution; the in-flight
server-root files still describe the long-gap execution until it completes.
Staging directory: `development/shortgap_staged_20261031`.
Continuation log: `reports/server28-motion-shortgap-continuation-20261031.log`.

`analyze_motion_gap_contrast28.py` is deployed and compiled for use after
both runs finish. It verifies their completion/report/worker-analysis
receipts, identical cohort/GPU/checkpoint assignments and bitwise identical
complete trajectories across durations. It then compares paired per-video
CV-minus-hold effects with10000 video bootstrap draws, seed2027.
Both duration and switching frequency change, so this comparison cannot
identify duration alone as a causal mechanism. No cross-duration result
has been produced yet.
