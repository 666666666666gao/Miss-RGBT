# State-Template History Control

## Reason And Intervention

The completed full97 development comparison finds an E0 SR penalty
of2.937169points on current11 frames after outage histories. This identifies
a history problem, not a proven template cause. Older E3/E4 official U0/U2
controls show that disabling updates or allowing only complete writes can
harm Miss tracking. Neither is promoted as the solution.

Test an ordinary state-conditioned template cache on fresh sequential
development indices68..75. Three arms use the same frozen E0 checkpoint:
native, recording-only sham, and state-matched reading. All retain the
original first template. Sham tracks the latest dynamic template as native
does while also recording state caches. The matched arm reads the most
recent dynamic template written under the current availability11/01/10.

All three caches initially contain the actual complete initial template
with truthful availability11 and sourcecall0. Before a state has a native
qualifying write, its cache therefore remains the initial template. No
missing sensor is fabricated. Writes retain the native modulo interval
and strict confidence predicate; current reads precede current writes.
Both-missing video frames hold box and caches without advancing calls.
Each observed frame makes one network call. No fitted threshold, extra
forward, optimizer, new network or frozen-runtime edit is introduced.

This can change template age as well as state compatibility. The experiment
cannot attribute an effect solely to state matching or establish novelty.
Crop and future template histories evolve jointly in each closed loop.

## Prospective Cohort

| Development index | Video | Frames |
|---|---|---:|
| 68 | lefthandfoamboard | 154 |
| 69 | rightrunninglatterone | 639 |
| 70 | 1phoneblue | 373 |
| 71 | leftgirlat1row | 525 |
| 72 | boyridesbesidesgirl | 302 |
| 73 | girlshead | 6024 |
| 74 | whitecarleave | 127 |
| 75 | whiteTboy | 254 |

Total8398video frames, no outcome/duration filtering. Greedy assignment gives
GPU0 the6024-frame video and GPU1 the remaining2374frames. Do not omit the
long video or report only the faster worker. Conditions are complete and
the existing32-frame alternating11/01/11/10/11/00schedule,48trajectories.

## Execution And Verification

New isolated files:
`state_template_tracker28.py`, `run_state_template28.py`,
`launch_state_template28.py`. Reuse the existing server environment and
the frozen runtime. GPU0/1 each receive one evaluator beside their trainer
only after at least3000MiB free per card is observed.

Before launch, run the actual CPU track-body witness for21calls/condition.
It requires exact native/sham boxes and confidence, complete-input native/
matched boxes, actual qualifying writes, differing selected sources under
switches, and an unchanged frozen-network digest. This limited witness
does not certify CUDA initialization or long-trajectory efficacy.

Workers record hashes for source, checkpoint, registration, GT, boxes and
traces. They require full-trajectory native/sham equality and complete-input
native/matched equality. The final analyzer rereads every trajectory,
recomputes PR20/SR, reconstructs each bank's past-only source and actual
availability, verifies native write predicates, call counts and blind holds.
Frozen-network identity is a worker digest witness, not independent replay.
The final paired-video bootstrap resamples8videos with10000draws/seed2027;
it does not measure training-seed uncertainty.

Output `reports/server28-state-template8-20261031`, sibling `.log`.
Initial estimate45-90minutes after workers start plus up to300seconds
controller observation delay, driven by the longest video's33100approximate
observed calls across all arms/conditions. Refine from actual receipts.
Poll near expected completion,180-300seconds apart. Keep all registered
sources unchanged until final analysis. There are no trained weights to
delete in this experiment. A completed favorable result would still need
independent replication and stronger mechanism controls before adoption.

## Actual Launch

Final Python3.7 compilation and CPU track-body witness passed. The witness
records14actual writes and15calls with differing selected template sources,
with exact sham boxes/confidence, complete-input box equality and unchanged
frozen-network digest. Earlier witness before count-audit additions remains
archived on the server; the following hash refers to the final source.

Witness SHA256:
`17e16924e0347ba507c7e762fa5d024bb4f22c8fdac4238798925d6ea506e1fd`.
Tracker SHA256:
`8f3c0b48ee40d74aca4925f1aeca23b867d38b82fb3d91631b013d8244276a4f`.
Runner SHA256:
`a4bbf19900df0aacf4992a1168454da64d033b61fe905305cef253835cfb2bc3`.
Registration SHA256:
`7b86360a23527e90e68ec73f2828cbebed6be54ca5f3f71364526a750a913f6e`.

Controller342664 launched20:07:20 UTC+08. It finished all48 trajectories and
the final analyzer; controller COMPLETE was observed at20:56 UTC+08.
Completion receipt SHA256
`8ffe5b57d56c3be01a6d7f51f5854c39290dcedf2377d1afa8eb868ef6fa32be`
names report SHA256
`66b3b776e0152f01bf0115e322e371ca7e15ec0c7b1576e1f8550f11300ad3e9`;
downloaded report hash matches. Analyzer verified all trajectory, control,
GT/trace/box hashes and past-only bank/write history.

| Condition | Arm | PR20 (%) | SR (%) |
|---|---|---:|---:|
| Complete | Native/sham | 92.123743 | 74.997601 |
| Complete | State matched | 92.123743 | 74.997601 |
| Block32 outage | Native/sham | 80.072518 | 65.698845 |
| Block32 outage | State matched | 73.127902 | 59.989876 |

Matched-minus-native paired video differences:

| Metric | Mean difference pp | 95% paired-video interval | Wins / ties / losses |
|---|---:|---:|---:|
| Block32 PR20 | -6.944616 | [-18.359874, -0.145757] | 1 / 2 / 5 |
| Block32 SR | -5.708969 | [-14.650757, -0.689463] | 1 / 1 / 6 |
| Outage current11 PR20 | -6.864088 | [-17.031998, -0.082919] | 0 / 4 / 4 |
| Outage current11 SR | -5.698328 | [-14.003148, -0.159444] | 1 / 1 / 6 |

Bootstrap intervals resample8videos for one E0, not training seeds.
Current11 includes pre-outage frames and full-complete blocks; it is not
exclusively first restoration frames. `leftgirlat1row` SR drops35.796074pp.
That matched arm records34 block32 writes versus64 native; on `girlshead`,
527 versus560. The confidence-controlled predicate is unchanged, so
trajectory confidence also changes subsequent template-write frequency.
This coupled loop does not isolate cache-age or availability causality.

Decision: reject state-matched cache reading. Complete tracking is unchanged,
but missing-history performance degrades materially. Do not adopt it as a
paper module or tune a state gate on these same videos. Seek an independent
prospective alternative for temporal recovery.

Full report: `reports/state-template-report-20261031.json`.
