# val97 CPU prefix diagnostic

Date: 2026-10-30.

## Scope

Historical A utility epoch20, validation sequence `boywalkinginsnow3`,
first 257 video frames. The sequence is the first entry in the fixed val97
list; the prefix and four conditions were fixed before execution.
Each condition has 256 noninitial frames.

This is a real neural tracking and image-decoding check on CPU with four
threads and CUDA disabled. It does not establish full-val97 performance,
official complete/Miss benchmark scores, numerical equivalence with GPU,
recovery efficacy, novelty, or a module improvement.

The formal GPU0/1 training and its frozen runtime remain unchanged.

## Execution and Checks

Native tracker and preprocessor require CUDA. The separate
`/home/gaob/Miss-RGBT/cpu_diagnostic_tracker.py` explicitly ports them to CPU,
preserving crop construction, six-channel normalization, Hann window,
box decoding, clipping, availability, and template update rules.
It additionally records the raw score maximum. This is an isolated diagnostic
implementation with no recovery intervention.

Runner: `/home/gaob/Miss-RGBT/run_cpu_recovery_prefix28.py`.
Server records:
`/home/gaob/Miss-RGBT/reports/server28-val97-cpu-prefix-20261030`.

Checks cover:

- CPU parameter placement, source/checkpoint/protocol/GT/mask hashes.
- Finite predicted boxes and recorded confidences.
- Identical first 31 noninitial predictions across conditions, before outages.
- Both-missing holds without tracker calls or state changes.
- Actual template replacement only at the native call interval and confidence.
- Exactly 32 both-missing holds and 224 tracker calls for alternating outages.

GT after initialization is used only for offline labels. Both-missing frames
have null crop-support labels because no actual crop is processed.
Hann and raw scores are uncalibrated responses.

## Completed Prefix Results

All checks passed. Completion receipt status:
`REAL_CPU_PREFIX_PROTOCOL_CHECK_COMPLETE_NOT_BENCHMARK`.
Elapsed time: 575.93 seconds.

| Condition | Mean IoU | Tracker calls | Both-missing holds | Template writes | Single-modality writes |
|---|---:|---:|---:|---:|---:|
| complete | 0.796168 | 256 | 0 | 39 | 0 |
| RGB outage | 0.799209 | 256 | 0 | 38 | 22 |
| TIR outage | 0.766971 | 256 | 0 | 32 | 14 |
| alternating outage | 0.716680 | 224 | 32 | 27 | 10 |

Mean IoU includes held predictions during both-missing frames. None of the
observed frames had a target center outside the previous search crop or zero
target/crop intersection. This prefix does not reproduce target escape from
the crop and therefore provides no positive evidence for wider search.

At alternating-condition frame192, after the 32-frame both-missing block,
the tracker call counter was160 and IoU was0.883237. The native update
schedule follows tracker calls, which differs from video frame number when
both modalities are missing. This follows the original evaluator policy.

The single-modality conditions each restored complete observations four
times. The last restoration is at frame256; only one frame remains in this
prefix. The saved restoration-window summaries explicitly record this
shortened window.

Single-modality template writes actually occur, but this observation alone
does not establish template contamination. No intervention, threshold tuning,
or template-write rule change was made.

Local evidence:

- `server28-val97-cpu-prefix-20261030/report.json`
- `server28-val97-cpu-prefix-20261030/completed.json`
- Four trace JSON files in the same directory, verified against report hashes.
- `server28-val97-cpu-prefix-observations.json`

GPU equivalence and full-val97 neural evaluation remain unverified. Do not
add the development diagnostic to the current official evaluation queue.

## Training Status

At 2026-10-30T13:13:37Z:

| Run | GPU | Updates | Logged key losses | Recent seconds/update |
|---|---:|---:|---|---:|
| A utility | 0 | 1990/50000 | finite | 2.345 |
| A random | 1 | 2144/50000 | finite | 2.209 |

Both are existing seed2027 full20 configurations. The independent evaluator
is waiting for successful completion of both training runs and checks source
hashes before complete/Miss evaluation. Its wait interval is 300 seconds.
The training remains unfinished; no new official metric is available.
