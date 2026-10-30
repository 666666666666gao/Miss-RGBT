# Fixed CPU recovery diagnostic cohort

Started: 2026-10-30T13:24:34Z.
Server: SSH2028. PID3608927. CPU4 threads, CUDA disabled.
Completed: 2026-10-30T14:39:41Z, elapsed4507.02 seconds (75.12 minutes).
No automatic restart or GPU use.

## Frozen Selection

First8 entries from the existing val97 protocol, all frames, historical
A utility epoch20. Complete and alternating-outage conditions only.
The alternating condition follows the existing32-frame blocks:
11,01,11,10,11,00. Both-missing frames retain the last box without a
tracker call. GT after initialization is for offline diagnostics only.

| Sequence | Video frames | Invalid-size GT |
|---|---:|---:|
| boywalkinginsnow3 | 565 | 0 |
| leftdrillmasterstanding | 2192 | 0 |
| leftgirlunderthelamp | 258 | 0 |
| girlridesbike | 175 | 0 |
| midboyplayingphone | 329 | 1 |
| boywithumbrella | 909 | 0 |
| manrun | 61 | 0 |
| ab_pingpongball | 113 | 25 |
| Total per condition | 4602 | 26 |

Short sequences may not contain a full both-missing block; expected call
counts are computed from their actual fixed masks. Invalid-size GT retains
predictions but has no geometry/IoU label.

## Evidence and Decision

Source:
`/home/gaob/Miss-RGBT/run_cpu_recovery_cohort28.py`.
Source SHA256:
`b9b9282ebf47008759b7576418b8e412b4aafa695f96d5efd3d7e0923cb19835`.

Receipts/traces:
`/home/gaob/Miss-RGBT/reports/server28-val97-cpu-cohort8-20261030`.
Log:
`/home/gaob/Miss-RGBT/reports/server28-val97-cpu-cohort8-20261030.log`.

Launch records checkpoint/protocol/source hashes before neural execution.
The runner checks finite outputs, identical initial31 predictions across
conditions, both-missing holds, native template replacement rules and exact
tracker-call counts. Each sequence saves paired support-loss frame indices
where the complete trajectory crop intersects GT and the outage trajectory
crop does not. Each trace has a recorded SHA256.

At2026-10-30T14:39:41Z PID3608927 completed normally. All eight fixed
sequences have paired traces:

| Sequence | Complete mean IoU | Complete zero-overlap crops | Alternating mean IoU | Both-missing holds | Observed paired frames: outage-only crop miss | Complete-only crop miss | Observed paired IoU delta |
|---|---:|---:|---:|---:|---:|---:|---:|
| boywalkinginsnow3 | 0.831257 | 0/564 | 0.741936 | 85 | 0/479 | 0 | -0.005962 |
| leftdrillmasterstanding | 0.344989 | 717/2191 | 0.128999 | 352 | 770/1839 | 0 | -0.222428 |
| leftgirlunderthelamp | 0.721527 | 0/257 | 0.472391 | 32 | 66/225 | 0 | -0.175554 |
| girlridesbike | 0.848877 | 0/174 | 0.742930 | 15 | 0/159 | 0 | -0.060538 |
| midboyplayingphone | 0.781717 | 0/327 | 0.773347 | 32 | 0/295 | 0 | +0.033628 |
| boywithumbrella | 0.847099 | 0/908 | 0.754629 | 128 | 0/780 | 0 | -0.018524 |
| manrun | 0.917806 | 0/60 | 0.889110 | 0 | 0/60 | 0 | -0.028696 |
| ab_pingpongball | 0.071520 | 67/87 | 0.070928 | 0 | 0/87 | 6 | -0.000592 |

For `leftdrillmasterstanding`, outage crops also missed GT on609 paired
observed frames where the complete crop missed too. There were no paired
frames where the complete crop missed while the outage crop still overlapped
GT. Outage-only support-loss runs included frames661-735 (75 frames),
960-1119 (160),1152-1311 (160),1344-1503 (160), and1536-1662 (127),
among shorter intervals. These are development trajectory diagnostics,
not proof of identity recovery or causal attribution to a particular
modality. The alternating mask is fixed in32-frame blocks; interpret run
locations against that protocol. This sequence is already a poor
complete-input trajectory, so the missing-versus-complete paired comparison
is essential.

`leftgirlunderthelamp`'s66 outage-only support-loss frames form one run at
frames192-257. Its paired-IoU median difference is only-0.000234 despite
mean-0.175554, indicating the mean is driven by a subset of divergent
frames. The complete trajectory has no crop-support loss in this sequence.

Across all eight sequences, 3924 observed valid frame pairs contain836
outage-only lost-support frames,670 frames where both crops miss, and6
complete-only lost-support frames. The paired crop-support failure rates
are38.38% outage versus17.23% complete. The frame-weighted paired observed
IoU delta is-0.119095; it is descriptive and dominated by the long
`leftdrillmasterstanding` sequence. Only two of eight sequences have any
outage-only support-loss frames; six have none. The only complete-only
support-loss frames occur in `ab_pingpongball`, where both trajectories
perform very poorly. Therefore do not assume a universal failure or
generalize this fixed cohort's aggregate to official Miss metrics.

The existing response peaks provide a possible but incomplete recovery cue.
Across observed valid frames, low Hann peak predicts zero GT/crop
intersection with AUC0.837 in alternating outage but only0.553 in complete
tracking. Raw peak AUC is0.847/0.559. The outage score signal is concentrated
in two sequences with positive support-loss frames; six sequences have no
such outage frames, so per-sequence AUC is undefined there. This is
retrospective geometry labeling; AUC is not a calibrated probability and
selects no threshold. Low response alone is not a demonstrated universal
trigger, especially given near-chance separation in complete trajectories.
Recovery gating should investigate independent target/template consistency
alongside response evidence.

The earlier single-prefix check showed no observed crop escape, so it does
not justify a wider-search intervention. This larger fixed cohort tests
whether crop-support loss occurs in development data. Geometry alone does
not establish original-target identity or template contamination. No model
threshold or intervention is being selected here.

This is development failure reproduction, not official metrics, full-val97,
GPU equivalence, a recovery improvement, or three validated modules.
Formal training/evaluation remains frozen and continues separately.

## Optimizer Audit

During source review an optimizer-state aliasing concern was checked with
a minimal AdamW example in the actual Torch1.12.1+cu113 environment.
All three saved fields (`step`, `exp_avg`, `exp_avg_sq`) had distinct storage
after `load_state_dict`, and all saved values remained unchanged after an
optimizer step. The suspected flaw did not reproduce; no defensive copy,
probe rewrite or training change was made. This checks that API behavior,
not performance or complete-task protection.
