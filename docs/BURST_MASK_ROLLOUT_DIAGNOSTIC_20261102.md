# Burst-Mask Rollout Diagnostic

## Question and evidence

`miss_rgbt.train.sample_masks` draws each of the three causal search-frame
availability states independently with probabilities `0.4/0.3/0.3` for
complete/RGB-only/TIR-only. The official tracker instead carries availability
with dynamic templates and updates its predicted box/template state over
time. The training call does not roll its predicted boxes or template updates
through the three search frames.

This diagnostic asks whether temporal clustering of missing states changes
frozen-E0 closed-loop tracking on training videos when each video has exactly
the same number of complete, RGB-only, and TIR-only frames in both arms. It
does not test a new method or establish benchmark gains.

## Registered protocol

- Data: 24 fresh contiguous 64-frame clips from LasHeR `train882` fitting
  names only. Clip selection is seeded and excludes source frames already
  referenced by the previous registered train-only pilots. No val97,
  RGBT234, LasHeR test, or official test masks are read.
- Model: frozen CADTrack E0 availability checkpoint, unchanged native
  `MissingTracker`, box update, confidence threshold, and template update.
- Paired schedules: for each clip, draw the same exact categorical state
  counts from probabilities `[0.4, 0.3, 0.3]` for both arms, with frame 0
  complete. The IID arm shuffles those labels; the clustered arm places each
  state in one contiguous block with seeded random block order. This
  deliberately contrasts temporal autocorrelation while holding per-clip
  state counts fixed. It is a stress-test schedule, not a claim about the
  official masks' run-length distribution.
- Readouts: per-sequence mean IoU over single-modality frames, separately for
  RGB-only and TIR-only, and over the first five complete frames after an
  outage-to-complete transition. Report paired clustered-minus-IID effects
  with 10,000 sequence-bootstrap draws. Also retain frame-level boxes,
  confidence, state, and GT for arithmetic verification.
- State controls: model parameter digest before/after must match, gradients
  remain absent, all predictions/IoUs finite, and all frames in both arms
  must have the same per-state counts for each clip.
- Advancement gate: consider a matched burst-aware training pilot only if
  clustered schedules reduce sequence-macro missing-frame IoU by at least
  `0.01` with the 95% interval entirely below zero, or reduce recovery-frame
  IoU by the same amount with the interval entirely below zero. A pass only
  justifies a separate matched train-only training experiment; it does not
  validate a paper module or benchmark result. Otherwise close this direction.

## Scope and limitations

This is a training-split closed-loop diagnostic of frozen E0, not a
development benchmark. The two schedules place the same state counts on
different temporal positions; scene motion and state position can therefore
interact. Sequence-clustered paired intervals quantify between-sequence
uncertainty, but this single seeded schedule construction is not a robustness
study. It does not evaluate both-sensors-missing hold frames, test-set
performance, training convergence, or any of the requested three modules.

Use GPU2 only after checking occupancy, temperature below 85 C, and the
existing 250 W power cap. Estimated runtime is 5-15 minutes for 3,024
tracker frames plus image loading. Inspect near that window, using 180-300
second waits if it is still running. Preserve all existing weights and
reports.

## Completed result

GPU2 PID144335 completed normally in123.645seconds on 2026-11-02. The
diagnostic used24 clips, each64 contiguous frames, from24 fitting sequences.
Each paired schedule tracks63 frames after the first-frame initialization;
the report contains3,024 tracked-frame rows. Every clip had the same counts
in both schedules:26 complete,19 RGB-missing/TIR-present, and19
RGB-present/TIR-missing frames. Input image hashes and exact frame IDs are
retained.

Effects below are clustered schedule minus IID schedule in mean IoU, with
10,000 sequence-bootstrap draws. Negative values favor IID; every interval
crosses zero.

| Readout | Mean difference | 95% interval | Paired sequences |
|---|---:|---:|---:|
| All single-modality frames | -0.015331 | [-0.032356, 0.001973] | 24 |
| RGB-only input | -0.009342 | [-0.037280, 0.018596] | 24 |
| TIR-only input | -0.013135 | [-0.052738, 0.026904] | 24 |
| First five complete frames after outage | -0.043505 | [-0.178025, 0.056210] | 18 |

The registered advancement gate is false. The point estimates suggest
clustered outages may be harder, particularly on recovery, but sequence
uncertainty is too large to establish that effect. Do not launch
burst-aware full training from this result or count it as a paper module.
This schedule comparison also does not establish the actual test-mask
run-length distribution.

The model state digest remained
`c4cc43454f80d838db0130f8664b36e73063aadb99a3a645ee12bfee1fd5c20e`;
no gradients were created. Peak allocated/reserved GPU memory was
477,212,160/547,356,672 bytes. Independent analysis rechecked source and
checkpoint hashes, every frame's schedule/recovery label and box IoU, and
recomputed all sequence-bootstrap intervals. It passed.

The producer's first summary encoded the paired-sequence count under a
`frames` key. The immutable raw report SHA256 is
`b44dc635c7b6c1807ab08dd6ec9e7d1e3976fce8d4e40ceacb8fd198b420212b`;
`analysis_recomputed.json` reports the corrected `paired_sequences` name
and independently recomputed values. Full frame records and receipts are in
`reports/server28_burst_mask_rollout_20261102/`.
