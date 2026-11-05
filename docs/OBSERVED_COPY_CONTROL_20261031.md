# Observed-Feature Copy Control

The completed matched candidate epoch6 inference audit found small,
state-dependent compensation effects. Increasing the selected region count
did not consistently reduce loss. The earlier four-tuple privileged repair
diagnostic showed that exact complete features are not uniformly better.
Those results do not show whether ordinary copying of currently observed
features is a useful zero-parameter baseline.

## Registered Computation

`audit_observed_copy28.py` compares frozen packaged E0 against the same E0
with an intervention immediately after backbone block6. At positions where
exactly one modality is unavailable, copy the other stream's corresponding
feature. Other positions remain unchanged at that layer. The native model
continues to remove unavailable raw pixels before its transformer blocks.
No complete teacher, GT feature, fitted coefficient, optimizer, learned
gate, template policy or new model weight is involved.

This is an ordinary control, not a proposed novel module or integrated
FlexTrack implementation. Existing source review describes cross-modal
feature substitution as prior work; no external code is imported here.

Use seed1702, batch4,16tuples and all3configured search frames.
The input population is the existing705train sequences from
`reports/server28-protected-sixepoch-20261030/inputs.json`, disjoint from
its177development and97heldout sequences. The loader uses val-mode
transforms on this train-only population; this is not a new validation set.
No performance-based sequence/sample filtering is used.
The diagnostic creates new fixed image/annotation hashes before evaluation.
GT constructs the ordinary offline crops and task loss, but does not choose
the copied features. These are GT-based crops, not closed-loop tracking.

The initial template stays complete. Dynamic-template availability and
search availability independently take11/01/10:9cells. Every cell compares
native and observed-copy arms on identical samples, giving18arm-cells with
16losses each. All three searches use the chosen search availability.

## Checks And Limits

The producer strictly loads the packaged E0 checkpoint, verifies its
unchanged tensor-package receipt, freezes all model parameters and saves
source/checkpoint/manifest/input hashes. It checks:

- All3search forwards and the2template/64template-token/256search-token layout.
- Unchanged observed positions at block6.
- Exact complete-history11_11 output-map hashes and losses across arms.
- Finite model outputs and per-tuple native task losses.
- Unchanged whole-model state digest after all interventions.

Separate post-run arithmetic analysis checks source/checkpoint hashes,
all36batch/state records,16losses/arm/cell, per-search copied-token counts,
complete-history equality and win/tie/loss counts. It does not re-execute
the producer's network witnesses. The focal loss has batch-coupled scaling;
the16tuples are correlated and are not16independent videos. No confidence
interval or PR/SR claim is planned.

CPU-only with4Torch threads. No active GPU training or registered geometry/
gap source is edited. The existing original-weight cleanup does not cover
this E0 checkpoint. Keep this source unchanged until analysis finishes.
Do not choose a modality gate or promote the copy arm from partial results.

Output: `reports/server28-observed-copy-20261031`.
Launch receipt: `reports/server28-observed-copy-launch-20261031.json`.
Log: `reports/server28-observed-copy-20261031.log`.
The `run` mode saves complete raw results and then runs arithmetic analysis;
it emits `COMPLETE` only after both completion receipts exist.

Provisional duration5-15minutes, to be revised using the first completed
batch. Observe near that milestone with180-300second polling. A launch or
passing complete-history check alone is not a completed experiment result.

## Actual Launch

The new source passed Python3.7 compilation, then launched PID220773.
The actual registration timestamp is2026-10-31 18:31:17 UTC+08.
At18:31:53 the process was alive at360%CPU, RSS approximately1GiB, with
the native model already executing and no completed full-batch receipt yet.
This observation is execution progress, not a completed result.

Source SHA256:
`13ee19f1f82b0d8966c0ab2c2a63039a4b56063c25d1fa470231d2294127b431`.
Registration SHA256:
`cdf8f129b1352a3e9b4127dca2d1a2ef591dfdbce566ebc690f3eb90ec4de40d`.
Fixed16input SHA256:
`52a5ac3b48bea5224dfbf46f53276b718041e1159fd889d1d5a9be0c37a5c7c9`.
Checkpoint is the unchanged E0 package
`c4cc43454f80d838db0130f8664b36e73063aadb99a3a645ee12bfee1fd5c20e`.
Server source and downloaded registration hashes match local files.
Provisional completion18:36-18:46 UTC+08; revise after the first batch.

## Completed Result

The producer finished in360.60seconds. Both completion receipts and the
separate arithmetic analysis exist. Downloaded report/analysis hashes match
their receipts:

- Report: `014156689a47ab34cc750fc55e4b9caa9215019fb30daaadab8a42790658e229`.
- Analysis: `188752b5ab8326a1c214b4865650165ed224d0b1ded7d9fa075e689514d51a43`.

All complete-history equality, observed-position invariance and unchanged
whole-model state checks passed.

| Dynamic/search availability | Native loss | Copy loss | Copy minus native | Wins/ties/losses |
|---|---:|---:|---:|---|
| 11/11 | 3.148465 | 3.148465 | 0 | 0/16/0 |
| 11/01 | 5.437583 | 5.987543 | +0.549961 | 7/0/9 |
| 11/10 | 4.897718 | 5.128312 | +0.230594 | 8/0/8 |
| 01/11 | 3.485264 | 3.320604 | -0.164660 | 7/0/9 |
| 01/01 | 5.460592 | 5.687718 | +0.227126 | 7/0/9 |
| 01/10 | 5.286779 | 5.279069 | -0.007710 | 8/0/8 |
| 10/11 | 3.381500 | 3.423233 | +0.041733 | 5/0/11 |
| 10/01 | 6.884528 | 7.443542 | +0.559015 | 9/0/7 |
| 10/10 | 4.964787 | 5.392993 | +0.428207 | 6/0/10 |

Six of eight noncomplete cells worsen and two improve on mean loss.
Reject unconditional observed-feature copying as a replacement supported
by this diagnostic. The data do not identify a general beneficial modality
gate. Do not tune one on these16correlated tuples. This is a negative
offline control; no PR/SR, trained module or deployment benefit is proved.
