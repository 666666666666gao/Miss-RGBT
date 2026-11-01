# Availability-State Block Diagnostic

Status: completed; candidate rejected by the registered diagnostic gate.

## Evidence and hypothesis

The current runtime gates unavailable patch embeddings before adding position
embeddings. Its shared RGB/TIR Transformer blocks then run self-attention and
MLP separately on each stream without receiving availability. A missing
stream therefore begins as position-only tokens but still receives ordinary
block updates; availability is explicitly used again by the layer-6 repair.
See `MISSING_MODALITY_SOURCE_REVIEW_20261102.md`.

Hypothesis: ordinary updates to the position-only stream alter the final
tracking representation under a missing-modality state. The direction and
whether this matters to tracking are unknown.

## Diagnostic

- Data: 128 fixed tuples sampled only from `train882`, not val97 or either
  official test set. Record tuple identities and input hashes.
- Conditions: complete, RGB-only, and TIR-only, using the same tuples and
  ground-truth boxes.
- Native arm: unchanged checkpoint/runtime behavior.
- Intervention: bypass per-block updates only for the unavailable stream.
- Matched sham: apply the same bypass to the available stream instead.
- Hold the checkpoint, all inputs, RNG state, repair module, and detection
  head fixed. No optimizer updates.
- Readouts: training-objective tracking loss, box localization error, and
  missing-stream feature drift across blocks. These are mechanism diagnostics,
  not benchmark PR/NPR/SR or MPR/MSR.
- Complete-input parity: intervention and native outputs must be bitwise
  identical when both modalities are present.

## Decision

First measure native repeated-run noise, then compare paired tuple-level
effects with sequence-level resampling. Advance only if the unavailable-stream
intervention improves the missing-state tracking loss beyond that noise, the
matched sham does not reproduce the effect, and complete-input parity holds.
Otherwise reject this candidate without a full training run.

## Completed Result

The diagnostic completed on server28 at 2026-11-02 06:53 CST on GPU2. It
used 128 tuples from 118 train882 sequences, batch size32, and the frozen
CADTrack E0 epoch20 checkpoint (`d20c8641e4263df3ceadf9e430d0ac96c181c13ad482bca28c2dddef592a8bb8`).
The train split hash is
`27d39054b194f1b2e855c0bc1a1e23651f5e2b1cfef1e34d88074d71565df846`.
Every tuple records template, search, box, and sequence/frame identities.
No optimizer updates were performed; val97 and official test data were not
used. Complete-input intervention matched native output bitwise, with max
absolute difference0. Native repeat task-loss noise was0 in both missing
directions.

The intervention readout is defined as native minus intervention for task
loss and center error, and intervention minus native for IoU; positive is
improvement. Sequence-macro effects over118 sequences were:

| Search condition | Task-loss change | 95% interval | IoU change | 95% interval | Center-error change |
|---|---:|---:|---:|---:|---:|
| RGB only | -4.847049 | [-5.466646, -4.230780] | -0.226034 | [-0.273621, -0.179301] | -0.059862 |
| TIR only | -12.076711 | [-13.086938, -11.087066] | -0.367300 | [-0.420182, -0.312548] | -0.128255 |

Negative values mean bypassing updates to the unavailable stream worsened
the readout. The intervention did not exceed the repeat-noise gate in either
direction. The registered advance flag is false. Do not launch matched
training for this hypothesis; the observation does not validate a module or
explain the RGBT234-Miss gap.

Full tuple-level evidence remains on server28 at
`runtime/task_utility_20261018/reports/server28_train882_block_state_diag_20261102/diagnostic.json`
and locally at
`outputs/server28_train882_block_state_diag_20261102.json`.
The local JSON SHA256 is
`fa6e4c98b0d51730b288024d59b369581d717c96cbd878ea335d6105da6fa09f`.

This candidate did not pass its diagnostic gate, so its matched-training run
is not authorized by this protocol. Any different mechanism requires its own
predeclared diagnostic and matched control before training.

## Prior-Art Boundary

Do not rename this intervention as a contribution without a broader novelty
check. SCDT already uses modality-state-conditioned denoising and temporal
features. FlexTrack-V2 already includes missing-feature reconstruction,
expert fusion, and completeness-curriculum distillation. The candidate here
tests a different, code-specific question: whether processing the current
runtime's position-only stream through ordinary shared blocks is harmful.
Even a positive diagnostic would establish only a local mechanism signal,
not novelty, official benchmark gains, or one of the three requested paper
modules.

## Execution Notes

The first launch attempt failed before model loading because invoking the
script by path did not put the runtime root on Python's import path. The
second attempt loaded the model but stopped at the first batch: this
repository's `generalized_box_iou` returns per-pair IoU as a vector, not a
matrix. The third attempt exposed a scope mismatch: each sampled item has two
templates and three search frames, while the model invokes the backbone once
per search frame. The diagnostic had concatenated availability for all three
search frames into one mask. It now configures one aligned token mask per
backbone call and retains drift readings for each search frame. Its first
execution also caught a stale CLS-mask variable reference; CLS availability
now derives from the aligned frame mask. No optimizer step occurred in any
failed attempt; GPU2 was released each time. Launch with the runtime root
explicitly set in `PYTHONPATH`.
