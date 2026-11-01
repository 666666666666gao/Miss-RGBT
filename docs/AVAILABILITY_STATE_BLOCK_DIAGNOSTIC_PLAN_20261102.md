# Availability-State Block Diagnostic

Status: proposed diagnostic only; not a paper-module claim and not launched.

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

If it passes, train one native control and one intervention arm with matched
data order, seed, schedule, optimizer, and 20-epoch budget. Use two idle GPUs,
one per arm; calibrate the largest stable batch size first and scale learning
rate in proportion to the reference global batch size. Keep epochs 16-20 for
full official `complete` and `miss` evaluation, 479 sequences/337,353 frames
per run. Report every audited final-five result and disclose that selecting
the maximum official-test score is post-hoc. Preserve all weights until all
receipts and audits finish.

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
