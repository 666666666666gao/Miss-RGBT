# Crop Exposure Prior Art Boundary

The present experiment is a frozen causal diagnostic, not a new tracking
module. Previous-GT crops supply unavailable annotation information.
Improving with an oracle would not establish a useful training policy.

## Sequential prediction

Ross, Gordon and Bagnell, AISTATS2011, "A Reduction of Imitation Learning and
Structured Prediction to No-Regret Online Learning", PMLR15:627-635,
explicitly studies predictions that affect future observations and an
iterative policy-learning reduction. Its general principle is established
prior art; collecting predictions and training on induced states is not a
novel contribution here. No implementation or guarantee is transferred to
our tracker.

Primary source inspected2026-11-02:
https://proceedings.mlr.press/v15/ross11a.html

## Tracking implementation

Read-only existing local `visionml/pytracking` checkout and Git remote HEAD
both resolve to7eb9e74bd3d40e29dbcec444902237da13de247b.
`pytracking/util_scripts/create_distractor_dataset.py` lines108-138
initializes a tracker, runs it sequentially, extracts candidate/GT-derived
state labels, and persists candidate data. This supplies a concrete example
of data collection from tracker trajectories. It does not implement our
paired previous-GT crop experiment, prove benefit for Miss-RGBT, or establish
that more crop exposure resolves sensor absence.

The existing repository review records GPLv3. No source file is copied,
linked into the runtime, or installed. Raw UTF8 content SHA256:
bcb274bc0da7daa54e253e6f3fd31f7f37970afdde75ec7ce82fcbdcf0fa75f6.
GitHub API access returned a rate-limit error; the commit was verified
using git ls-remote and the pinned raw file instead.

Primary source:
https://github.com/visionml/pytracking/blob/7eb9e74bd3d40e29dbcec444902237da13de247b/pytracking/util_scripts/create_distractor_dataset.py

## Consequence

If the registered train-only diagnostic passes, any next experiment needs
matched native training images, an actual causal crop policy, independent
train-split development/audit trajectories, complete-input protection,
measured full-image batch/LR behavior and final-five preservation. Ordinary
trajectory augmentation is a control until missing-modality-specific
mechanism evidence and broad performance establish a defensible module.
