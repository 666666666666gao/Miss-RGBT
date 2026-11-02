# Causal Crop Exposure Diagnostic

## Evidence and question

The native training processor jitters each search crop around its current
GT box. MissingTracker instead crops at its previous prediction. Its three
training searches do not feed predictions into subsequent search crops.
Previous historical oracle-crop findings used development data and current
GT centers; they are not new evidence for this train-only experiment.

Measure whether replacing only the crop's previous predicted box with the
previous GT box changes next-frame localization, while keeping the native
template history fixed. This is an unavailable oracle diagnostic, not a
deployable tracker, novelty claim, or official metric.

## Fixed protocol before launch

- Original frozen CADTrack E0; original checkpoint SHA256
  d20c8641e4263df3ceadf9e430d0ac96c181c13ad482bca28c2dddef592a8bb8.
  Use the existing availability wrapper checkpoint, also record its hash,
  and assert its loaded weights equal the original model's state digest.
- Seed 57101; 48 unique fitting-sequence clips, 96 contiguous frames each.
  Existing train882 sequence partition: index modulo5 nonzero (705 names).
  All GT boxes must be finite/positive and all frames visible. Exclude
  frames referenced by prior pilot tuple identities and the prior burst
  rollout. Sample once, before inspecting model outcomes.
- Two native trajectories per clip: complete availability and IID mixed
  availability (.4 complete/.3 RGB missing/.3 TIR missing), first frame
  complete. No both-sensors-missing state. Save actual schedules.
- At every subsequent frame snapshot the native prior state, templates,
  availability and frame counter. Run a shadow MissingTracker.track call
  using the previous GT box as crop state, then restore the snapshot and
  run the native call. Shadow updates are discarded; only the native
  prediction/template history persists. Both arms use identical current
  images, observed availability, Hann decode, clip and update rules.
  First3 frames per clip/trajectory additionally run and discard a native
  sham from the snapshot; boxes/confidences must match the retained native
  call exactly. At first frame shadow/native must match as previous box
  equals GT initialization. Assert the snapshot is restored before native.
- Persist every prior/GT/predicted box, availability, confidence, per-frame
  IoUs, template availability and input image-sequence hashes. Retain model
  state digest before/after and assert absent gradients and finite records.
  Describe any model-state assertions as producer checks, not replay.
- Report sequence-macro shadow-minus-native IoU under complete trajectory,
  mixed complete frames, mixed RGB-missing and mixed TIR-missing frames,
  and pooled mixed single-modality frames. 10000 paired-sequence bootstrap
  resamples, seed57301. Also report IoU by normalized prior-center error;
  this is descriptive, not a new decision threshold.
- Advance only to a separately registered matched image-training pilot if
  pooled mixed single-modality delta >=.02, its95% lower bound >0, and all
  48 sequences contribute. Failure closes this exact crop-exposure cohort;
  do not retune schedules, clips or gate. A pass establishes causal crop
  dependence under native histories, not trainability, closed-loop efficacy,
  benchmark improvement or a validated paper module.

## Resources and integrity

Warm-reuse server28 environment; GPU2 only after occupancy<500MiB,
temperature<85C and existing250W power-limit checks. Native causal
batch1 inference is required for the intervention; no artificial padding
to occupy memory. No learning rate or optimizer. Estimated10-20min;
inspect near estimated milestones at180-300second intervals.
Preserve original/final-five weights. No val97, official frames/masks,
checkpoint selection, cleanup, or changes to frozen runtime source.
Independently recompute arithmetic, sequence partition/exclusions,
schedules, sham controls, source/checkpoint/protocol hashes and bootstrap.
No raw-image neural replay is promised.

Full complete-input+2 goal, RGBT234-Miss88.1/64.3 targets, and three
validated modules remain incomplete.

## Completed result

GPU2 PID804850 completed in594.422seconds,48unique fitting sequences,
96frames/clip,9120 paired-frame rows across complete/mixed trajectories.
The mixed trajectory contains2731single-modality frames. No training,
test data, official masks or checkpoint selection occurred.

Sequence-macro previous-GT-crop minus native-prediction-crop IoU:

| Readout | Delta | 95% sequence bootstrap interval |
|---|---:|---|
| Complete trajectory | +0.032329 | [+0.010975,+0.059321] |
| Mixed trajectory, complete frames | +0.057996 | [+0.028700,+0.092851] |
| Mixed trajectory, RGB missing | +0.047694 | [+0.021744,+0.081731] |
| Mixed trajectory, TIR missing | +0.038413 | [+0.019240,+0.061804] |
| Mixed trajectory, all single-modality | +0.042196 | [+0.022036,+0.066595] |

Registered advance=true. This supports separately registering a matched
image-training crop-exposure experiment; it does not show that training on
these states succeeds or that the oracle is a usable tracker. Complete
tracking also improves under the oracle, so this is not evidence of a
uniquely missing-modality mechanism.

Post-hoc nominal factor4 crop geometry:2696/2731 mixed missing frames
fully contain the current GT box. In that descriptive subset the
sequence-macro delta is+.039820,95%CI[+.021298,+.062868]. Only15frames
from4sequences have zero nominal overlap;20frames from6sequences have
partial overlap. Thus this short-clip effect is not solely nominal target
escape. These are rounded bounds before image padding, selected after
outcomes, not an additional gate or a claim about long official videos.

Both server and local analyzers recomputed every saved IoU, prior center
error, native-history continuity, schedules, first-frame parity, all288
native sham checks, frame exclusions, partitions, bootstrap and gate.
Server checked actual annotation/visibility, source hashes, original and
wrapped checkpoint hashes. In a separate process, first-clip190rows were
replayed: native predictions/confidences exactly matched; direct shadow
prediction with independently written inverse-crop mapping had zero box
error and exactly matching confidence. It reuses native network/crop
utilities and is not full-report image replay or external review.
Frozen-state/absent-gradient checks are producer assertions.

During bookkeeping a temporary local runner edit was reverted before
deployment; final local/server/launch runner SHA256 all equal
ae3e517c90085b5e371f7dd99ec16041f0cb07ecd423dd7ffc862d7f7f7b7c53.
The running server source was never changed.

Report SHA256:
ef0e7aa20a83097e265e2603d3b4ffdb328bc4c0c6aa30f6fe1e23e458be6a2c.
Evidence: `reports/server28_causal_crop_exposure_20261102/` and adjacent
immutable launch/log/protocol snapshot. Official scores remain unchanged.
All original/final-five weights remain present. Full objective is unmet.
