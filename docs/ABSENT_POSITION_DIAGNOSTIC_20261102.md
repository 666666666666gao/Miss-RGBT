# Absent-Stream Position Diagnostic

## Question

The frozen model gates unavailable patch content before adding the shared
absolute position embedding. Earlier train882 diagnostics found that bypassing
all block updates to the absent stream and masking its search tokens as
attention keys/values both reduced crop IoU. This diagnostic isolates the
input position embedding: does the unavailable stream's absolute search
position help the native head, or can it be removed without losing its
spatial scaffold?

This is a frozen single-crop mechanism diagnostic. It is not a proposed
module, closed-loop tracking result, official metric, or paper contribution.
Availability-conditioned denoising and missing-feature reconstruction are
established prior art; this intervention is only a code-specific control.

## Registered protocol

- Checkpoint: original CADTrack E0 epoch20,
  SHA256 `d20c8641e4263df3ceadf9e430d0ac96c181c13ad482bca28c2dddef592a8bb8`.
- Data: 128 seeded first-search tuples from the 705 fitting names in
  LasHeR train882. Exclude exact sequence/template-frame/search-frame tuple
  identities present in the fusion, head-supervision, modality-head,
  response-geometry, and absent-search K/V pilot records.
- Conditions: all nine dynamic-template/search availability states. The
  initial template is complete. Inputs, model weights, native head, and
  targets are paired across arms.
- Native arm: unchanged frozen E0.
- Intervention: at the first Transformer block input, subtract the shared
  absolute search-position embedding from search tokens of the unavailable
  modality only. The stream remains in all Transformer blocks and is not
  masked as queries, keys, or values.
- Matched sham: apply the same subtraction to the available search stream.
  This is a capacity-free intervention control, not an inert sham.
- Readouts: native-minus-changed task loss and changed-minus-native IoU,
  averaged across the three dynamic-template states for each missing-search
  direction, with sequence-clustered 10,000-draw bootstrap intervals.
- Controls: complete-search outputs must be bitwise equal to native;
  checkpoint/state digest must remain unchanged; parameter gradients must
  remain absent; tuple overlap exclusions and finite metrics are required.

Advance only to a new train882 closed-loop diagnostic if, for both
missing-search directions, removing absent-stream position improves
sequence-macro crop IoU by at least 0.01 with a positive 95% lower bound,
and its paired IoU advantage over the matched sham also has a positive 95%
lower bound. A pass would justify only a separate closed-loop diagnostic,
not training or official evaluation. A failure closes this exact
position-removal hypothesis on this cohort.

No val97, RGBT234, LasHeR test, optimizer, threshold tuning, or checkpoint
selection is permitted. Preserve all weights and evidence.

## Execution

Use server28 GPU2 after confirming it is idle, below 85 C, and limited to
250 W. Run the registered script with Python 3.7/Torch 1.12 in the frozen
runtime. Expected runtime is under two minutes based on the previous
single-crop diagnostic; inspect once near that estimate, then wait
180-300 seconds if it remains active.

## Result

The first two launch attempts stopped before model inference because the
registered prior-pilot reports were stored outside the runtime working
directory and used three verified identity-record schemas. Their empty
outputs and logs are retained. The third attempt completed computation but
an implementation review found its target/sham stream routing was fixed to
RGB/TIR order rather than selected from each missing-state condition. Its
metrics are invalid and are not interpreted. The fourth run corrects the
condition-specific routing and uses a fresh output directory; only that run
answers the registered question.

The corrected run completed on server28 GPU2 in60.109seconds. It used128
tuples covering120 fitting sequences and384 rows per missing-search
direction. Independent analysis rechecked the report digest, all row
identities/cells, finite readouts, bootstrap intervals, state/gradient
receipts, and frozen runtime source hashes. The model state digest was
unchanged, parameter gradients were absent, and all384 complete-search
intervention rows matched native outputs exactly.

Effects are intervention minus native; positive IoU favors removing the
absent stream's absolute position embedding. Task-loss effects use
native-minus-intervention, so negative means the intervention increased
loss.

| Missing search stream | IoU change [95% sequence-bootstrap interval] | Task-loss change [95% interval] | Intervention vs matched sham |
|---|---:|---:|---:|
| RGB (`01`) | -0.053025 [-0.080336, -0.024700] | -0.342150 [-0.471263, -0.203991] | -0.053025 [-0.081781, -0.024710] |
| TIR (`10`) | -0.006421 [-0.015715, +0.002530] | -0.114635 [-0.186047, -0.037842] | -0.006421 [-0.015811, +0.002804] |

Removing the absent stream's position embedding clearly harmed the RGB-missing
crop readout and did not improve TIR-missing IoU. The registered advancement
gate is false. Do not train, tune this cohort, or run an official evaluation
for this intervention. This result does not alter the current official
metrics or validate a paper module.

Corrected report SHA256:
`3383b6f752c4e61fb97e857fead2e847f31adb9225663e17df9c6fec6b7d0b41`.
Evidence is retained in `reports/server28_absent_position_20261102_v4/`;
the earlier invalid run is retained separately on server28.
