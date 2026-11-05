# Late-Fusion Stream Contribution Diagnostic

## Question

CADTrack encodes RGB and TIR tokens through separate calls to shared
Transformer blocks. Its CENTER path concatenates the two search streams and
passes them to `decode_fuse_search`; availability is not supplied to that
fusion layer. Earlier interventions changed missing-stream processing inside
the Transformer. This diagnostic isolates the downstream contribution: after
all native stream processing, does the absent stream's feature map help or
hurt the fixed fusion/head on training crops?

This is a frozen mechanism control, not a new module or benchmark. Generic
missing-feature reconstruction, temporal denoising, and expert fusion are
established prior art; a positive ablation would not establish novelty.

## Registered protocol

- Checkpoint: original CADTrack E0 epoch20,
  SHA256 `d20c8641e4263df3ceadf9e430d0ac96c181c13ad482bca28c2dddef592a8bb8`.
- Data: 128 seeded first-search tuples from LasHeR train882 fitting names.
  Exclude exact tuple identities in prior fusion, head-supervision,
  modality-head, response-geometry, absent-search K/V, and absent-position
  diagnostics. No val97 or official test data.
- Conditions: all nine dynamic-template/search availability states; initial
  template complete. Native, target intervention, and matched sham use the
  same inputs, weights, targets, and native head.
- Target intervention: immediately before `decode_fuse_search`, zero the
  feature channels belonging to the unavailable search stream. The missing
  stream still passes through every native Transformer block.
- Matched sham: zero the available search stream at the same fusion input.
- Readouts: native-minus-changed task loss and changed-minus-native IoU,
  pooled equally across the three dynamic-template states for each missing
  search direction. Use sequence-clustered 10,000-draw bootstrap intervals.
- Controls: Complete outputs must match native bitwise; frozen model state
  digest must remain unchanged; gradients must remain absent; tuple
  exclusions and finite metrics are required.

Advance only to a fresh train882 closed-loop diagnostic if, in both
missing-search directions, the target intervention improves sequence-macro
crop IoU by at least0.01 with a positive95% lower bound and its paired
advantage over the sham also has a positive95% lower bound. A pass does not
authorize full training or official evaluation. A failed gate closes this
specific fusion-input ablation on this cohort.

## Execution

Use server28 GPU2 only after checking it is idle, below85 C, and limited to
250W. Run in the frozen Python3.7/Torch1.12 runtime. Estimated runtime is
under two minutes; inspect once near that estimate, then use180-300second
waits if it remains active. Keep all checkpoints and caches.

## Result

Pending execution.
