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

GPU2 PID275755 completed normally in79.108seconds on2026-11-02 UTC.
The128 new tuples covered117 fitting sequences; each of nine cells contains
128 paired inputs and two interventions, totaling2,304 records.
Positive IoU means intervention-minus-native; positive task loss means
native-minus-intervention. These values are crop diagnostics, not official
tracking metrics.

| Missing search stream | Zero absent stream: IoU delta [95% CI] | Task-loss gain [95% CI] | Absent-vs-available stream ablation: IoU delta [95% CI] |
|---|---:|---:|---:|
| RGB (`01`) | -0.008314 [-0.029745, +0.014144] | -0.105032 [-0.237164, +0.027616] | +0.503744 [+0.450618, +0.553853] |
| TIR (`10`) | +0.010427 [-0.002732, +0.024063] | +0.031917 [-0.033780, +0.098533] | +0.669482 [+0.617250, +0.718507] |

Neither absent-stream ablation establishes improvement over native.
Removing the available stream is substantially worse, but that contrast
does not establish that removing the absent stream is useful. The registered
advance flag is false. Do not train this ablation, tune on these tuples,
or advance it to official evaluation. No paper module is validated.

The producer checked the chosen channel slice was actually zeroed, Complete
outputs equal native maps/boxes, frozen state digests identical, and parameter
gradients absent. The independent saved-evidence analyzer checked the
report/completion digest, all2,304 identities/cells, finite readouts,
six prior-report hashes and exact tuple exclusions, bootstrap recomputation,
registered gate, protocol snapshot hash, and server source hashes. It passed
on server and locally. This is arithmetic/evidence verification, not
independent neural replay. Per-input image tensor hashes and full native
output maps were not persisted; state/parity assertions are producer checks.
Only listed source files are hashed, not every transitive import.

Before launch a source review caught a mask that would have made the
available-stream control a no-op; it was corrected before any run. Both
arms now zero the selected stream whenever search is single-modality.
No failed attempt or restart occurred for this diagnostic.

Report SHA256:
`c523a537ecf00279548f2ed8bec2ea4ff88c256e51537086ed11950adf996e89`.
Full small evidence is in `reports/server28_late_fusion_streams_20261102/`
with sibling launch, log, and preregistered protocol files. No training,
val97, official benchmark access, or checkpoint cleanup occurred.

## Next Implication

Do not treat missing-stream tokens as inherently harmful based on their
availability bit. Together with earlier failed block/KV/position ablations,
this rules out promoting these simple deletion controls on their registered
cohorts; it does not rule out learned interactions. Before another training
candidate, examine a train-only matched interaction experiment with retained
missing-stream spatial/template structure and a spatially shuffled context
control. Establish its own mechanism and Complete-parity evidence, rather
than tuning this ablation against its audit sample. Complete approximately
+2 points, RGBT234-Miss targets, and three validated modules remain unmet.
