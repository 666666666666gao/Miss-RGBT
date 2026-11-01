# Joint Utility-Set Diagnostic

## Motivation

`TaskUtilityRepair` selects four regions at inference, while its auxiliary
supervision samples one region and labels its individual task-loss change.
Prior selector audits on small cohorts found weak and state-dependent
single-region ranking. The unresolved question is whether the trained
utility ordering improves the **joint** four-region intervention over
equal-budget alternatives, and whether single-region gains add up.

This is a preregistered mechanism diagnostic, not a new method claim or a
benchmark evaluation.

## Frozen Protocol

- Checkpoint: Fixed A Utility epoch16, the first epoch in the fixed
  final-five window, selected by the training protocol rather than official
  test ranking. SHA256:
  `42ed78bb821a34f2695423a3a5d8db3ea0af0ab7820022b78588378fea803c8d`.
- Data: 128 causal tuples sampled with seed28001 only from the 882-sequence
  LasHeR train split. Record sequence/frame identities and input hashes.
- Conditions: complete search, RGB-only search, and TIR-only search;
  both template frames remain complete.
- Interventions share the same frozen pre-repair features, correction
  weights, and six-block prediction tail: no correction, utility top-4,
  random top-4, and initial-template similarity top-4.
- For utility top-4, additionally measure each selected region alone. Compare
  the joint loss gain with the sum of those four single-region gains.
- No optimizer step, val97 access, official-test access, parameter selection,
  or checkpoint cleanup.

## Decision Gate

Advance to a matched training comparison for joint utility supervision only
if, in **both** missing-search conditions, the sequence-clustered bootstrap
95% lower bound of utility-top4 loss improvement over both random-top4 and
similarity-top4 is greater than zero. The complete condition is an
instrumentation parity check: normal forward and replayed utility-top4
predicted boxes must differ by less than `1e-5`.

If the gate fails, do not train this candidate. A positive result would
justify only a matched train-only training experiment; it would not establish
closed-loop tracking improvement, novelty, or a paper module.

## Execution

Runner: `scripts/diagnose_joint_utility_sets.py`.

Server: `2028`, GPU2, runtime
`/home/gaob/Miss-RGBT/runtime/task_utility_20261018`.

Output:
`reports/server28_joint_utility_sets_20261102/diagnostic.json`.

Poll the actual process at 180-300 second intervals near estimated
completion. Do not restart after an SSH observation timeout without checking
the process and output receipts.

## Completed Result

Completed on server28 at approximately 2026-11-02 07:14 UTC+08,
equivalent to 2026-11-01 18:14 America/New_York. Actual process4108301
was observed alive and producing batches, then absent after all receipts
were written. All128 tuples /384 condition rows completed, covering118
train882 sequences, with batch8 on GPU2. No optimizer update occurred.

The separate `scripts/analyze_joint_utility_sets.py` recomputed every saved
mean, sequence-bootstrap interval, input pairing, region-set cardinality,
and decision gate. It verified report/completion agreement and the train
sequence-list hash. The report SHA256 is
`1a36cb0f3235fc5d2ad56821efe16d3a6e110a0bcc7ec17e806589b2ac37f0d7`.

Positive contrasts below mean a lower loss for utility top4. These are
sequence-macro training-objective diagnostics, not official tracking scores.

| Search | Utility vs random gain [95% CI] | Utility vs similarity gain [95% CI] | Utility joint gain vs off [95% CI] |
|---|---:|---:|---:|
| Complete | +0.000285 [-0.000426, +0.001048] | -0.000149 [-0.000757, +0.000455] | +0.000024 [-0.000867, +0.000899] |
| RGB only | +0.000069 [-0.000282, +0.000476] | -0.000136 [-0.000383, +0.000110] | +0.000058 [-0.000209, +0.000353] |
| TIR only | -0.000374 [-0.003970, +0.003216] | +0.000179 [-0.004195, +0.004632] | -0.002957 [-0.007742, +0.000714] |

The joint-gain-minus-summed-single-gains residual is +0.00000267
[-0.00004904,+0.00005391] complete, -0.00000896
[-0.00005963,+0.00004775] RGB-only, and -0.00009651
[-0.00044070,+0.00028507] TIR-only. Every interval crosses zero.
There is no demonstrated directional interaction supporting the proposed
joint-supervision explanation on this checkpoint/cohort. This does not
prove exact additivity or rule out interactions on other inputs.

Decision: registered advance flag **false**. Do not launch joint-utility
training from this diagnostic. Current utility selection has no established
advantage over these equal-budget controls; weak correction signal and
selector quality remain unresolved rather than established causes.

Limitations: one checkpoint, one sample seed, one search frame per tuple,
complete template histories, and one random subset per tuple/state. The
run asserts native/replayed utility predicted-box tolerance but does not
persist all-map parity, repeated-native noise or before/after frozen-state
digests. The analyzer checks saved arithmetic, not independent neural replay.
This is neither a fresh-agent integrity review nor validation of a module.
Preserve the final-five checkpoints and all diagnostic artifacts.
