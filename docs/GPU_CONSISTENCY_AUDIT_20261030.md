# Fixed8 native GPU consistency diagnostic

Date: 2026-10-30. SSH2028, project /home/gaob/Miss-RGBT.

User explicitly authorized physical GPU2/3. These were empty before launch.
GPU0/1 formal utility/random training and frozen runtime source were preserved.
The project AGENTS.md and compute ledger now record the expanded authorization.

## Execution

Historical A utility epoch20, fixed first8 val97 entries, all4602 video frames
per condition, native MissingTracker. Physical GPU2/PID3778014 ran complete
input in177.76s; GPU3/PID3778016 ran predetermined alternating outages in156.35s.
Both completed8 sequences normally. GPU2/3 were released after completion.

A read-only utility forward hook copied the final two context channels at
layer6. A model forward hook sampled RGB/TIR similarity at the Hann response
argmax. Available-modality similarity is the arithmetic mean over available
channels. No tracking box, mask, template update, threshold or model parameter
was changed. The script checked finite values, observed-call/hold counts,
native template-update rules and equality of captured/native confidence.
All8 common31-frame prefixes matched exactly across conditions.

Source, checkpoint, protocol, mask, GT and trace hashes are recorded in the
launch/report receipts. Local analysis verified all trace hashes.

## Result

Positive diagnostic label: previous search crop has no positive-area
intersection with the current valid GT. Higher diagnostic score means lower
cue value. No threshold was fitted.

| Condition | Valid observed frames | No-support frames | Hann peak AUC | Raw peak AUC | Available similarity AUC |
|---|---:|---:|---:|---:|---:|
| Complete | 4568 | 784 | 0.564110 | 0.570248 | 0.478631 |
| Alternating outage | 3924 | 1505 | 0.835098 | 0.845034 | 0.765269 |

Layer6 similarity has weaker pooled separation than response in outage and
near-chance separation in complete tracking. This does not justify adding it
as a universal trigger. Combined-cue incremental value has not been tested.
Per-state/per-sequence descriptives are in the analysis JSON; pooled frames
are correlated within sequences and are not independent repetitions.

The initial-template tokens at layer6 have already interacted with current
search tokens through Transformer blocks. This is not an independent frozen
identity verifier. No claim of identity recovery or novel module benefit follows.
GPU trajectories differ from CPU trajectories; no CPU/GPU numerical equivalence
is claimed. Prefix agreement compares the two hooked GPU conditions, not a
hook-off control.

## Files

- Server traces: reports/server28-val97-gpu-consistency-20261030.
- Local traces: outputs/server28-val97-gpu-consistency-20261030.
- Local analysis: outputs/server28-gpu-consistency-analysis-20261030.json.
- Runner: run_gpu_consistency_cohort28.py, outside the frozen runtime.

Next method work needs independent target evidence and train882 calibration,
followed by fixed held-out validation and matched interventions. Do not select
thresholds on val97 or official LasHeR/RGBT234 test results.

## Formal Training Snapshot

At2026-10-30T15:07:17Z utility4866/50000 and random5217/50000 remained live;
all recorded key losses were finite. Recent speeds2.396/2.263s per update
estimate completion October31 at21:10/19:16 UTC (17:10/15:16 EDT;
November1 at05:10/03:16 Beijing time). Estimates can change.
Evaluation controller3463984 remains WAITING_FOR_TRAINING with300-second
checks. The overall research goal remains incomplete.
