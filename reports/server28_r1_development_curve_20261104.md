# R1 Completed Development-Loss Curve

## Scope

Read-only analysis on 2026-11-04 of the completed R1 run:20 epochs,50000
optimizer updates, effective batch8. Each epoch evaluates the same128
cached development tuples under search availability11,01,10. Templates
remain complete. These are fixed-crop task losses, not sequence-tracking
scores, benchmark results, or a newly executed validation experiment.
The repeated tuples and single seed do not provide independent replication.

`research/protocol_r/train.py:validate` sums per-sample task losses and
divides by128 for each state; the reported mean gives the three states
equal weight. `miss_rgbt/objectives.py:task_loss` retains a per-sample loss
and sums across search frames. Values below retain this original scale.

## Raw Epoch Table

Lower loss is better. Values are rounded to six decimals from the terminal
history JSON.11=Complete,01=RGB missing,10=TIR missing.

| Epoch | Complete | RGB missing | TIR missing | Three-state mean |
| --- | ---: | ---: | ---: | ---: |
| 1 | 4.415439 | 6.181858 | 4.909837 | 5.169045 |
| 2 | 4.254825 | 6.166029 | 4.783752 | 5.068202 |
| 3 | 4.220235 | 5.930922 | 4.739513 | 4.963557 |
| 4 | 4.206787 | 5.886752 | 4.841265 | 4.978268 |
| 5 | 4.020705 | 5.890444 | 4.596308 | 4.835819 |
| 6 | 4.083587 | 5.829470 | 4.661235 | 4.858098 |
| 7 | 4.084982 | 5.678448 | 4.621506 | 4.794979 |
| 8 | 4.082959 | 5.488451 | 4.618124 | 4.729845 |
| 9 | 4.032979 | 5.482406 | 4.594472 | 4.703285 |
| 10 | 4.117356 | 5.607001 | 4.928923 | 4.884427 |
| 11 | 4.065519 | 5.443733 | 4.816884 | 4.775379 |
| 12 | 4.085916 | 5.375468 | 4.759183 | 4.740189 |
| 13 | 4.044942 | 5.356563 | 4.745253 | 4.715586 |
| 14 | 4.067020 | 5.341878 | 4.702796 | 4.703898 |
| 15 | 4.021005 | 5.350840 | 4.633272 | 4.668373 |
| 16 | 4.033502 | 5.401523 | 4.641469 | 4.692165 |
| 17 | 4.031221 | 5.399566 | 4.669109 | 4.699966 |
| 18 | 4.084855 | 5.403682 | 4.670903 | 4.719813 |
| 19 | 4.070341 | 5.449612 | 4.690501 | 4.736818 |
| 20 | 4.064658 | 5.437636 | 4.674747 | 4.725680 |

## Findings

1. The lowest three-state mean over the complete history is epoch15,
   4.668372538. The preregistered selection considers only epochs16-20,
   therefore still selects epoch16,4.692164507. Its loss is0.509642%
   above the descriptive all-epoch minimum. This is not permission to
   change the selected checkpoint or the official evaluation schedule.
2. Final-five mean loss is4.714888297, with range0.044653469. Epoch20 is
   0.714294% worse than epoch16, and no final-five epoch improves on15.
   Thus extending this run to20 did not yield sustained late improvement
   on this particular proxy. This does not establish tracking convergence
   at15, an optimal epoch budget, or that more training necessarily hurts.
3. State-specific minima occur at different epochs:Complete5,4.020704817;
   RGB missing14,5.341878168; TIR missing9,4.594471678. Equal averaging
   can hide different temporal patterns. These are diagnostics within one
   model history, not three independently selected model results. They do
   not identify gradient conflict, module damage, or a data-distribution
   cause, and do not justify adding stronger masking.
4. The later proxy values support examining epochs14-16 as a monitoring
   window in a separately registered future run. The epoch10 loss spike
   and recovery show why a single early improvement is insufficient for
   a stopping decision. Any new stopping rule must use development data,
   specify patience/minimum improvement before launch, and be checked
   against full retained-epoch tracking results. No such rule is applied
   retroactively to the fixed20-round R/P campaigns.

## Next Evidence Required

Finish the existing ten full R1 evaluations and ten R2 evaluations under
their current controllers. Keep the registered development selection and
all checkpoints. Evaluate whether the proxy-selected model also preserves
Complete localization using the full reported metrics; do not select a
replacement from official-test maxima. The RPPM acceptance gate and its
test-conditioned continuation limitation remain unchanged. This analysis
does not satisfy any benchmark threshold or promote any paper module.

## Evidence Identities

Server root:`/home/gaob/Miss-RGBT`. History had exactly20 ordered epoch
entries, each with128 samples. Local source hashes match the two files in
the frozen R runtime; this comparison does not certify historical inputs.

| Artifact | SHA256 |
| --- | --- |
| runs/server28_protocol_r_20261103/r1/development_history.json | f2d6577ac0e7c1930a906d4aaaee1fc36fbb243e358250b476936798757be00c |
| runs/server28_protocol_r_20261103/r1/completed.json | 4eb984002f5599dadd19b4e834ab800a8eea3e98b75d265c2004958fc2f585ae |
| runtime/protocol_r_20261103/research/protocol_r/train.py | 1709f566d80cad56405d0601ea0f1301168d559c70f4790086bfaebfebc4d5d3 |
| runtime/protocol_r_20261103/miss_rgbt/objectives.py | 12de96674e86faac3e8e771035b244f41279e44b07bac4cf7315b4ad884995aa |

The completed-receipt hash was reconfirmed by the live controller's
`r1_training_verified.json` in the preceding read-only check; the history
and source hashes above were directly computed for this analysis. No
training manifest, image, checkpoint tensor or active configuration was
read or changed for this report.
