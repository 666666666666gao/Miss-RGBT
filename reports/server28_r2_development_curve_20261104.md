# R2 Completed Development-Loss Curve

## Evidence And Scope

Read-only analysis on 2026-11-04 of the completed R2 training run. Both
its completion receipt and original controller's training verification
confirm 20 epochs, 25,000 optimizer updates and selected epoch17. Raw
terminal history, config and receipts were copied unchanged into
`reports/server28_r2_terminal_20261104/`; local SHA256 values match the
server files. All 20 ordered records have 128 samples, 1,250 updates per
epoch, and a mean equal to the three task losses within 1e-12.

These are previously saved fixed-crop development losses, not new tracking
evaluations or official scores. The existing frozen validation sums native
per-sample losses over the three search images, averages over 128 cached
tuples, and weights availability states 11/01/10 equally. Templates remain
complete. The same tuples and single seed do not establish replication.

## Raw Epoch Table

Lower loss is better; values are rounded to nine decimals.

| Epoch | Complete 11 | RGB missing 01 | TIR missing 10 | Three-state mean |
| --- | ---: | ---: | ---: | ---: |
| 1 | 4.538521022 | 6.288041957 | 4.924367324 | 5.250310101 |
| 2 | 4.269525822 | 6.159669623 | 4.769075248 | 5.066090231 |
| 3 | 4.376879368 | 6.070819609 | 4.816920906 | 5.088206628 |
| 4 | 4.233827945 | 5.857439578 | 4.753301866 | 4.948189796 |
| 5 | 3.971061368 | 5.754961170 | 4.531995259 | 4.752672599 |
| 6 | 4.120476987 | 5.858673207 | 4.673262827 | 4.884137674 |
| 7 | 3.960780989 | 5.681525014 | 4.676954120 | 4.773086708 |
| 8 | 4.110111199 | 5.671834841 | 4.662686531 | 4.814877524 |
| 9 | 4.143984046 | 5.627724394 | 4.721615285 | 4.831107908 |
| 10 | 4.072719425 | 5.593973033 | 4.808055654 | 4.824916037 |
| 11 | 4.095959324 | 5.492933966 | 4.823776159 | 4.804223150 |
| 12 | 4.109809972 | 5.500088088 | 4.783005755 | 4.797634605 |
| 13 | 4.049648620 | 5.434228897 | 4.693529453 | 4.725802323 |
| 14 | 4.093445037 | 5.459166147 | 4.760809742 | 4.771140308 |
| 15 | 4.018993217 | 5.409743533 | 4.709540978 | 4.712759243 |
| 16 | 4.060798667 | 5.439133085 | 4.676356856 | 4.725429536 |
| 17 | 4.039979596 | 5.380386099 | 4.696554717 | 4.705640137 |
| 18 | 4.036432326 | 5.407777295 | 4.752967052 | 4.732392224 |
| 19 | 4.081557963 | 5.454922773 | 4.685538489 | 4.740673075 |
| 20 | 4.085615285 | 5.419628762 | 4.656387862 | 4.720543969 |

## Findings And Consequences

1. Epoch17 is both the descriptive whole-history minimum and the registered
   final-five selection: mean4.705640137195587. Epoch20 is0.316722737%
   worse. The final-five mean is4.724935788412889, range0.035032937924067.
   There is no sustained final-five improvement on this proxy. This does
   not establish tracking convergence or authorize retroactive early stop.
2. State-specific minima differ: Complete at epoch7 (3.960780989378691),
   RGB-missing at17 (5.380386099219322), TIR-missing at5
   (4.531995259225369). The equal mean can conceal that divergence. It
   does not prove gradient conflict, a masking cause or stronger masking
   as a remedy. Keep the single registered reporting checkpoint.
3. R1 and R2 use microbatch4, seed2027, the same initializer, head LR1e-4,
   backbone LR1e-5, warmup500 updates and drop after epoch10. R1 accumulates
   2 microbatches (effective8, 50,000 updates); R2 accumulates4 (effective16,
   25,000 updates). Each sees400,000 accepted training tuples by budget.
   Warmup spans4,000 versus8,000 tuples; the epoch10 boundary follows
   25,000 versus12,500 updates. This compares accumulation/update-budget
   policies at fixed tuple budget and fixed LR, not a controlled test of
   larger batch with proportionally larger LR. Actual historical frame
   provenance remains uncertified, so equal budgets/seeds do not certify
   paired inputs. Do not change the completed or frozen protocols.
4. The already recorded R2 epoch16 Complete scores are an unselected
   reporting row. They must not be presented as the development-selected
   R2 result: epoch17's full Complete/Miss results are still required.
   R1 remains selected at16, and its failed original P gate is unchanged.

For a separately registered future stopping policy, these two histories
support inspecting roughly epochs14-17, while retaining state-specific
curves and predefined patience. They do not justify stopping the registered
fixed20 R3/RPPM experiments or choosing official-test maxima. Finish all
existing full evaluations. The matched R3 objective comparison remains the
next prepared diagnostic, subject to independent review and paired native
smokes; no additional parameter sweep or GPU task is authorized by this
analysis. All produced checkpoints remain retained.

## Source Identities

Server root is `/home/gaob/Miss-RGBT`. No growing manifest, image, weight
tensor, live configuration or frozen source was read or changed.

| Artifact | SHA256 |
| --- | --- |
| runs/server28_protocol_r_20261103/r2/development_history.json | 379d3b22141abc185c51679254524589e4d43ab81489a2d4dd712c59de00e89a |
| runs/server28_protocol_r_20261103/r2/completed.json | a41ab08c7044c0a2aff63425ec9755f77385a31782930749854b80b476a4e8a5 |
| reports/server28_protocol_r_20261103/r2.json | b3342c9ec91960dceedcf7eb520fd1976fd690a8a413e13e63efa477655dd0df |
| reports/server28_protocol_r_20261103/r2_training_verified.json | 18f414b945b71986e61ddfe41071e42e6e9a69cacfd3d2e789af1de4113f0a3b |
| runtime/protocol_r_20261103/research/protocol_r/train.py | 1709f566d80cad56405d0601ea0f1301168d559c70f4790086bfaebfebc4d5d3 |
| runtime/protocol_r_20261103/miss_rgbt/objectives.py | 12de96674e86faac3e8e771035b244f41279e44b07bac4cf7315b4ad884995aa |
