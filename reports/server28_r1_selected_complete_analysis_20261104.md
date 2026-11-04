# R1 Selected Epoch16 Complete Result

The first R1 full Complete evaluation finished479 sequences/337353 frames
in6948.8344 seconds. At2026-11-04 11:19:17UTC a separate read-only invocation
of the existing `verify_recovery_campaign28.verify_official` recomputed all
five metrics from saved boxes and ground truth, checked full inventories,
receipts, masks, evaluator/native source identities and saved curves.
The actual epoch16 checkpoint hash was also recomputed.

Epoch16 is the registered final-five development-loss selection. This is
not an official-score-selected epoch. It is one of the twenty full R1/R2
evaluations required by the frozen R/P protocol.

| Complete metric | E0 | R1 epoch16 | Delta pp |
| --- | ---: | ---: | ---: |
| LasHeR PR | 75.636503 | 74.738998 | -0.897504 |
| LasHeR NPR | 71.772837 | 70.997651 | -0.775186 |
| LasHeR SR | 60.125714 | 59.504064 | -0.621650 |
| RGBT234 MPR | 90.677134 | 90.909132 | +0.231998 |
| RGBT234 MSR | 66.845228 | 67.059630 | +0.214401 |

Coverage is LasHeR245 sequences/220703 frames and RGBT234234/116650.
No partial scores, subset evaluation or new tracker execution were used.
CUDA remained uninitialized during the independent saved-output verification.

## Consequence For The Registered Gate

The frozen P entry requires every Complete metric of the internally selected
R1 checkpoint to be at least E0 minus0.1pp. All three LasHeR metrics are
below that tolerance. This verified component therefore does not pass the
registered condition; larger scores in other epochs cannot replace the
development-selected checkpoint. R2 is not an automatic substitute under
that registered R1-specific rule.

This is not the final R campaign receipt or terminal RPPM queue verdict.
The R controllers still owe their remaining full evaluations, and the
persistent P queue still waits for both R terminal official receipts.
Preserve that protocol and its eventual recorded outcome. Do not change
the tolerance, selection rule, R1 identity or frozen queue sources post hoc.

Any R3 or alternative continuation needs a separately registered protocol.
Future choices should use development evidence, with official benchmarks
reserved for final reporting, as discussed in
`docs/RESEARCH_DECISION_20261104.md`. This observation does not establish
which change would repair the regression.

## Interpretation Limits

This checkpoint improves full RGBT234 by approximately0.23/0.21pp but
regresses full LasHeR. It neither establishes the requested roughly2pp
cross-dataset gain nor supports a claim that OSTrack initialization alone
has solved complete localization. Its Miss evaluation is still pending;
Complete metrics cannot establish either Miss threshold or robustness.
R1/R2 are protocol bundles, so their comparison cannot isolate one
optimizer, batch-size or initializer cause without matched controls.
No three-module efficacy conclusion follows from this result.

## Evidence

- Checkpoint SHA256:
  `0fe0af921bfc4ce2eba1301d3bc6f37a7adf68ea32a93542ecfc822dbcb8e03b`.
- Full completion receipt SHA256:
  `14512320c6241259976ed971bed5bde4954effbd3f38ecffe68193041c660f31`.
- Saved metrics SHA256:
  `920b98b96d9fa7b092c535eeedd368f920d145ef5788a21eaaeed822861b35a9`.
- Independent verification record:
  `reports/server28_r1_epoch16_complete_verified_20261104.json`, SHA256
  `1e12d14bb94125303264f6910c4934b742da543166cfc192f9e7f492952b3469`.

All checkpoints remain retained. No live config, frozen source, running
controller or growing training manifest was changed.

At11:20:23UTC the original R1 controller automatically launched epoch16
Miss on GPU0, PID2331902, using the same checkpoint and four workers. Its
live process was confirmed about one minute later. No manual dispatch or
duplicate evaluation was performed.
