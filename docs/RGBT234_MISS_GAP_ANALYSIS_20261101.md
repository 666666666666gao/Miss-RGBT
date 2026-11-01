# RGBT234-Miss Gap Analysis

## Official Full-Miss Results

All entries below are complete official evaluations. Each result covers 479
sequences and 337,353 frames across LasHeR and RGBT234.

| Model | RGBT234-Miss MPR | RGBT234-Miss MSR |
|---|---:|---:|
| Original CADTrack E0, epoch20 | 82.794815 | 59.574435 |
| Historical A Utility, epoch20 | 84.964353 | 60.511286 |
| Batch10 Utility, epoch13 | 84.903450 | 60.656957 |
| Requested threshold | >88.1 | >64.3 |

The requested RGBT234-Miss thresholds are +5.305185 MPR and +4.725565 MSR
above the measured E0 baseline. Historical A adds +2.169538 and +0.936851;
Batch10 Utility epoch13 adds +2.108635 and +1.082522. Thus the current
training has not materially changed this bottleneck, and the requested
threshold is not equivalent to a two-point gain over E0.

Batch10 Utility epoch13 is nearly tied with historical A: -0.060903 MPR and
+0.145671 MSR. It is not the final-five winner. The final-five official
evaluations are still running; no final model selection is justified yet.

## What the Existing Controls Show

- The published missing-state masks have similar missing-frame rates:
  approximately 83.9% for RGBT234 and 84.7% for LasHeR. Missing frequency
  alone is not a demonstrated explanation for the larger RGBT234 target gap.
- In the fixed-weight template control, current dynamic updating (U1) exceeds
  disabled updating (U0) on E3 RGBT234-Miss by 2.897473 MPR and 1.614670 MSR.
  U1 still scores 85.063210/60.325720, below the requested thresholds.
- Freezing all BN running statistics did not close the gap: E3-allBN and
  E4-allBN scored 85.063210/60.325720 and 83.975517/60.033254.
- These are controlled observations, not a single-cause diagnosis. No
  independently certified RGBT234 development split exists. Do not tune
  model choices or thresholds against RGBT234 test scores and call that
  validation.

## Current Official Evaluation Queue

At 2026-11-01 14:20 CST, all four official evaluators were alive. Fixed A
epoch16 `complete` finished for both arms and passed an independent full-scope
check: each run has 479 receipts and 337,353 summed frames, matching
`completed.json` and `run.json`; checkpoint and evaluator hashes match the run
manifests. These are complete-input results, not Miss results:

| Arm | LasHeR PR/SR | RGBT234 MPR/MSR |
|---|---:|---:|
| Fixed A Random epoch16 complete | 74.490607 / 59.572963 | 91.043051 / 66.287325 |
| Fixed A Utility epoch16 complete | 75.425079 / 60.316126 | 90.847494 / 65.941674 |

The four queues then advanced to:

| Queue | Current evaluation | Sequence receipts |
|---|---|---:|
| Batch10 Random | epoch14 complete | 136/479, 236,020 frames |
| Batch10 Utility | epoch14 complete | 155/479, 247,589 frames |
| Fixed A Random | epoch16 miss | 239/479, 287,601 frames |
| Fixed A Utility | epoch16 miss | 212/479, 276,434 frames |

The receipt counts are progress only, not performance metrics. GPUs 0-3 were
active at 249-250 W in the 14:15 CST sample; temperatures were 71-78 C.
Preserve all weights. Do not run val97. Report Miss scores only after the
official full evaluation has 479 sequence receipts and 337,353 frames and its
aggregate result has been independently checked.

Selecting the best checkpoint from test-set final-five scores has selection
bias. Report the requested best-of-five result with that limitation explicit;
do not present it as an unbiased estimate.
