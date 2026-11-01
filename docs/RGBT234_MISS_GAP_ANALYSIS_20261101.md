# RGBT234-Miss Gap Analysis

## Official Full-Miss Results

All entries below are complete official evaluations. Each result covers 479
sequences and 337,353 frames across LasHeR and RGBT234.

| Model | RGBT234-Miss MPR | RGBT234-Miss MSR |
|---|---:|---:|
| Original CADTrack E0, epoch20 | 82.794815 | 59.574435 |
| Historical A Utility, epoch20 | 84.964353 | 60.511286 |
| Batch10 Utility, epoch13 | 84.903450 | 60.656957 |
| Fixed A Random, epoch16 | 84.746741 | 60.612033 |
| Fixed A Utility, epoch16 | 84.946943 | 60.677252 |
| Requested threshold | >88.1 | >64.3 |

The requested RGBT234-Miss thresholds are +5.305185 MPR and +4.725565 MSR
above the measured E0 baseline. Historical A adds +2.169538 and +0.936851;
Batch10 Utility epoch13 adds +2.108635 and +1.082522. Fixed A epoch16 adds
+1.951926/+1.037598 (Random) and +2.152128/+1.102817 (Utility). The Utility
epoch16 result remains 3.153057 MPR and 3.622748 MSR below the requested
threshold. The requested threshold is not equivalent to a two-point gain
over E0.

Batch10 Utility epoch13 is nearly tied with historical A: -0.060903 MPR and
+0.145671 MSR. Fixed A epoch16 is also near the historical result (-0.017410
MPR, +0.165966 MSR for Utility). Epoch16 is only the first of five official
checkpoints in this final-five campaign; no final model selection is
justified yet.

The paired complete-input results show substantial Miss-condition drops on
both benchmarks. From complete to Miss, Fixed A Utility drops 7.21/6.07 points
on LasHeR PR/SR and 5.90/5.26 on RGBT234 MPR/MSR; Random drops 6.43/5.44 and
6.30/5.68, respectively. Because PR/SR and MPR/MSR use different benchmark
aggregations, these numeric differences cannot establish that missing input
is more or less damaging on one dataset. They do not isolate a cause for the
target gap. What is established is that RGBT234-Miss remains around
84.75-84.95/60.61-60.68, below the requested 88.1/64.3.

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

At 2026-11-01 14:44 CST, Fixed A epoch16 `complete` and `miss` finished for
both arms. Each of the four runs passed independent full-scope checks:
479 receipts, 337,353 summed frames, matching `completed.json` and `run.json`;
the two Miss runs also match per-dataset receipt/frame counts, and checkpoint
and evaluator hashes match their manifests.

| Arm | LasHeR PR/SR | RGBT234 MPR/MSR |
|---|---:|---:|
| Fixed A Random epoch16 complete | 74.490607 / 59.572963 | 91.043051 / 66.287325 |
| Fixed A Utility epoch16 complete | 75.425079 / 60.316126 | 90.847494 / 65.941674 |

The paired full Miss metrics are LasHeR PR/SR 68.065197/54.134896 and
RGBT234 MPR/MSR 84.746741/60.612033 for Random; Utility is
68.217157/54.247395 and 84.946943/60.677252. These remain intermediate
epoch16 results, not the best-of-five result.

The four queues then advanced to:

| Queue | Current evaluation | Sequence receipts |
|---|---|---:|
| Batch10 Random | epoch14 complete | 281/479, 302,126 frames |
| Batch10 Utility | epoch14 complete | 330/479, 315,606 frames |
| Fixed A Random | epoch17 complete | 0/479 |
| Fixed A Utility | epoch17 complete | 0/479 |

The receipt counts are progress only, not performance metrics. The fixed A
epoch16 Miss queues completed and advanced to epoch17 complete (zero sequence
receipts at this snapshot); Batch10 epoch14 complete remains active. Preserve
all weights. Do not run val97.
Report Miss scores only after each official evaluation has 479 sequence
receipts and 337,353 frames and its aggregate result has been independently
checked.

Selecting the best checkpoint from test-set final-five scores has selection
bias. Report the requested best-of-five result with that limitation explicit;
do not present it as an unbiased estimate.
