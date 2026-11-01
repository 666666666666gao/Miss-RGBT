# RGBT234-Miss Gap Analysis

## Official Full-Miss Results

All entries below are complete official evaluations. Each result covers 479
sequences and 337,353 frames across LasHeR and RGBT234.

| Model | RGBT234-Miss MPR | RGBT234-Miss MSR |
|---|---:|---:|
| Original CADTrack E0, epoch20 | 82.794815 | 59.574435 |
| Historical A Utility, epoch20 | 84.964353 | 60.511286 |
| Batch10 Utility, epoch13 | 84.903450 | 60.656957 |
| Batch10 Utility, epoch14 | 85.139809 | 61.393057 |
| Batch10 Utility, epoch15 | 84.877660 | 60.116616 |
| Batch10 Random, epoch15 | 85.562010 | 60.642597 |
| Fixed A Random, epoch16 | 84.746741 | 60.612033 |
| Fixed A Utility, epoch16 | 84.946943 | 60.677252 |
| Fixed A Random, epoch17 | 85.328042 | 60.523251 |
| Fixed A Utility, epoch17 | 85.649309 | 60.685303 |
| Requested threshold | >88.1 | >64.3 |

The requested RGBT234-Miss thresholds are +5.305185 MPR and +4.725565 MSR
above the measured E0 baseline. Historical A adds +2.169538 and +0.936851;
Batch10 Utility epoch13 adds +2.108635 and +1.082522. Fixed A epoch16 adds
+1.951926/+1.037598 (Random) and +2.152128/+1.102817 (Utility). Batch10
Utility epoch14 adds +2.344994/+1.818622; Fixed A Utility epoch17 is now
highest at +2.854494/+1.110868. The requested threshold is not equivalent
to a two-point gain over E0.

Fixed A Utility epoch17 is now the highest verified RGBT234-Miss result in
this table at 85.649309/60.685303, still 2.450691/3.614697 below the target.
Against epoch16 Utility, RGBT234 MPR/MSR improve by 0.702366/0.008051 and
LasHeR PR/SR by 0.468011/0.434108. Fixed A Random epoch17 improves its
RGBT234 MPR by 0.581301 but decreases MSR by 0.088782 against its epoch16.
These candidates show modest gains and metric tradeoffs, not a dataset-
specific cause.

Batch10 Utility epoch13 is nearly tied with historical A: -0.060903 MPR and
+0.145671 MSR. Fixed A epoch16 is also near the historical result (-0.017410
MPR, +0.165966 MSR for Utility). Epoch16 is only the first of five official
checkpoints in this final-five campaign; no final model selection is
justified yet.

Batch10 Utility epoch14 Miss passed the full official audit: `condition=miss`,
`limit=null`, 479 receipts and 337,353 frames matching `run.json` and
`completed.json`; per-dataset scope is 245/220,703 LasHeR-Miss and
234/116,650 RGBT234-Miss; checkpoint and evaluator hashes match the run
manifest. Metrics are LasHeR PR/NPR/SR 69.418287/64.214523/55.036005 and
RGBT234 MPR/MSR 85.139809/61.393057. It clears both LasHeR targets but misses
both RGBT234 targets by 2.960191/2.906943; it is not the final-five
selection. Relative to Batch10 Utility epoch13, changes are +0.165615 PR,
-0.052619 SR, +0.236358 MPR and +0.736100 MSR.

Batch10 Utility epoch15 `miss` passed independent full-scope audit:
`condition=miss`, `limit=null`, 479 receipts and 337,353 frames; dataset
counts are 245/220,703 LasHeR-Miss and 234/116,650 RGBT234-Miss.
`completed.json`, checkpoint/evaluator hashes, and all 479 bbox receipt
hashes match. Metrics are LasHeR PR/NPR/SR 69.486548/64.192847/54.973660
and RGBT234 MPR/MSR 84.877660/60.116616. Versus epoch14, LasHeR PR/SR
change +0.068261/-0.062346 while RGBT234 MPR/MSR decrease
0.262149/1.276440. This is an observed candidate tradeoff, not a causal
explanation. It clears the two LasHeR thresholds but not either RGBT234
threshold; final-five selection is pending.

The paired complete-input results show substantial Miss-condition drops on
both benchmarks. From complete to Miss, Fixed A Utility drops 7.21/6.07 points
on LasHeR PR/SR and 5.90/5.26 on RGBT234 MPR/MSR; Random drops 6.43/5.44 and
6.30/5.68, respectively. Because PR/SR and MPR/MSR use different benchmark
aggregations, these numeric differences cannot establish that missing input
is more or less damaging on one dataset. They do not isolate a cause for the
target gap. What is established is that RGBT234-Miss remains around
84.75-85.65/60.12-61.39, below the requested 88.1/64.3.

The earlier paired Batch10 Utility epoch13 run shows the same pattern:
RGBT234 changes from 89.677569/65.372151 complete to 84.903450/60.656957
Miss (-4.774118/-4.715194); LasHeR changes from 75.459824/60.179363 to
69.252673/55.088625 (-6.207151/-5.090738). This confirms a substantial
missing-condition loss in that model, but the cross-benchmark differences
still do not identify a RGBT-specific cause.

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

At 2026-11-01 20:57 CST, Fixed A epoch16 `complete` and `miss` finished for
both arms. Each of the four runs passed independent full-scope checks:
479 receipts, 337,353 summed frames, matching `completed.json` and `run.json`;
the two Miss runs also match per-dataset receipt/frame counts, and checkpoint
and evaluator hashes match their manifests.

| Arm | LasHeR PR/SR | RGBT234 MPR/MSR |
|---|---:|---:|
| Fixed A Random epoch16 complete | 74.490607 / 59.572963 | 91.043051 / 66.287325 |
| Fixed A Utility epoch16 complete | 75.425079 / 60.316126 | 90.847494 / 65.941674 |

Batch10 Random epoch15 `miss` passed independent full-scope audit:
`condition=miss`, `limit=null`, 479 receipts/337,353 frames, dataset counts
245/220,703 and 234/116,650, completion record agrees, checkpoint/evaluator
hashes match, and all bbox receipt hashes match. Metrics are LasHeR
PR/NPR/SR 69.568285/64.141402/55.071246 and RGBT234 MPR/MSR
85.562010/60.642597. Against Batch10 Utility epoch15, Random is lower on
LasHeR PR/SR by 0.082263/0.097587 but higher on RGBT234 MPR/MSR by
0.684350/0.525981. This is an observed arm tradeoff, not a cause diagnosis.
It clears both LasHeR thresholds but neither RGBT234 threshold; final-five
selection is pending.

Batch10 Utility epoch14 `complete` also finished and passed the same
479-sequence/337,353-frame receipt check, with checkpoint and evaluator
hashes matching its manifest. Its complete-input metrics are LasHeR
75.024540/59.742296 PR/SR and RGBT234 89.747434/65.849284 MPR/MSR. Its paired
epoch14 Miss result is reported in the full-Miss table above.

Batch10 Random epoch14 `complete` independently passed the same full-scope
and hash checks. Its complete-input metrics are LasHeR PR/SR
75.229949/59.903038 and RGBT234 MPR/MSR 90.279097/66.311785. Both Batch10
complete-input RGBT234 results exceed the numeric Miss targets, but complete
input is a different condition and cannot count as meeting those targets.

The paired full Miss metrics are LasHeR PR/SR 68.065197/54.134896 and
RGBT234 MPR/MSR 84.746741/60.612033 for Random; Utility is
68.217157/54.247395 and 84.946943/60.677252. These remain intermediate
epoch16 results, not the best-of-five result.

The four queues then advanced to:

| Queue | Current evaluation | Sequence receipts |
|---|---|---:|
| Batch10 Random | epoch16 complete | 14/479 |
| Batch10 Utility | epoch16 complete | 31/479 |
| Fixed A Random | epoch18 miss | 48/479 |
| Fixed A Utility | epoch18 miss | 25/479 |

Fixed A Random epoch17 `miss` passed independent full-scope audit:
`condition=miss`, `limit=null`, 479 sequence receipts and 337,353 frames;
dataset counts are 245/220,703 for LasHeR-Miss and 234/116,650 for
RGBT234-Miss. `completed.json` matches the run, checkpoint and evaluator
SHA256 values match the manifest, and every bbox file hash matches its
receipt. Metrics are LasHeR PR/NPR/SR 68.729255/63.438200/54.635222 and
RGBT234 MPR/MSR 85.328042/60.523251.

Fixed A Utility epoch17 `miss` also passed independent full-scope audit:
`condition=miss`, `limit=null`, 479 receipts and 337,353 frames; per-dataset
counts match, `completed.json` agrees, checkpoint and evaluator SHA256 values
match, and all 479 bbox hashes match their receipts. Metrics are LasHeR
PR/NPR/SR 68.685168/63.539463/54.681503 and RGBT234 MPR/MSR
85.649309/60.685303. Queue receipt counts are progress only, not performance
metrics. Preserve all weights. Do not run val97.
Report Miss scores only after each official evaluation has 479 sequence
receipts and 337,353 frames and its aggregate result has been independently
checked.

Selecting the best checkpoint from test-set final-five scores has selection
bias. Report the requested best-of-five result with that limitation explicit;
do not present it as an unbiased estimate.
