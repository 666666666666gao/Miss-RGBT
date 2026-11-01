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
| Fixed A Random, epoch18 | 85.840473 | 61.418089 |
| Fixed A Utility, epoch18 | 86.039255 | 61.454244 |
| Requested threshold | >88.1 | >64.3 |

The requested RGBT234-Miss thresholds are +5.305185 MPR and +4.725565 MSR
above the measured E0 baseline. Historical A adds +2.169538 and +0.936851;
Batch10 Utility epoch13 adds +2.108635 and +1.082522. Fixed A epoch16 adds
+1.951926/+1.037598 (Random) and +2.152128/+1.102817 (Utility). Batch10
Utility epoch14 adds +2.344994/+1.818622; Fixed A Utility epoch18 is now
highest at +3.244440 MPR/+1.879809 MSR. The requested threshold is not
equivalent to a two-point gain over E0.

## Best Audited Official Metrics So Far

These are per-metric maxima among completed, full-scope official evaluations
available so far. Different columns can come from different checkpoints; this
is not one model's combined score. The final-five evaluation is still running,
so later epochs can change these maxima.

| Condition | Metric | Best candidate / epoch | Candidate | Original E0 | Delta vs E0 |
|---|---|---|---:|---:|---:|
| Complete | LasHeR PR | Batch10 Utility / 13 | 75.459824 | 75.636503 | -0.176679 |
| Complete | LasHeR NPR | Batch10 Utility / 13 | 72.049205 | 71.772837 | +0.276368 |
| Complete | LasHeR SR | Fixed A Utility / 16 | 60.316126 | 60.125714 | +0.190412 |
| Complete | RGBT234 MPR | Fixed A Random / 16 | 91.043051 | 90.677134 | +0.365917 |
| Complete | RGBT234 MSR | Batch10 Random / 14 | 66.311785 | 66.845228 | -0.533443 |
| Miss | LasHeR PR | Batch10 Random / 15 | 69.568285 | 63.472507 | +6.095778 |
| Miss | LasHeR NPR | Batch10 Utility / 13 | 64.255180 | 58.581851 | +5.673330 |
| Miss | LasHeR SR | Batch10 Utility / 13 | 55.088625 | 50.583149 | +4.505476 |
| Miss | RGBT234 MPR | Fixed A Utility / 18 | 86.039255 | 82.794815 | +3.244440 |
| Miss | RGBT234 MSR | Fixed A Utility / 18 | 61.454244 | 59.574435 | +1.879809 |

Including E0 itself, E0 remains the overall complete-input maximum for LasHeR
PR/SR and RGBT234 MSR; current candidates lead on LasHeR NPR and RGBT234 MPR.
The current candidate Miss maxima still miss the requested RGBT234 thresholds
by 2.060745 MPR and 2.845756 MSR.

Fixed A Utility epoch17 was the previous highest verified RGBT234-Miss result.
Fixed A Utility epoch18 now leads both RGBT234-Miss metrics at 86.039255/
61.454244, still 2.060745/2.845756 below the target.
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

At 2026-11-01 21:39 CST, Fixed A epoch16 `complete` and `miss` finished for
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
| Batch10 Random | epoch16 miss | 4/479 |
| Batch10 Utility | epoch16 miss | 17/479 |
| Fixed A Random | epoch19 complete | 39/479 |
| Fixed A Utility | epoch19 complete | 18/479 |

At 22:42 CST, all four evaluator processes were still alive and advancing.
GPU0-3 utilization was 99-100%, temperature 75-83 C, and draw
249.01-249.54 W under the existing 250 W power caps. Batch10 Random epoch16
complete has passed full audit and its Miss evaluation is underway. Fixed A
Random and Utility epoch18 Miss have both passed full audits, and both arms
are evaluating epoch19 complete. No process was restarted, no new training
was launched, and all candidate weights remain preserved. Do not run val97.

Fixed A Random epoch18 `miss` passed full-scope audit: `condition=miss`,
`limit=null`, 479 receipts/337,353 frames; dataset counts are 245/220,703
LasHeR-Miss and 234/116,650 RGBT234-Miss. `run.json` and `completed.json`
agree; checkpoint SHA256 is
`929d6538ae350a5dcfb7198620c46c49fa77f752e4c177bd2fd5a2a300da7dc1`;
evaluator SHA256 is
`bc45da1293bb2e4735a3ac79cf7c7bb5fb195b7ef9069744b2ddd891eae307b1`.
All 479 individual receipt identities and hashes match `all_receipts.json`,
and all 479 bbox file hashes match. Metrics: LasHeR PR/NPR/SR
68.162673/63.150544/54.273160; RGBT234 MPR/MSR 85.840473/61.418089.

Fixed A Utility epoch18 `miss` also passed full-scope audit: `condition=miss`,
`limit=null`, 479 receipts/337,353 frames; dataset counts are 245/220,703
LasHeR-Miss and 234/116,650 RGBT234-Miss. `run.json` and `completed.json`
agree; checkpoint SHA256 is
`1f907ae2238d94a509b4924dcb051699ff97c90d50f75a2998c4d10cd234f492`;
evaluator SHA256 is
`bc45da1293bb2e4735a3ac79cf7c7bb5fb195b7ef9069744b2ddd891eae307b1`.
All 479 individual receipt identities and hashes match `all_receipts.json`,
and all 479 bbox file hashes match. Metrics: LasHeR PR/NPR/SR
67.730655/62.711185/53.942749; RGBT234 MPR/MSR 86.039255/61.454244.

Batch10 Random epoch16 `complete` passed full-scope audit:
`condition=complete`, `limit=null`, 479 receipts/337,353 frames, with
245/220,703 LasHeR and 234/116,650 RGBT234 frames. Run/completed records
agree; checkpoint SHA256 is
`16dc20c62d8464c884147aecaa732f5acbdbe5529daf1d55867297567a0da932`;
evaluator SHA256 is
`bc45da1293bb2e4735a3ac79cf7c7bb5fb195b7ef9069744b2ddd891eae307b1`.
All 479 individual receipt identities match `all_receipts.json`, and all
479 bbox hashes pass. Metrics are LasHeR PR/NPR/SR
74.541153/71.053097/59.463409 and RGBT234 MPR/MSR 89.963207/65.482712.
This does not change the current per-metric complete maxima above.

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

At 23:00 CST, the same four official queues were still live. Batch10 Random
and Utility epoch16 Miss had 21/479 and 56/479 sequence receipts; Fixed A
Random and Utility epoch19 complete had 83/479 and 48/479. These are progress
counts, not metrics, and no new full-scope result was available. GPU0-3 were
82/80/78/75 C at 249.10/248.88/248.89/249.39 W under 250 W limits. No
training or val97 evaluation was started; all weights remain preserved.

## Newly audited final-five entries at 23:55 CST

| Arm / epoch | Condition | LasHeR PR/NPR/SR | RGBT234 MPR/MSR |
|---|---|---:|---:|
| Fixed A Random / 19 | Complete | 73.398075 / 69.879029 / 58.671485 | 90.159512 / 65.761226 |
| Batch10 Utility / 16 | Miss | 69.541995 / 64.221692 / 55.006010 | 84.979303 / 60.632103 |

Both runs have `limit=null`, 479 sequences and 337,353 frames, matching
`run.json` and `completed.json`. Dataset totals are 245/220,703 LasHeR and
234/116,650 RGBT234. Each of 479 individual receipts matches the aggregate
receipt, every bbox hash matches, checkpoint/evaluator hashes match, and all
110 pinned runtime source-file hashes match.

Neither result changes the per-metric maxima above. Batch10 Utility epoch16
Miss clears the LasHeR PR/SR thresholds (69.3/54.4), but RGBT234 remains
3.120697 MPR and 3.667897 MSR below 88.1/64.3. Selection among final-five
test results remains post-hoc and test-selection-biased.

At 23:59 CST, remaining active evaluations were Batch10 Random epoch16 Miss
290/479, Batch10 Utility epoch17 complete 0/479, Fixed A Random epoch19
Miss 0/479, and Fixed A Utility epoch19 complete 346/479. GPUs were
75-82 C and 249.19-249.61 W under 250 W limits. No training or val97 run
was started; all checkpoints remain.
