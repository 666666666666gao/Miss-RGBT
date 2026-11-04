# Research Decision - 2026-11-04

## Objective

The primary result must be one model/checkpoint whose full and modality-missing evaluations all use the registered complete benchmark scopes. The requested Miss thresholds are LasHeR PR/SR >69.3/54.4 and RGBT234 MPR/MSR >88.1/64.3. A roughly two-point gain over E0 is not sufficient for every metric: E0's full RGBT234 MPR/MSR are 90.677/66.845, so the corresponding SCDT targets are about 2.423/2.755 points higher.

The SCDT comparison values are transcribed from the [CVPR 2026 paper](https://openaccess.thecvf.com/content/CVPR2026/html/Lu_Spatio-Temporal_Conditional_Denoising_Transformer_for_Modality-Missing_RGBT_Tracking_CVPR_2026_paper.html). Exact scoring-code parity with SCDT has not been independently established, so these are comparison targets, not a claim of a controlled head-to-head reproduction.

## Audited Evidence

The full-scope paired table in `PAIRED_OFFICIAL_METRICS_20261102.md` reports Batch10 Utility epoch17 as one same-checkpoint candidate:

| Condition | LasHeR PR/NPR/SR | RGBT234 MPR/MSR |
|---|---:|---:|
| Complete | 75.751492 / 71.965679 / 60.195400 | 89.839668 / 65.314284 |
| Miss | 70.397396 / 65.127562 / 55.789882 | 85.097378 / 60.685934 |

Against the requested Miss thresholds, that checkpoint clears LasHeR PR/SR by 1.097396/1.389882 points, but misses RGBT234 MPR/MSR by 3.002622/3.614066 points. Its complete RGBT234 MPR/MSR also trail the SCDT table by 3.260332/4.285716 points. The complete-to-Miss drops for this checkpoint are 4.742290 MPR and 4.628350 MSR, versus 5.0/5.3 in the SCDT table. This descriptive comparison suggests that absolute complete-condition localization is also a material gap; it does not identify a cause.

Do not combine per-metric maxima from different checkpoints into a single model result. The currently running A final-five queue is still evaluating Utility epoch15 and Random epoch16, so its final same-checkpoint comparisons and queue-level verification remain pending.

## Protocol Finding

The current six-arm RPPM campaign is frozen and must not be modified. Source inspection found:

- `research/curriculum_p/pacing.py` assigns a lower missing probability to samples marked hard by the complete-view loss EMA.
- `research/curriculum_p/schedule.py` defines `shuffled` by permuting the epoch-level sequence of 0.2/0.5/0.8 probabilities. It does not permute per-sample probability assignments within a batch.

Therefore, `shuffled` tests schedule order, not whether associating difficulty with the same sample is useful. If the registered campaign motivates a sample-conditioned pacing claim, a separately preregistered follow-up should compare `paced` with a per-batch assignment-shuffled control. The control must preserve each batch's probability multiset and mean, paired data order, initialization, optimization, and evaluation scope; only the mapping from samples to probabilities should change. This is an attribution gap, not evidence that the existing implementation is faulty.

## Test-Conditioned Continuation Risk

The frozen RPPM preparation gate also decides whether P proceeds by checking
the R1-selected checkpoint against E0 minus0.1pp on all five Complete
official metrics (`docs/RPPM_PROTOCOL_ENTRY_20261103.md`). Since these are
official benchmark evaluations, this uses test outcomes to choose the next
research branch, not only to report a final score. Preregistration makes the
decision reproducible but does not make the test set an independent
development set.

Do not alter this already-running R/P protocol or reinterpret its gate
post hoc. Preserve and report the registered outcome with this limitation.
For future method, curriculum, and continuation decisions, use a development
split and freeze the decision rule before official evaluation; reserve the
official test for final reporting.

## Module Evidence and Order

1. **RPPM:** The registered six-arm queue is `WAITING_FOR_R` until both R arms have training verification and full-official receipts. Its real-image smokes, formal runs, and full evaluations are not yet complete. Do not start the proposed matched control or duplicate the queue before the existing protocol reaches its terminal gates.
2. **RCFR:** The five-arm design includes `single`, `none`, `shallow`, `deep-no-response`, and `deep`, which separates the deep-feature and response-conditioning contrasts. The staged CPU checks are contract evidence only; no real-image smoke, formal run, complete benchmark result, or efficacy claim is established.
3. **OHCAS:** Keep gated. The residual prototype is not integrated into `MissingTrackerModel`; the online history crop/feature, coordinate, and age contract is unresolved, and historical R-frame exclusion provenance has not passed its data gate.

Any formal model result must report all four conditions: LasHeR Complete/Miss at 245 sequences and 220,703 frames each, and RGBT234 Complete/Miss at 234 sequences and 116,650 frames each. Keep every checkpoint and select weights only by the preregistered development rule, not official-test scores.

## Live Gate Snapshot

Read-only Server 28 observation at 2026-11-04 01:22 UTC:

- A official queue PID 1355742 and arm controllers 1601932/1601933 were alive. Its Utility epoch15 Complete and Random epoch16 Complete evaluators were still running; A queue terminal receipts and queue-level formal verification were absent.
- A training completion receipts were present for Utility (17 epochs/34,000 updates) and Random (18 epochs/36,000 updates); training-scope `verification_v2.json` was `passed`. This does not satisfy the separate official-queue verification gate.
- R1/R2 trainers 3388846/3395457 remained alive at epoch16/update37625 and epoch15/update18425.
- RPPM queue PID 3539938 remained `WAITING_FOR_R` on the four R training-verification and full-official receipts.
- GPU0-3 were at 77/77/76/72 C and 247.43/246.01/249.49/248.97 W under 250 W caps, with 91/100/100/100% utilization. All cards had active assignments.

No process, configuration, manifest, checkpoint, or queue was changed; no additional GPU task was dispatched.
