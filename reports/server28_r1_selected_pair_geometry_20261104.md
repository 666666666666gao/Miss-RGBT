# R1 Selected Pair: Full-Trajectory Geometry Diagnosis

## Scope And Denominators

CPU-only post-hoc analysis of the already verified R1 epoch16 Complete/Miss
pair, checkpoint0fe0af921bfc4ce2eba1301d3bc6f37a7adf68ea32a93542ecfc822dbcb8e03b.
All479 sequences/337353 frames per condition were inspected:958 prediction
files and713 GT files. Source receipts, inventory GT identities and every
prediction hash were checked. The two conditions' four terminal evidence
hashes each still match their prior full verification reports.

No state, difficulty or sequence subset was selected. Descriptive statistics
exclude the first frame of every sequence and annotation rows with any
nonpositive coordinate, following the existing localization diagnostic.
This coordinate-positive policy is a diagnostic convention, not a claim
that every excluded border-coordinate annotation is invalid for official
scoring. LasHeR excludes245 first frames and1259 other rows; RGBT234 visible
excludes234/596 and infrared234/633, respectively. Each denominator
reconciles with the full inventory. Complete/Miss use exactly the same
retained annotation rows.

RGBT234's visible and infrared GT are reported separately on the same234
sequences. They must not be summed as additional scenes or combined by
choosing the more favorable sensor. Continuous boxes and frame-weighted
means differ from the installed sequence-averaged benchmark protocol.
These are not new PR/NPR/SR/MPR/MSR scores. The ten official-protocol metrics
and strict targets remain in server28_r1_selected_pair_analysis_20261104.md.

## Full-Inventory Error Statistics

Normalized center error divides pixel error by sqrt(GT width times height).

| Dataset / GT | Condition | Retained rows | Median center px | P90 center px | P90 normalized center | Mean IoU % |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| LasHeR visible | Complete | 219199 | 8.055950 | 228.246065 | 2.994133 | 54.789048 |
| LasHeR visible | Miss | 219199 | 11.200025 | 248.040620 | 3.523394 | 49.759499 |
| RGBT234 visible | Complete | 115820 | 3.271830 | 24.189843 | 0.566751 | 65.913409 |
| RGBT234 visible | Miss | 115820 | 4.302911 | 72.025204 | 1.756136 | 59.414217 |
| RGBT234 infrared | Complete | 115783 | 4.775391 | 24.623665 | 0.643260 | 61.408792 |
| RGBT234 infrared | Miss | 115783 | 5.421041 | 71.714206 | 1.747585 | 57.157911 |

RGBT234's median changes are modest compared with its upper-tail changes.
The pixel P90 rises from about24 to72 under both annotations, while the
normalized P90 also rises substantially. Thus the observed Miss error
distribution has a much heavier localization-error tail. This is not a
temporal diagnosis: it does not locate the onset, identify particular
missing states, prove a history failure, or establish a causal mechanism.
LasHeR already has a large Complete P90, so its baseline localization and
the incremental missing-condition effect both need consideration.

## Geometric Oracle Contrasts

Recentered boxes keep each predicted width/height and move their centers
to GT centers. Resized boxes keep predicted centers and replace width/height
with GT dimensions. Both use unavailable GT. They are diagnostic geometric
contrasts, not implementable trackers, module forecasts or additive
contributions to total error. In particular, a resize can hurt an individual
displaced box, and the two contrasts do not partition observed failures.

| Dataset / GT | Condition | Original mean IoU % | Recentered mean IoU % | Resized mean IoU % | Center contrast pp | Size contrast pp |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| LasHeR visible | Complete | 54.789048 | 74.064845 | 58.329262 | 19.275797 | 3.540214 |
| LasHeR visible | Miss | 49.759499 | 70.691309 | 52.920538 | 20.931810 | 3.161039 |
| RGBT234 visible | Complete | 65.913409 | 75.382390 | 71.638264 | 9.468981 | 5.724855 |
| RGBT234 visible | Miss | 59.414217 | 72.352293 | 64.090613 | 12.938076 | 4.676395 |
| RGBT234 infrared | Complete | 61.408792 | 75.349899 | 65.602089 | 13.941108 | 4.193297 |
| RGBT234 infrared | Miss | 57.157911 | 72.333411 | 61.167960 | 15.175500 | 4.010048 |

The larger center contrast makes center localization a useful diagnostic
endpoint alongside box shape. It does not prove that RCFR, stronger masks,
a different LR or OHCAS will recover that contrast. Any proposed mechanism
must be tested through its registered development controls and full paired
reporting. These benchmark-exposed observations must not select weights,
hyperparameters or a replacement research gate. The frozen R/P outcome,
R3 objective comparison, review requirement and paired-smoke prerequisites
remain unchanged. No GPU experiment or extra validation was launched.

## Implementation And Checks

New analysis/diagnose_selected_pair_geometry.py reuses the existing
analysis/analyze_paired_baseline.py continuous-box implementation. It gates
on closed full results and identical checkpoint identities. Unlike the
older E0/E3/E4 diagnostic, it has no state/sequence grouping, per-sequence
rankings or alternate official scoring. No neural model, image or weight
tensor is loaded.

Five CPU unittest methods passed in0.005s on the existing Server28
environment: identical boxes, pure translation, pure size error, full
terminal-receipt gating, and rejection of a mismatched checkpoint/partial
scope. Fixtures are synthetic and are not benchmark evidence. Actual
full-data execution then produced six rows. All six denominators and
oracle arithmetic were separately checked after copying the JSON locally.
No fresh-agent semantic review was performed; these checks do not satisfy
R3's separate independent-review gate.

- Staging root:/tmp/codex_r1_pair_geometry_20261104.jjw3Hn.
- Analysis source SHA256:93e3fbb003c5f1207dd827133530dabe4e94452761e8160f74274e34685697b7.
- Reused helper SHA256:e9a28f43fed4546e3177fdfff74de8c4010f932dec9c8e2920a6b777747e2260.
- Raw report:reports/server28_r1_selected_pair_geometry_20261104.json.
- Local/server report SHA256:3b62729ce37bcde6b8d20c7bbb182eef15057be3488a7561fd112c8196a0348f.

At13:17:11UTC actual ps confirmed R1 epoch17 Complete2397874, R2 epoch16
Miss2358418, both R controllers and P3539938 alive. GPU0-3 temperatures
77/76/37/35C and power249.48/249.12/23.27/24.99W. All checkpoints retained.
