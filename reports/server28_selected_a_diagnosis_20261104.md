# Partitioned A: Selected-Checkpoint Diagnosis

Read-only analysis on 2026-11-04, approximately 08:45 UTC. This compares
already completed full evaluations; no evaluator or training job was launched.
The entire official queue has not completed its final verification.

## Scope And Selection

- Utility epoch14 and Random epoch15 are the campaign's development-selected
  checkpoints. Each row uses the same checkpoint for Complete and Miss.
- The arm controllers' verified_metrics.json files mark all four selected
  evaluations independently_recomputed=true. No official-test reselection occurs.
- Each condition covers LasHeR245/220703 frames and RGBT234234/116650 frames:
  479 sequences/337353 frames per evaluation,958/674706 per checkpoint pair.
- E0 is the historical baseline, not a matched initialization/protocol control.
  Utility/Random also stopped at different epochs. These observations do not
  identify the causal contribution of an individual module or prove multi-seed
  repeatability. The final epoch in each arm remains under evaluation.

## Full Metrics

Scores are percentages; deltas are candidate minus E0 in percentage points.

| Dataset | Metric | E0 | Utility14 | Delta pp | Random15 | Delta pp |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| LasHeR | PR | 75.636503 | 74.158724 | -1.477779 | 73.602038 | -2.034465 |
| LasHeR | NPR | 71.772837 | 70.542595 | -1.230242 | 70.002574 | -1.770263 |
| LasHeR | SR | 60.125714 | 59.154405 | -0.971309 | 58.640863 | -1.484851 |
| RGBT234 | MPR | 90.677134 | 87.473683 | -3.203451 | 88.414559 | -2.262575 |
| RGBT234 | MSR | 66.845228 | 63.972660 | -2.872568 | 64.548808 | -2.296421 |
| LasHeR-Miss | PR | 63.472507 | 67.174178 | +3.701671 | 67.890547 | +4.418040 |
| LasHeR-Miss | NPR | 58.581851 | 61.863332 | +3.281482 | 62.761895 | +4.180044 |
| LasHeR-Miss | SR | 50.583149 | 53.313374 | +2.730226 | 53.760802 | +3.177653 |
| RGBT234-Miss | MPR | 82.794815 | 84.735751 | +1.940936 | 83.890983 | +1.096169 |
| RGBT234-Miss | MSR | 59.574435 | 60.410309 | +0.835874 | 59.846600 | +0.272165 |

## Remaining Target Gaps

Positive gaps below are threshold minus result. Strictly exceeding each target
requires more than the listed gap; rounding to the threshold is insufficient.
Relative gains are100*(candidate-E0)/E0, not percentage-point gains.

| Metric | Threshold | Utility gap pp | Random gap pp | Utility relative gain % | Random relative gain % |
| --- | ---: | ---: | ---: | ---: | ---: |
| LasHeR-Miss PR | 69.3 | 2.125822 | 1.409453 | 5.8319 | 6.9606 |
| LasHeR-Miss SR | 54.4 | 1.086626 | 0.639198 | 5.3975 | 6.2820 |
| RGBT234-Miss MPR | 88.1 | 3.364249 | 4.209017 | 2.3443 | 1.3240 |
| RGBT234-Miss MSR | 64.3 | 3.889691 | 4.453400 | 1.4031 | 0.4568 |

## Interpretation And Next Steps

1. Both selected models lose all five Complete metrics relative to historical
   E0 while improving Miss metrics. Missing-input training is not the only
   unresolved issue: Complete localization remains a priority. This pattern
   does not establish whether initialization, optimization, sampling, or the
   modules caused the loss. Preserve the running R protocol experiments.
2. Utility improves RGBT234-Miss over Random by0.844767 MPR/0.563708 MSR,
   but trails Random on LasHeR-Miss by0.716369 PR/0.447427 SR. There is no
   uniform selector win across datasets. Do not splice the two models' peaks
   into one result or choose a new winner from official scores.
3. Wait for A's final queue verification and R's complete fixed20-round runs
   and full official evidence. Preserve the registered gates; matched Complete
   controls and module comparisons must follow the accepted plan. The present
   comparison cannot justify stronger masking or an unregistered R restart.
4. RPPM, RCFR, and OHCAS still require their registered acceptance, integration,
   image-smoke, full-evaluation, and reproducibility evidence before they can
   support the requested three paper contributions. No efficacy claim is made.

## Evidence

Server root: /home/gaob/Miss-RGBT.

- Baselines: runtime/task_utility_20261018/reports/{e0,e0_complete}/metrics.json.
  Local reports/{e0,e0_complete}/metrics.json contain the same metric values.
- Selected and verification flags:
  reports/server28_partitioned_a_20261102_official_queue_v2/{utility,random}_verified_metrics.json.
- Results: results/server28_partitioned_a_20261102/ followed by the names below.
  Checkpoint hashes were cross-checked between the selected Complete/Miss entries:
  Utility14=114571257d5f3cd34d954c6a2aab83607ad09f1c241f58eebfb0514603bec2f0;
  Random15=d9a0e88a1f88c786da33b11c85dbc63d1995a41b5746b28320a1a78430b8d29f.

| Result directory | completed.json SHA256 | metrics.json SHA256 |
| --- | --- | --- |
| utility_epoch14_complete | df0fa9055361d505b72963ee80f3bf7bc79c8c10fd2851102572bdd7b458bcf6 | 1c695d9beeb40752f419e3cfdd0883ab0b6b57f99089d1923d8db51107841079 |
| utility_epoch14_miss | f5a2a5305c3027e48e409ab3417013833f919e5efa3c25738cd7be6e38a41af4 | 6d00efb6b36aa7be3ceb40c49a493a0b45ec6c644aee87da85edf3ee0e7ba2d7 |
| random_epoch15_complete | 56d095cd62459b500f340b07cdeb69ea5063aabd1d947ff15a5f9b08d9b09f4f | dcde0dd4de995677e36f05ed8be80b64b8301ea831a867120b2e84c2e5c70932 |
| random_epoch15_miss | a9406732c155a89c7e779a9750a6e60ef183f275c360533deddccfef60ce3205 | cb8bb6f930bbc638c0071053f6d453d2b807dc8da082d598296f8b1d66ee322b |
