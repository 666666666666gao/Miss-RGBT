# Task-utility selector audit

Date: 2026-10-30.

## Question

Does the trained A utility selector rank 4x4 search regions by actual
single-region reduction in task loss across complete and missing inputs?

## Fixed Audit

- Historical A utility epoch20; checkpoint SHA256
  `c4aabefb495344b6d5e6e17655ac20fd51c4e2dec575843133096b319e047db3`.
- Four tuples from the native LasHeR val loader, seed2027; first search only.
- Complete templates, no patch dropout, model in eval mode.
- Sixteen single-region interventions per tuple in states11,01,10:
  192 total interventions.
- Native top4 compared with seeded random top4 and a diagnostic oracle
  selecting four regions with the best individual single-region gains.
- CPU4 threads. Not GPU-equivalence evidence.

## Results

Normalized reduction is `(loss_off - loss_selected) / loss_off`; positive
means lower task loss. Entries are means over only four tuples per state.

| Search availability | Spearman predicted logit vs. single-region gain | Mean top4 overlap | A native top4 reduction | Random top4 reduction | Offline-oracle top4 reduction |
|---|---:|---:|---:|---:|---:|
| 11 complete | -0.058 | 1.00/4 | -0.020% | +0.094% | +0.326% |
| 01 RGB missing | -0.136 | 1.25/4 | +0.082% | +0.110% | +0.384% |
| 10 TIR missing | +0.456 | 2.50/4 | +0.356% | +0.156% | +0.554% |

Mean per-tuple Spearman SD was0.272/0.593/0.370 for11/01/10. Between4 and
12 of16 regions had positive individual target per tuple. Mean actual best
single-region gains were0.178%/0.214%/0.231%; mean predicted-best-region
gains were-0.001%/-0.028%/+0.104%.

The 11/01 native reductions are near zero, with cross-tuple SD0.120/0.121
percentage points. In10, SD0.408pp exceeds the0.356% mean. Therefore the
apparent native-over-random advantage in10 is not reliable at n=4.

## Interpretation

There is a preliminary state-specific ranking signal for10. It is absent in
this small audit for11 and01. Do not describe A's selector as generally
task-optimal or a validated contribution based on these results.

This does not demonstrate a training implementation defect: n=4/state is
insufficient, original E0 exposure to val97 is unverified, and scores may
vary with seed, augmentation and training mode. Only first-search,
eval-mode, no-patch-dropout behavior was measured. Oracle joint top4 is a
diagnostic upper bound because individual region effects need not add.

Do not change the live training or tune thresholds on these data. Next
evidence: predeclared larger train882 ranking/calibration, stratified by
11/01/10, with utility/random/similarity/all at equal K and capacity; then
an independent val97 audit if its exposure boundary can be established.
RGBT234 remains test-only without a certified development split.

Artifacts:

- `server28-utility-ranking-20261030/report.json`
- `server28-utility-ranking-analysis-20261030.json`
- `server28-utility-ranking-20261030/completed.json`
- `server28-utility-ranking-20261030.log`
