# Historical A utility20 official Miss evaluation

The historical `a_utility_full20_seed2027` checkpoint completed the official
paired Miss evaluation on 28 October 31, 2026. It covered all479 sequences
and337353 frames in12528.71seconds using one worker on each of GPUs0/1.
The evaluated checkpoint SHA256 is
`c4aabefb495344b6d5e6e17655ac20fd51c4e2dec575843133096b319e047db3`.
The standalone evaluator SHA256 is
`bc45da1293bb2e4735a3ac79cf7c7bb5fb195b7ef9069744b2ddd891eae307b1`.

| Dataset | Metric | E0 | Historical A utility20 | Difference | Requested threshold | Remaining |
|---|---:|---:|---:|---:|---:|---:|
| LasHeR-Miss | PR | 63.472507 | 68.565174 | +5.092667 | >69.3 | 0.734826 |
| LasHeR-Miss | SR | 50.583148 | 54.621376 | +4.038228 | >54.4 | passed by0.221376 |
| RGBT234-Miss | MPR | 82.794815 | 84.964353 | +2.169539 | >88.1 | 3.135647 |
| RGBT234-Miss | MSR | 59.574435 | 60.511286 | +0.936850 | >64.3 | 3.788714 |

The historic complete-input LasHeR/RGBT234 metrics and their paired
deltas were not recomputed in this job. Therefore this result does not
establish the requested complete-input gains of about2points. Three of
four Miss thresholds remain unmet; this is a baseline/control checkpoint,
not the newly trained batch10 candidate.

The six evidence files in `reports/historical-a-miss-20261031/` contain the
aggregate metrics, completion receipt, evaluator/source manifest, all
sequence receipts and both success curves. The result output remained on
server28 under
`/home/gaob/Miss-RGBT/results/server28_historical_a_20261030/miss_single_worker_per_gpu`.
