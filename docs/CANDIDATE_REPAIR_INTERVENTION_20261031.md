# Candidate Repair Inference Intervention

## Question And Existing Evidence

Epoch5 full97 tracking has Utility below Random in complete and synthetic
outage PR/SR, with paired-video confidence intervals crossing zero. The
historical four-tuple selector audit and the frozen regression-versus-ranking
control did not show consistent selection improvement. These observations
do not identify whether the current compensation branch is useful, whether
four-region selection is a bottleneck, or whether gains are mostly in the
jointly trained trunk. A same-model intervention can narrow this question
without changing the running training or selecting on official test scores.

## Fixed Protocol

`audit_candidate_repair28.py` runs CPU-only with four threads.
PID101337 launched approximately17:18 UTC+08; launch receipt written17:18:05.
At17:19:07 it was alive and had completed actual first-batch interventions
and written `utility_progress.json`, not just loaded a model.

The pair uses epoch6/update12000 weights, which both campaigns retained.
The Random epoch5 checkpoint was already removed by rolling retention.
Epoch5 full97 tracking motivates the experiment, but these are different
weights. The checkpoint hashes are:

| Model | SHA256 |
|---|---|
| Utility epoch6 | ec9b4b087f2f44f71831ada45c3c34ee82efaad4d5344e048aaf9a7ce526a527 |
| Random epoch6 | 66fa8175a6f000e711da8577903eba386c2a8b03c94b504e68e7a82c5f4577ee |

Inputs reproduce the existing128 cached validation tuples exactly:
`13105d2512b095da153e0ec8cd38cf6264c0971db7d40b036ddfc10a71455391`.
Validation seed1701, batch4, two complete template frames, three search frames,
search states11/01/10, no patch masking. No optimizer or new model weight.
The same fixed generator seed2027 supplies the random control in each model.

For every model, tuple and search state, four arms reuse exactly the same
features immediately before layer6 compensation and the same context:

- `native4`: the checkpoint's native four-region selection.
- `off`: the compensation branch is omitted.
- `random4`: apply compensation to four uniformly sampled distinct regions.
- `all16`: apply compensation to all16 regions.

Native4 is re-executed through the tail; predicted boxes and score/size/offset
maps must match the original full forward bitwise on every search.
Original selector RNG buffer is restored before verifying the full model
state digest is unchanged. All intervention losses must be finite.
Raw per-sample losses, selected regions and execution witnesses are saved.
Shared runtime and running/queued GPU evaluation sources are unchanged.

## Analysis And Interpretation

`analyze_candidate_repair28.py` independently checks completion/report/source
and checkpoint hashes,32 batches of4 samples per model/state, loss shapes,
all random-subset draws, aggregate means and win/tie/loss counts.
It reports within-model native-minus-off, native-minus-random and
all16-minus-native loss, plus Utility-minus-Random under every arm.
The producer's bitwise/state witnesses are explicitly not re-executed by
the analyzer. No video-independent confidence intervals are computed from
these reused correlated tuples.

`finish_candidate_repair_audit28.py`, PID105825, waits for the actual producer
with300-second polling, checks the registered analyzer hash, runs analysis
only after a completed receipt, then writes `analysis_completed.json`.
There is no automatic restart or change to training/stopping.
Estimated full CPU runtime40-65minutes, provisional until measured progress.
Expected completion approximately18:00-18:25 UTC+08.
At17:21:40 the producer and queue105825 were both alive;16/128 Utility
samples had completed all interventions in185.28seconds, with every native
tail equality witness passing. Linear throughput projects roughly50minutes
for both models, so the next useful check is around18:05-18:15.

A benefit of all16 would be a higher-cost inference control, not evidence of
a paper contribution. Turning compensation off in a trained model is an
inference ablation, not retraining without the branch. Between-model effects
also include different trained trunk and correction parameters; they cannot
identify selection alone. These task losses are not tracking PR/SR or
official results, and do not establish an unseen validation set.
Final conclusions require the complete report and independent checks.

## Artifacts

Server output: `reports/server28-candidate-repair-audit-20261031`.
Producer log: `reports/server28-candidate-repair-audit-20261031.log`.
Queue log: `reports/server28-candidate-repair-audit-20261031-analysis.log`.
Keep producer, analyzer, queue controller and registered runtime sources
unchanged until analysis completes.

## Corrected Analyzer

The producer completed both models at18:04 UTC+08 in2760.96seconds.
The original queued analyzer failed because it assumed two search frames.
The hashed frozen YAML specifies `DATA.SEARCH.NUMBER:3` and
`DATA.TEMPLATE.NUMBER:2`; the loader uses those counts, and the producer's
actual loop processed all three searches, with three recorded region arrays
per model/batch/state. The launch protocol's phrase "both search frames"
was inaccurate. The original source, launch, report and failure log remain
unchanged; this is an analysis/documentation correction, not a model rerun.

`analyze_candidate_repair_corrected28.py` reads the hash-verified YAML,
requires its actual3search/2template configuration, checks all three searches
and all random-subset draws, and writes separate `analysis_corrected.json`
and `analysis_corrected_completed.json`. It records original analyzer and
failure-log hashes. Successful independent verification must be recorded
separately after this corrected analyzer actually completes.

## Verified Completed Result

The corrected analyzer actually exited0 at approximately18:07 UTC+08.
It verified the full128samples/model/state, all3searches, every random draw,
original source/checkpoint hashes and the loss summary arithmetic.
Original report SHA256:
`070f5f10f4147e6c036442d3cc7d80b4c3e93865b60e51a7103dcf215666d939`.
Corrected analysis SHA256:
`0631845222c32de202b59f1f054185111af4afc1e5f18b56dc627eaf1571fa9b`.
Corrected analyzer SHA256:
`939b1dc0ac66f87d85ad7c490143b7ddbf9de1fa45ec7d34f1d2f912c44f1eb8`.
All match downloaded files and the separate corrected-completion receipt.

| Model | Search availability | Native4 loss | Off loss | Random4 loss | All16 loss |
|---|---|---:|---:|---:|---:|
| Utility6 | Both11 | 4.393930 | 4.393824 | 4.394165 | 4.393871 |
| Utility6 | TIR only01 | 5.730527 | 5.730585 | 5.730559 | 5.730503 |
| Utility6 | RGB only10 | 4.782562 | 4.790027 | 4.789282 | 4.782573 |
| Random6 | Both11 | 4.250733 | 4.250471 | 4.252820 | 4.259709 |
| Random6 | TIR only01 | 5.757667 | 5.757446 | 5.757560 | 5.757996 |
| Random6 | RGB only10 | 4.880789 | 4.881932 | 4.880055 | 4.879584 |

Utility native-minus-off mean loss is+.000105/-.000058/-.007465 for
11/01/10. Its RGB-only relative mean improvement is approximately0.156%;
only62/128tuples improve, so the negative mean is not majority consistency.
Native-minus-random4 is-.000235/-.000032/-.006720; RGB-only has59wins/
69losses. All16-minus-native is-.000058/-.000024/+.000011. There is no
uniform benefit supporting more selected regions.

Random native-minus-off is+.000262/+.000221/-.001143; native-minus-random4
is-.002087/+.000107/+.000734. Even the Random model has state-dependent
effects, and all16 increases its complete-state mean loss by.008976.

Utility-minus-Random with native selection is+.143197/-.027140/-.098227;
with compensation disabled it is+.143354/-.026861/-.091905.
Much of the between-model loss difference remains in the off arm. This
does not identify its cause: the trained trunks and correction parameters
differ, and disabling compensation is not matched retraining.

These are reused cached-tuple inference losses with batch-coupled focal
terms. No video-independent interval, closed-loop PR/SR, missing-template
history, official attainment, or validated paper module follows. Preserve
the running training/LR/stopping/final-five protocol. The result does not
justify increasing selected regions, claiming selector superiority or
converting this CPU diagnostic into an official contribution.

Publication artifacts are `reports/candidate-repair-audit-20261031.json`,
`reports/candidate-repair-audit-completed-20261031.json`,
`reports/candidate-repair-analysis-corrected-20261031.json`,
`reports/candidate-repair-analysis-corrected-completed-20261031.json`, and
`reports/candidate-repair-original-analysis-failure-20261031.log`.
The failed original analyzer and queue are terminal; do not restart them.
