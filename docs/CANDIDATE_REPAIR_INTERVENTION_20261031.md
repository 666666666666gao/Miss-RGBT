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
Validation seed1701, batch4, complete templates, both search frames,
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
