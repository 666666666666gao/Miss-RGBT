# Partitioned A epoch 4 development snapshot

Observed on server28 at 2026-11-02 23:27 CST. This is the fixed 128-sample,
train-only development set, evaluated separately for complete (11), RGB
missing (01), and TIR missing (10). These losses are diagnostic only; they
are not official PR/SR results, convergence evidence, or a basis for
test-set-based model selection.

| Arm | Epoch | Complete (11) | RGB missing (01) | TIR missing (10) | Mean |
|---|---:|---:|---:|---:|---:|
| Utility | 1 | 3.654450 | 5.315351 | 4.099837 | 4.356546 |
| Utility | 2 | 3.698152 | 5.364660 | 4.132909 | 4.398573 |
| Utility | 3 | 3.808033 | 5.488136 | 4.147695 | 4.481288 |
| Utility | 4 | 3.741152 | 5.551541 | 4.186196 | 4.492963 |
| Random | 1 | 3.579864 | 5.275305 | 4.075708 | 4.310292 |
| Random | 2 | 3.717756 | 5.357402 | 4.122523 | 4.399227 |
| Random | 3 | 3.866955 | 5.412043 | 4.137329 | 4.472109 |
| Random | 4 | 3.709107 | 5.334135 | 4.117464 | 4.386902 |

From epoch 1 to 4, both arms' complete and TIR-missing losses remain higher.
Utility RGB-missing loss also increased; Random RGB-missing loss at epoch 4
is close to, but still above, epoch 1. The uneven per-condition trajectory
does not justify changing the preregistered LR or stopping schedule.

Raw snapshots:

- `reports/server28_partitioned_a_dev_history_epoch04_utility_20261102.json`
- `reports/server28_partitioned_a_dev_history_epoch04_random_20261102.json`

At 23:27 CST both formal trainers and the registered official queue were
still live. Utility was at epoch 5, update 8100; Random at epoch 5, update
8150, with 2000 updates per epoch. GPU2/3 were 75/74 C, below the 85 C
target, with 250 W power limits. The next useful check is near the estimated
epoch 5 completion around 00:59 CST; use the actual epoch boundary rather
than this estimate. Do not inspect growing manifests or start P smoke until
both runs terminate and the official queue's independent verification
receipt passes.
