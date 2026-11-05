# Frozen-baseline adaptation control

Date: 2026-10-30. CPU-only experiment, SSH2028.

## Actual Change Tested

The existing LocalRepair at layer6 was trained while all baseline parameters
were frozen and the model stayed in eval mode. Only398466 repair parameters
entered AdamW (LR1e-4, weight decay1e-4, clipping0.1). This reuses an existing
branch as a control; it is not a new contribution or proof of novelty.

Baseline checkpoint:
d20c8641e4263df3ceadf9e430d0ac96c181c13ad482bca28c2dddef592a8bb8.
Cached8 train and8 val tuples, effective batch4,8 actual optimizer updates,
alternating search01/10. Each train tuple appears under both states.
Templates remain complete and there is no patch mask. CPU4threads,
PID3898469, completed228.64s. It has exited; no GPU resources used.

## Observed Result

| Split | Complete11 loss change | TIR-only01 change | RGB-only10 change |
|---|---:|---:|---:|
| Train8 | 0 | -0.02112556 | -0.00535750 |
| Val8 | 0 | -0.00284952 | +0.00021687 |

All complete score/size/offset maps and predicted-box tensors matched
bitwise before/after on these samples. Baseline parameter/buffer state
hash matched exactly, repair hash changed, and baseline gradients remained
absent. Finite losses and gradient norms were checked for all8 updates.

The existing residual gate is zero for fully available tokens. With the
baseline fixed this supports a structurally useful complete-input protection
control. Tested equivalence covers all11 template/search tuples. It does
not establish equivalence after a missing dynamic-template history or an
arbitrary modified tracking implementation.

Validation changes are tiny and disagree by modality. Eight updates on
eight samples cannot prove convergence, generalization, benchmark gain or
the requested two-point improvement. The small repair snapshot is retained
for reproducibility, not selected as a final model. Frozen formal source and
all four GPU training jobs were preserved.

## Utility Selection Scale

The existing4-tuple,16-region,3-state validation intervention audit was
also summarized without further training or threshold fitting:

| State | Median absolute benefit target | Median absolute selector prediction |
|---|---:|---:|
| 11 | 0.00015773 | 0.00166243 |
| 01 | 0.00037049 | 0.00243065 |
| 10 | 0.00014606 | 0.00190730 |

Selector MSE exceeds a zero-prediction baseline in all3 states on this
small set. Calibration error is not the same as failed ranking: affine
rescaling can change MSE without changing topK. This audit alone neither
explains the historical model outcome nor proves a ranking-loss benefit.

A potential ordinary baseline is within-tuple pairwise RankNet against
current absolute-benefit regression, with identical features, capacity,
counterfactual labels and train/development splits. allRank's implementation
filters ordered positive target differences and applies BCEWithLogits to
prediction differences; its unweighted form does not use target magnitude.
Exact ties receive no pairwise preference. Source and Apache-2.0 license
were read at pinned commit c88475661cb72db292d13283fdbc4f2ae6498ee4,
repository allegro/allRank, file allrank/models/losses/rankNet.py.
No source code copied and no external package installed.

No rank-loss hyperparameters, target normalizations or tolerance thresholds
were selected from validation/test results. Generic RankNet and existing
LocalRepair are controls, not the requested three validated innovations.

## Evidence

- reports/server28-protected-adaptation-20261030/report.json
- reports/server28-protected-adaptation-20261030/repair_after8.pth
- outputs/utility-scale-audit-20261030.json
- Source/checkpoint/input hashes and all8 update records in the report.

The full objective remains incomplete. GPU0/1 fixed campaign and GPU2/3
adaptive batch10 campaigns continue; first-epoch validation is still pending.
