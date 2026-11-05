# Frozen Selector Objective Control

## Completed Result

Completed in452.37s with48 train,12 development and12 heldout slates.
Regression stopped45 and selected45 among41..45; pairwise stopped30 and
selected26 among26..30. Each retained its development-selected checkpoint;
the other4 final weights were removed with receipts. Exact native, random,
single-region oracle and off-loss equality between controls passed.

| Joint intervention | Mean task loss, lower is better | Mean per-tuple relative benefit vs off |
|---|---:|---:|
| Historical native selector | 3.769790 | +0.0681% |
| Refit absolute regression | 3.774773 | -0.1515% |
| Refit pairwise ranking | 3.767578 | +0.0074% |
| One seeded random subset | 3.769483 | +0.0737% |
| Single-region-oracle top4 | 3.753697 | +0.4332% |

Pairwise improves raw task loss over refit regression by0.007195 on average,
and beats the seeded random subset on10/12 slates. Against historical native
selection it wins5, ties1 and loses6. Its raw mean is slightly better but its
mean per-tuple normalized benefit is worse; the conclusion depends on
aggregation. State11/01/10 raw loss changes versus native:
-0.006676/-0.000855/+0.000896, only4 correlated tuples/state.
Pairwise accuracy is50.49%, versus49.93% for regression; mean top4 overlap
is1.50 versus0.83. Ranking quality remains weak.

Median absolute region benefit train/development/heldout is
0.0001707/0.0001890/0.0001406. Ordinary ranking alone has not established
reliable task improvement. Neither objective is promoted into the running
tracker or formal training. This does not establish convergence of a full
tracker, PR/SR gains or any of the three requested module contributions.

Selected heads:

- regression_epoch045.pth:
  aff97894e7fba33c0febf3e0d797675b71950b45c33b63cc702d69e2c9eb5555
- pairwise_epoch026.pth:
  47b2777166fe4d01afa6e73d0fc85d2be57563f4c617eb3b8d5effdf5e4f85f0

Raw report, labels, features, selected heads and analysis were copied locally
under outputs/server28-selector-objective-control-20261030.

## Preregistered Protocol And Launch History

CPU pilot PID3918799 started on server28 at2026-10-30T16:21:45Z.
At16:26:08Z it had completed39 training slates across13 sequences, with
no terminal error. All four GPUs remain allocated to existing training.

The historical A utility epoch20 model supplies frozen features, correction
and task head. One causal tuple per sequence, complete templates, no patch
mask and11/01/10 search states produce16 single-region interventions.
Train: first16 train882 sequences; development: next4 train882 sequences;
heldout: first4 val97 sequences. All use deterministic validation
preprocessing. This is24 sequences/72 slates/1152 single-region labels.
Historical E0 exposure to val97 remains uncertified. "Heldout" describes
selector-head pilot fitting, not all historical backbone training.

Both selectors start from the same historical parameters and use AdamW
LR1e-4/weight_decay1e-4. Absolute utility SmoothL1 regression is compared
with ordinary unweighted pairwise logistic ranking on strictly ordered
benefits. Ranking ignores benefit magnitude and can learn very small
numerical differences. It is a generic control, not a novel module.

Each objective uses its own development objective, min30/max100 epochs,
patience10 and0.2%relative improvement. Select the best checkpoint among
the final5, then remove the other4 with hash receipts. Heldout evaluation
occurs only after selection. Different objective values are not directly
comparable.

Primary interpretation uses actual joint task loss after applying the
selected4 regions, compared with unchanged native selection and one seeded
random subset per tuple. The single-region oracle ranks individual benefits;
its joint subset is not guaranteed optimal. Four tuples per state are a
small correlated diagnostic, not tracker performance or benchmark evidence.
No changes to running training follow from this pilot alone.

Evidence:

- reports/server28-selector-objective-control-20261030/launch.json
- reports/server28-selector-objective-control-20261030/labels.json
- reports/server28-selector-objective-control-20261030/features_labels.pth
- reports/server28-selector-objective-control-20261030/report.json
- reports/server28-selector-objective-control-20261030/completed.json
- reports/server28-selector-objective-control-20261030/analysis.json
- selector_objective_control28.py and analyze_selector_control28.py

Latest verified training snapshot16:24:36Z:
GPU0 utility6786/50000, GPU1 random7254/50000, epoch3, finite recorded losses.
GPU2 utility1409 and GPU3 random1444 of2000 first-epoch updates, effective
batch10/LR1.25e-4,22710/24576MiB each, finite recorded losses. First epoch
boundaries estimated16:50:21/16:48:05Z before validation/save overhead.
Evaluation controller3463984 continues waiting at300s intervals.

User subsequently explicitly authorized concurrent evaluation/data collection
on one GPU. Candidate val97 and official evaluators already use4 workers
per GPU after the assigned training finishes. Actual full-run aggregate
memory and throughput are still unmeasured.

Latest snapshot16:31:14Z supersedes the launch snapshot: fixed utility6951
and random7429 of50000; candidates1561/1601 of2000 first-epoch updates.
All recorded key losses finite, no completed candidate epoch validation yet.
First boundaries still estimated16:50:20/16:48:08Z before validation/save.
