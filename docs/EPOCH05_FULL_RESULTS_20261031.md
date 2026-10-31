# Epoch Five Full Validation Results

The six-cell epoch5 health evaluation completed normally at
2026-10-31 11:23:42 UTC+08:00. Postprocessing completed at11:24:33.
All582 saved trajectories, GT hashes, per-video metrics and aggregate
metrics were verified. Each cell contains97 videos,54646 video frames
and54289 scored frames. This is development val97 with fixed alternating
outages, not either official Miss benchmark or final checkpoint selection.

| Model | Complete PR20 | Complete SR | Outage PR20 | Outage SR |
|---|---:|---:|---:|---:|
| E0 | 75.879164 | 61.654017 | 64.528129 | 52.346049 |
| Utility epoch5 | 79.596578 | 64.472777 | 70.868974 | 56.749288 |
| Random epoch5 | 80.667388 | 65.321675 | 71.457491 | 57.152398 |

Outage utility-minus-E0 gains are+6.340845 PR20/+4.403239 SR.
Paired-video bootstrap95% intervals are[1.781127,11.253817] and
[1.015466,8.022615]. Random-minus-E0 gains are+6.929362/+4.806349.
Utility-minus-random is-0.588517/-0.403110 with intervals
[-3.312053,2.193780]/[-2.399991,1.599819]. Utility has no demonstrated
advantage over matched random selection. Video resampling does not
measure training-seed uncertainty; historical E0 exposure to val97
remains uncertified.

## Outage Penalty Analysis

A new CPU-only analysis reads the same582 saved trajectories, checking
their hashes and metrics again. For each model/video, subtract its
complete-input score from its outage score. Compare these paired penalties
between models, using10000 video bootstrap draws with seed2027.
Positive penalty contrasts mean less degradation.

| Model | PR20 outage penalty, pp | SR outage penalty, pp |
|---|---:|---:|
| E0 | -11.351035 | -9.307968 |
| Utility | -8.727605 | -7.723488 |
| Random | -9.209897 | -8.169277 |

Utility's penalty improvement versus E0 is+2.623430/+1.584479,
with intervals[-2.262756,7.648158]/[-2.179436,5.385317].
Versus random it is+0.482292/+0.445789, with intervals
[-2.807630,4.010450]/[-2.048380,3.098218].
Thus the larger outage score gain cannot yet establish a distinct
robustness mechanism independent of complete-input improvement.

State-specific SR penalties, averaged over videos containing each state:

| RGB/TIR state | Videos | Frames | E0 | Utility | Random |
|---|---:|---:|---:|---:|---:|
| 00 hold | 62 | 7503 | -26.475055 | -30.159015 | -32.209204 |
| 01 TIR only | 97 | 10086 | -15.765364 | -11.400359 | -12.219802 |
| 10 RGB only | 84 | 8865 | -8.069995 | -4.953139 | -5.542517 |
| 11 both observed | 97 | 27835 | -2.937169 | -3.225898 | -3.384355 |

The00 and11 penalties have not improved at their point estimates.
The01 utility-minus-E0 penalty contrast is+4.365005 SR, with an
unadjusted exploratory interval[0.156455,8.720148]. All state-wise
utility-minus-random penalty intervals cross zero. These post-hoc groups
are descriptive, with multiple comparisons and different video membership;
they do not establish causal sensor effects, template damage, or a
validated motion/history module. Fit future method parameters on
train882 development, not these groups or official tests.

## Training And Next Decisions

At11:32:50 UTC+08 utility/random batch10 trainers3816056/3816057
were live, epoch8, updates14007/14466; all recorded losses finite.
Epoch7 mean task losses are5.069815951/5.136228281, worse than each
model's prior best. Keep the existing LR drop after10 and convergence
reference reset at11; no convergence claim from this rebound.
Next epoch8 boundaries were approximately12:52 for random and13:27
for utility, based on recent rates, with scheduling uncertainty.

At11:33:25 fixed20 trainers3453478/3453479 were live in epoch9,
updates20767/20806. Evaluation controller3463984 is waiting normally.
Batch10 controllers3816049/3816050 were also verified live.
Final-five full-val97 selection, one common checkpoint per model, other
four checkpoint cleanup, and official complete/Miss evaluations remain
pending. No live source, training schedule, weight, or GPU setting changed.
Official target attainment and three paper modules remain unproven.

## Evidence

- `reports/epoch05-full-paired-analysis-20261031.json`, SHA256
  `7034c378efa3dd1430dd32dcd94e968aaa11832a3e333f8a95b922aa106f7c31`.
- `reports/epoch05-full-analysis-completed-20261031.json`.
- `reports/epoch05-full-tracking-completed-20261031.json`.
- `reports/epoch05-{utility,random}-outage-20261031.json`.
- `reports/epoch05-robustness-analysis-20261031.json`, SHA256
  `fbda5dbf76cff4e812455b2202f3f97401bf5449318689f31dffa80452c5cfcb`.
- `reports/candidate-epoch07-progress-20261031.json`.
- `analyze_epoch05_robustness28.py`; uses existing frame metric and paired
  comparison helpers. No neural inference or training was repeated.
