# R1 Development-Selected Epoch16 Complete/Miss Pair

## Verified Scope

The registered R1 development-selected epoch16 now has both full results.
Complete was independently recomputed from saved boxes and dataset GT at
11:19:17UTC; Miss was independently recomputed at13:03:15UTC on2026-11-04.
Each covers LasHeR245/220703 plus RGBT234234/116650:479 trajectories and
337353 frames. The pair totals958 trajectories and674706 frames, ten
reported metrics. No new tracker or subset evaluation was executed.

Both use checkpoint SHA256
`0fe0af921bfc4ce2eba1301d3bc6f37a7adf68ea32a93542ecfc822dbcb8e03b`.
The Miss check rehashed the actual checkpoint and the four Complete
partner artifacts (run, completed, metrics, all_receipts); those hashes
match the earlier Complete verification. Complete was not recomputed
again. Both deterministic checks kept CUDA uninitialized. This is not
a fresh-agent semantic/code review, historical-input certification, a
terminal R/P campaign receipt or a three-module efficacy result.

## Ten Same-Checkpoint Metrics

| Condition | LasHeR PR | LasHeR NPR | LasHeR SR | RGBT234 MPR | RGBT234 MSR |
| --- | ---: | ---: | ---: | ---: | ---: |
| Complete | 74.738998468 | 70.997651296 | 59.504064142 | 90.909131531 | 67.059629734 |
| Miss | 68.959300473 | 63.846786667 | 54.809584553 | 84.674454775 | 61.388323549 |
| Complete minus Miss, pp | 5.779697995 | 7.150864629 | 4.694479589 | 6.234676756 | 5.671306185 |

## Strict Miss Targets

Equality does not pass. Gaps are score minus threshold in percentage points.

| Metric | Score | Required strict threshold | Gap pp | Passed |
| --- | ---: | ---: | ---: | --- |
| LasHeR-Miss PR | 68.959300473 | >69.3 | -0.340699527 | No |
| LasHeR-Miss SR | 54.809584553 | >54.4 | +0.409584553 | Yes |
| RGBT234-Miss MPR | 84.674454775 | >88.1 | -3.425545225 | No |
| RGBT234-Miss MSR | 61.388323549 | >64.3 | -2.911676451 | No |

The model passes one of four targets. LasHeR PR is close, but the full
RGBT234 Miss shortfalls remain substantial. These are one paired seed's
results; they neither show that stronger masking solves the gap nor
identify initialization, BN, gradient interaction or a history-module
defect as the cause. The Complete RGBT234 scores and their Miss losses
must be considered together; a Complete-only gain is not Miss attainment.

## Original Complete Gate And Next Action

Relative to E0, Complete LasHeR PR/NPR/SR change by
-0.897504420/-0.775185510/-0.621649534pp; RGBT234 MPR/MSR change by
+0.231997609/+0.214401404pp. Thus the frozen R1-selected E0-minus0.1pp
gate still fails all three LasHeR metrics. Do not replace R1 by R2,
another epoch or R3, relax the gate, or start old P around it.

The existing controller automatically launched R1 epoch17 Complete on
GPU0 at13:01:00UTC, PID2397874, with four workers. Its remaining registered
weights and the R2 full reporting campaign must finish unchanged. Do not
duplicate that dispatch or select a replacement using official maxima.
All produced checkpoints remain retained.

The separately prepared R3 C-only/C+M comparison remains the registered
next diagnostic for Complete-objective transfer, not proof of its cause
or a replacement for this failed gate. Its independent code review is
still unavailable and the requested human waiver has not been granted;
paired real-image smokes must also pass before formal training.

## Evidence

- `reports/server28_r1_epoch16_complete_verified_20261104.json`: earlier
  full Complete recomputation.
- `reports/server28_r1_epoch16_miss_verified_20261104.json`: full Miss
  recomputation, target checks and current Complete-partner evidence.
  Local/server SHA256:
  `12d89dda10413cade1baa3b644166fbd9a4e3f88b76cdd82054ed02398ca6b0f`.
- `reports/server28_r1_development_curve_20261104.md`: registered selection
  and its interpretation limits.

The Miss report's status=passed refers to deterministic saved-output
verification. Its all_miss_targets_met=false is the separate research
outcome; neither field authorizes new GPU work.
