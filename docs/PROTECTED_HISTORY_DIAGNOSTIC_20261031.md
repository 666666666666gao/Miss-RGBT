# Protected Branch With Missing Dynamic Templates

The existing tracking trace audit found missing dynamic templates on
985/3950 observed frames, including134/2336 complete-search frames. The
six-epoch LocalRepair pilot only trained with complete templates. This
diagnostic measures the resulting input mismatch without training or
changing any running GPU job.

Use the first four of the pilot's16 cached development tuples, with the
same loader, seed27182, and asserted original input hashes. These are
reused development data, not an independent test. Evaluate the first
search frame only. Keep the initial template complete and independently
set dynamic-template and search availability to11/01/10: nine cells.

Compare zero-initialized LocalRepair to the already selected epoch3 branch
using identical inputs. Record per-sample task losses, per-cell means,
selected-minus-zero differences and output hashes. Assert frozen baseline
state and bitwise equality for fully complete history11_11. Complete-search
cells with missing dynamic templates do not have that equality guarantee.
No model selection, threshold fitting, optimization or checkpoint deletion.

Runner: `audit_protected_history28.py`; CPU only, four Torch threads.
Output: `/home/gaob/Miss-RGBT/reports/server28-protected-history-20261031`.
The exclusive output directory prevents overwriting this diagnostic.
Estimate duration from the first cells; observe near completion at
180-300 second intervals. The results are a sampled task-loss diagnostic,
not tracking PR/SR, convergence, a new contribution or official metrics.

## Completed Result

PID552251 completed normally in40.86seconds, starting at
2026-10-31T00:02:12.792762Z (08:02:12+08:00). Both cached input hashes and
pilot source/checkpoint hashes matched. Baseline state was unchanged.

Selected-minus-zero task-loss differences (lower is better):

| Dynamic template | Complete search11 | TIR-only search01 | RGB-only search10 |
|---|---:|---:|---:|
| Complete11 | 0.000000 | -0.167142 | -0.060701 |
| TIR-only01 | -0.000819 | -0.126823 | -0.067348 |
| RGB-only10 | +0.000196 | -0.327170 | -0.043051 |

All six single-modality-search cell means improved on these four reused
development samples, including opposite-modality dynamic templates.
That does not prove uniform per-sample improvement or deployment benefit.
With complete search and missing dynamic templates the changes have mixed
signs. Their output hashes differ, demonstrating that the complete-output
invariant is limited to complete histories. Fully complete11_11 outputs
remain bitwise equal. Do not disable repair or template writes based on
this small, confounded diagnostic.

The next protected training should include the already established dynamic
template masks and evaluate mixed histories. This result does not by itself
establish harmful exposure, a new module, convergence or a benchmark gain.
No additional GPU training was launched.

Evidence: `reports/protected-history-report-20261031.json` (SHA256
`b1aa534cac68c16ed05c2ddbc8f88e40afb0a7b40a441a3ab6d9bea5cec9978c`)
and `reports/protected-history-completed-20261031.json`.
