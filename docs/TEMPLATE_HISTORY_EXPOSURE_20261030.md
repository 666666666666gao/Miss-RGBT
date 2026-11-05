# Missing Dynamic Templates In Existing Tracking

This analyzes the already completed fixed8 CPU cohort of historical
A utility epoch20. It does not run a new tracker or change live training.
Trace SHA256 values are checked against the original cohort receipts.
For every frame, the dynamic template's source is reconstructed before
the optional current-frame write, and the recorded post-write source
must match. This avoids counting a new template as an input to the same
frame's prediction.

## Actual Exposure

Under the fixed alternating32-frame schedule:

| Quantity | Count |
|---|---:|
| Video frames after initialization | 4594 |
| Observed search frames | 3950 |
| Observed frames with a missing-modality dynamic template | 985 |
| Complete search frames | 2336 |
| Complete search with a missing-modality dynamic template | 134 |
| Single-modality search with a missing-modality dynamic template | 851 |
| Single search and dynamic template with opposite available modalities | 32 |
| Dynamic-template writes | 448 |
| Single-modality dynamic-template writes | 154 |

Missing dynamic templates are present on24.94% of observed frames and
5.74% of complete-search frames. The fixed initialization template remains
complete; "opposite modalities" describes only the current search and the
dynamic slot, not absence of all matching target information.

| Sequence | Observed | Missing dynamic | Complete search, missing dynamic | Single-modality writes |
|---|---:|---:|---:|---:|
| boywalkinginsnow3 | 479 | 215 | 50 | 27 |
| leftdrillmasterstanding | 1839 | 245 | 26 | 39 |
| leftgirlunderthelamp | 225 | 65 | 10 | 8 |
| girlridesbike | 159 | 60 | 5 | 11 |
| midboyplayingphone | 296 | 125 | 15 | 19 |
| boywithumbrella | 780 | 250 | 28 | 46 |
| manrun | 60 | 25 | 0 | 4 |
| ab_pingpongball | 112 | 0 | 0 | 0 |

These counts include all observed frames, including invalid-size GT that
cannot supply geometry labels. This explains why3950 differs from the
prior geometry cohort's3924 observed valid frame pairs.

## Consequence For The Protected Branch Control

The six-epoch frozen-E0 LocalRepair pilot uses complete templates in every
sample. A complete-current search after an outage can therefore differ
from its all-complete sampled reference. The template tokens themselves
may have an unavailable modality, which activates the existing repair gate.
Sampled11 output equality does not imply equality on those mixed histories.

This is a demonstrated exposure mismatch for the simplified CPU pilot.
It is not a newly discovered omission in historical A formal training:
`miss_rgbt.train.sample_masks` keeps the first template complete and
samples the dynamic template and searches independently among11/01/10.
No change is needed to the running A campaigns on this evidence.

The queued paired development tracker intentionally retains native
dynamic-template availability and updates, so it will test actual mixed
histories if they arise. After that fixed control, a subsequent full
protected-branch training should include the already established
dynamic-template masks rather than claiming the all-complete-template
pilot covers deployment. Do not add a separate compatibility path or
change the current pilot halfway through.

The association between template availability, frame difficulty, IoU and
crop loss is confounded. These descriptive groups do not establish harmful
template writes or justify disabling native updates. Historical U0/U1/U2
controls remain separate evidence. This is not a module contribution,
official benchmark result or independent test of selected LocalRepair.

Source: `analyze_template_history28.py`.
Input: `reports/server28-val97-cpu-cohort8-20261030`.
Local input copy: `outputs/server28-val97-cpu-cohort8-20261030`.
Result: `reports/template-history-exposure-20261030.json`.
