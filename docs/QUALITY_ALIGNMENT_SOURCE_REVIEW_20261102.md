# Quality Alignment Source Review

Read-only investigation while the preregistered tail pilot runs.
This does not alter its optimizer, stopping criterion, audit or gate.

## Evidence In This Project

`miss_rgbt/objectives.py:task_loss` supervises a Gaussian response map and
the box gathered at the current predicted peak. `CenterPredictor.cal_bbox`
in `lib/models/layers/head.py` selects the maximum response and gathers
size/offset there. The deployed tracker also applies a Hann prior.
Response probability is not explicitly supervised to equal predicted-box
IoU. This code difference is an observation, not proof of a ranking cause.

The earlier train882 response/geometry diagnostic found additional
GT-peak oracle headroom, especially for TIR-only searches, but its oracle
changes location and gathered geometry together. The subsequent matched
target-index regression training failed its gate. Do not repeat that
intervention or claim its negative result proves a scoring cause.

The earlier raw/Hann candidate controls also found removing Hann harmful.
Any new ranking study must retain the native prior and separate selecting
existing geometry from learning new regression geometry.

## Primary Source

Official `fcjian/TOOD` master read at
`1bda27c6f290c5b828c566eb95ea50fed75a53cb`.
File `mmdet/models/dense_heads/tood_head.py`, downloaded text UTF8 SHA256:
`2f92e486d2ac7db835dede8f7b8008c2d1dcddd1e2166fb9e516fff804a599ef`.
URL: https://github.com/fcjian/TOOD/blob/1bda27c6f290c5b828c566eb95ea50fed75a53cb/mmdet/models/dense_heads/tood_head.py

- Lines223-225 combine class logits and a learned probability branch.
- Lines321-328 pass alignment metrics into the classification loss after
  the initialization phase.
- Lines343-353 weight regression with those alignment metrics.
- Lines684-689 normalize assigned metrics and scale them with assigned
  IoUs before forming targets.

These are detector-specific task-alignment interfaces. No file was copied,
dependency installed or detector assigner transplanted. The source alone
does not establish a missing-modality tracker benefit, temporal stability,
novelty or an adequate target-identity signal.

## Distinct Prospective Question

If another mechanism investigation is needed, first measure whether native
high-response candidate locations already contain materially better boxes
than the selected peak on fresh train882 crops. Compute dense boxes from
unchanged size/offset maps; compare native selection, Hann selection,
fixed top-k native candidates and a diagnostic-only GT-IoU oracle.
Keep the regression maps fixed and record complete as well as both missing
search states and template histories. This separates existing candidate
headroom from retraining regression.

Only after registering that diagnostic and finding useful headroom should
we consider a matched learned quality-ranking test. A future learned
ranking head would compare box-quality targets against a same-capacity
response-target control, preserve native inference availability and test
fresh closed-loop data before promotion. A GT oracle is never a deployable
selector. Ordinary task-aligned ranking is prior art, not one of the three
validated paper modules merely because it is implemented.

No new training, diagnostic execution, official score or positive claim is
recorded by this source review. Full Complete +2 points, RGBT234-Miss
thresholds and three-module requirements remain unmet.
