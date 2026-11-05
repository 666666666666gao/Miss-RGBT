# Training And Selection Lifecycle Audit

Observed2026-10-31T05:25:42Z from actual server configuration files,
launch receipts, current source hashes and live controller command lines.
This distinguishes the two active campaigns; neither has completed.

| Campaign | GPUs | Stopping | Weight selection |
|---|---|---|---|
| Original fixed-budget utility/random | 0/1 | Fixed20epochs | Epoch20 only; saves10/20 |
| Adaptive batch10 utility/random | 2/3 | Post-LR-drop plateau, maximum20 | Full val97 tracking on final5, one common selected weight |

The original fixed-budget controls do **not** implement final-five
checkpoint selection. Their run configs contain checkpoint_epochs[10,20],
and the live `evaluate_after_training28.py` explicitly usesepoch20.
No claim that all four runs perform last-five selection is valid.
Those frozen controls remain a fixed-budget comparison. Adaptive candidate
campaigns implement the user's requested final-five workflow for candidate
models. Completion of that workflow remains unproven until real final
weights, evaluations, selection and cleanup receipts exist.

## Candidate Workflow

Effective batch10, LR1.25e-4, backbone multiplier0.1. LR drops afterepoch10.
The validation reference resets atepoch11; patience3 means the earliest
possible actual stop isepoch14, despite configured minimum12.
Every epoch saves a checkpoint and removes the one older than the rolling
five-epoch window. Stopping retains the lastfive, not the global loss best.

Each final checkpoint receives complete and alternating-outage tracking
over97 validation videos/54646frames per condition. Rank checkpoints by
mean SR over those two conditions, choose earlier epoch on exact ties.
One selected checkpoint is used for all official complete/Miss datasets.
The other four final-window weights are deleted after selection/hash
verification with file-size/hash receipts. Official scores do not select
the weight. Training-validation proxy loss is not the final tracking score.

Actual candidate controllers3816049/3816050 remain live. Current
train_candidates28.py, candidate_validation28.py, evaluate_candidate_val97_28.py
and candidate_campaign28.py match their launch hashes for both arms.
The official evaluator differs from its launch hash for the already audited
GPU-option/worker/subset-smoke changes, not a newly discovered mutation.
See docs/CANDIDATE_EVALUATOR_SOURCE_AUDIT_20261030.md. The future run.json
must record the actual evaluator version used.

## Fixed-Budget Evaluation Queue

Live PID3463984 invokes `evaluate_after_training28.py`. It waits for both
successful original training exits, requires20epochs/50000updates and
frozen runtime source hashes, checks GPUs0/1 are free, then sequentially
runs utility complete/Miss and random complete/Miss. Every condition
requires479videos/337353frames. It does not evaluate the lastfive or
delete an unselected final-five set.

The exact server script is now archived in the publication repository,
without modifying the running copy. Its SHA256 is
`b0f91914a219109f679424e3692c21ace5a66a16ac7730c6ea010783fba3f912`.
This archive closes a source-provenance gap; it changes no training,
evaluation behavior or GPU settings.

Evidence: reports/training-lifecycle-audit-20261031.json.
No convergence, future successful completion, target attainment or
three-module validation follows from this lifecycle audit.
