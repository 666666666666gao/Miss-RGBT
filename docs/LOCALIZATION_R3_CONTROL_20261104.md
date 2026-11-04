# R3 Sidecar Control And Full Reporting

## Placement And Scope

The prepared R3 training snapshot is frozen and remains unchanged:
`/home/gaob/Miss-RGBT/runtime/localization_r3_20261104`.
Its source-manifest SHA256 remains
`454f97dbb72cb0f967aa06d69da6326a2cc044418426c3b9e7c54d6de3b524ac`.

New control files live outside it at
`/tmp/codex_r3_control_20261104.uYfXia/research/localization_r3/`:
`control.py`, `collect.py`, `check_control.py`. Execute the control/collector
by direct script path with the frozen snapshot, then the project root,
on PYTHONPATH. Do not put the staging checkout first for actual operation;
that would import the preparation module from the wrong source root.
The staged checkout is used first only for synthetic CPU unittest execution.

This adds no new model, mask schedule, hyperparameter choice, official-test
gate or weight selection. It does not amend R1/R2/P or unlock the failed
original R1 condition. Both real-image smokes and formal R3 remain unstarted.

## Per-Arm Lifecycle

`control.py` supports two phases, `smoke` and `train`. It first checks the
frozen preparation and the actual independent-review record. Missing
review returns waiting without querying a GPU or spawning a worker.
The expected review is the actual native fresh Codex result, with no
unresolved blocking issue, modelgpt-6-astra/max reasoning, same-family
independence and provisional acceptance as required by the installed
workflow. Tests create explicitly synthetic review fixtures in temporary
directories; none is a real review or may be copied into the campaign.

The smoke phase launches one registered four-update real-image smoke,
then returns without formal training. Before either train phase, both
smoke receipts must exist and the frozen trainer's paired-smoke checker
must pass. GPU preflight reuses the existing rule: memory below500MiB,
temperature below85C and power limit250W. Physical GPU is a controller
argument, keeping the paired frozen training configs identical.

The train phase owns one arm from full20 training through its ten full
official evaluations. It checks this arm's terminal training receipt,
all100,000 microstep records/50,000 updates, development history, retained
weights and the final-five development selection before evaluation.
It does not inspect the other arm's growing manifest. Each final weight
gets Complete then Miss using the same checkpoint and four evaluation
workers on that arm's GPU. The original full verifier recomputes each
result against ground truth before it enters the verified row list.

An arm may begin its own ten evaluations while its paired arm is still
training. This changes no other arm's frozen training policy; benchmark
scores never affect remaining training, selection or budget. The final
paired contrast is unavailable until both arms pass the full paired
training verifier. This avoids an unnecessary global wait on the faster
Complete-only arm's GPU.

Launch records and process logs use exclusive creation. One train
controller owns all its evaluations; there is no second standalone
evaluation/resume entry point. Child polling is300s. A nonzero child exit
is recorded and stops progression, without fallback, retry or restart.
No weights or existing results are removed or overwritten.

## Terminal Collector

`collect.py` waits for both official terminal receipts before calling the
frozen full paired-training verifier or reading a manifest. It then checks
the exact20 jobs, checkpoint pairing, internal selection and saved-output
evidence, and independently recomputes all20 full official results once.
Each result has LasHeR245/220703 plus RGBT234234/116650:479 sequences and
337353 frames. Total coverage is9580 trajectories/6747060 frames.

The report includes every registered row and each development-selected
checkpoint, its Complete deltas against frozen E0, strict Miss target
checks (PR69.3/SR54.4; MPR88.1/MSR64.3), and the registered fixed epoch20
development contrast. Exactly-equal Miss values do not pass. The2pp
comparison is a reporting reference, not a new selection/routing rule.
Passing saved-evidence checks does not establish multi-seed significance
or the effectiveness of all three proposed modules.

## Checks And Actual Readiness

Seven CPU unittest methods passed in3.135s. They exercise22 mocked child
commands (two trainings and20 full evaluations),40 mocked full-scoring
calls including collection, completed-arm100,000-record scans, duplicate
launch refusal, failed-child stop, missing review/smoke waits, terminal
binding rejection and strict target thresholds. All training checkpoints,
images, review, GPU readings and scorer outputs in those tests are
synthetic. No reviewer model or GPU training was called.

An actual read-only probe at2026-11-04 12:15:51UTC executed both sidecar
smoke readiness checks through the frozen imports. Both return
WAITING_FOR_INDEPENDENT_REVIEW. The collector returns
WAITING_FOR_TWENTY_FULL_RESULTS. Neither smoke/formal output exists,
CUDA is uninitialized and the frozen manifest is unchanged.
Raw report: `reports/server28_r3_control_readiness_20261104.json`.

The immediate unmet execution requirement is the independent review.
The installed experiment-bridge skill states: "Before deploying, send the
experiment code to a secondary Codex reviewer with max reasoning".
The current callable tool set has no native fresh reviewer/spawn operation.
Status remains REVIEW_UNAVAILABLE; local code checks do not satisfy it.
No real review record has been generated. Do not fabricate one to dispatch.
