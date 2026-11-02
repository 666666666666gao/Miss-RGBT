# Official queue after partitioned A training

Registered after training began and before any formal checkpoint or official
outcome. Separate from the immutable training sources/protocol. No live
training code, optimizer, stopping rule, partition or selection changes.

The controller follows original training PIDs1288280/1288281. Every300s,
check actual process command identity and terminal receipt presence. A dead
worker without a completion receipt fails the queue; no automatic restart.
Wait until both original workers are terminal. Then run
scripts/verify_partitioned_a.py --phase formal on CPU. Any failed source,
data, weight, pairing, stopping or final-five selection check stops the queue.
Only a passed verification receipt authorizes evaluation.

For Utility onGPU2 and Random onGPU3, independently enumerate final-five
epochs chronologically, each in complete then miss. Each full run uses
evaluate_candidate_official28.py with4workers per GPU, no limit argument.
Check assigned GPU<500MiB,<85C,250W at each evaluator launch. No other jobs
are authorized on these two GPUs between training and evaluation.
Expected20evaluations total,479sequences/337353frames EACH,9580trajectories
and6747060frames across the full campaign. No partial evaluation, val97
development, official-driven stopping, parameter adjustment or weight cleanup.

After each full evaluation, call ONLY verify_recovery_campaign28.verify_official:
rehash checkpoint, evaluator, frozen runtime sources, masks, GT and all479
individual receipt/bbox outputs; recompute the full official toolkit metrics
and curves. Never call verify_development or that module's main. Imports
do not execute their CLI main functions.

Persist every verified official outcome as selector_verified_metrics.json.
Do not present an interim subset score as an official result. The same
preselected development checkpoint applies to complete and Miss. Full
final-five comparisons are disclosed as post-hoc; preserve all weights.
No new module, target attainment or model superiority is inferred from queue
registration or execution compatibility.

Checks before registration: Python3.7compile, inspect generated smoke-weight
plans without executing a benchmark, strict inference compatibility witness
on synthetic images only, and source/hash/actual worker identity checks.
No dependency or environment changes. Queue frozen-source hashes include
evaluator and official verifier; later source changes stop the queue.

Remote registration (same warm environment; helper detaches the controller):

```bash
/home/gaob/Miss-RGBT/conda-envs/miss_rgbt/bin/python /home/gaob/Miss-RGBT/scripts/launch_partitioned_official_queue.py
```

The helper requires a passed queue-plan/synthetic-inference check. It writes
a persistent log and controller PID/command receipt. The sleeping controller
has CUDA_VISIBLE_DEVICES empty; only later evaluator children receive GPUs.
Return promptly to avoid blocking the chat; no user-facing repeated
unchanged-state notifications. Training ETA remains about24-34h; official
evaluation adds substantial time and ETA must use actual completed full runs.
