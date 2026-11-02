# Train882 paired replication launch evidence

The previous goal turn only reported existing metrics and made no research
progress. Rechecked server28 idle and the clean publication checkout7340a53.
The availability-state block proposal was already rejected and not repeated.
Old candidate/recovery entry points use val97 and were not launched.

Added a separate full-network A trainer using the unchanged runtime's model,
masks, objectives and optimizer settings. The new train882705/88/89 boundary
and seed differ from historical training, so this is a reproducibility control
under a corrected boundary, not an exact historical rerun or a new module.
No environment rebuild, install or runtime-source modification occurred.

## Short-run verification

Each arm completed6epochs/12updates,120fitting and8development tuples.
Saved-evidence verification passed source/checkpoint hashes, data partitions
and exclusions, every paired input/mask digest, initial parameter equality,
finite last-five weights, stopping/selection arithmetic and actual backbone,
head and utility updates. Changed state tensors223/224;39BNbuffers frozen.
Both selectedepoch6 within final-five2-6. All12epoch weights retained.
Smoke losses establish no method benefit or full-training convergence.

The verifier shares recorded trainer/runtime sources; this is not independent
raw-image/optimizer replay or external scientific acceptance.
Evidence: reports/server28_partitioned_a_20261102_smoke/verification.json.
Small inputs/history/completion records published; weights stay on28.

## Formal launch

Launched2026-11-02 08:19:00UTC (16:19CST):

| Arm | GPU | PID | Seed | Effective Batch | LR |
|---|---:|---:|---:|---:|---:|
| Utility | 2 |1288280|2028|10|.000125|
| Random | 3 |1288281|2028|10|.000125|

Microbatch5xaccumulation2,20000tuples/epoch,max20,LRdrop10,
patience3afterdrop,earlieststop14. Fixed development128tuples seed63102.
All prior registered frames plus smoke frames excluded. No val97 or official
benchmark access by trainer/launcher. All checkpoints preserved, final-five
selection uses fixed train882 development loss. No evaluator auto-launched.

At16:21:22CST both original PIDs live at epoch1/update>=25,100%GPUutilization,
74/73C,22710/22742MiB,250Wcaps. First50microstep input/mask digests and
initial parameter digests match. All partition lists,879excluded-sequence
frame sets and prior hashes compare structurally equal; serialized boundary
bytes differ only by dict key order. This proves startup pairing, not a full
run or convergence. Observed3.014/3.006seconds/update predicts101min/epoch,
roughly24-34hours for14-20epochs; firstepoch around18:00CSTNov2.
Inspect near milestone,180-300s actual live-process checks; do not restart
on observation timeout.

Remote logs:
`/home/gaob/Miss-RGBT/reports/server28_partitioned_a_20261102_formal/{utility,random}.log`.
Weights/input manifests:
`/home/gaob/Miss-RGBT/runs/server28_partitioned_a_20261102_formal/{utility,random}/`.
Future fresh samplers exclude every training/development frame recorded
in active manifests; no parallel diagnostic reuses these cohorts.

After terminal receipts, run
`python scripts/verify_partitioned_a.py --phase formal` on server28.
Recheck stopping/selection/data/source/actual last-five weights before any
full official evaluation. Report all registered official final-five outcomes
as post-hoc comparison, retain a common development-selected weight, and do
not use test ranking for cleanup.

Official scores unchanged. Complete-input approximately+2,
RGBT234-Miss88.1/64.3 and three validated paper modules remain unmet.
