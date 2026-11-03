# OSTrack Utility Six-Arm Training Entry

## Status

The isolated preparation and training entries exist and passed the CPU
contracts below on server28 on 2026-11-03. They have not read real images,
created a campaign, run an optimizer smoke, or produced official metrics.
The four GPUs remain assigned to the existing R1/R2 and A Utility/Random jobs.

Sources:

- `research/curriculum_p/protocol_prepare.py`
- `research/curriculum_p/protocol_train.py`
- `research/curriculum_p/check_protocol.py`

These entries do not modify the old E0-continuation P workflow. Do not restart
its retired controller PID2911513.

## Preparation Gate

Preparation first requires both R1/R2 training-verification and full-official
completion receipts. It checks frozen R sources, training configs and
completion records, twenty development-loss entries, final-five checkpoint
hashes, and the development-selected checkpoint. It independently recomputes
all twenty full R evaluations through `verify_official`, including the
479-sequence / 337353-frame coverage assertions.

The preregistered R1 selected checkpoint must meet the E0-minus-0.1pp
tolerance for all five Complete metrics. Failure requires a separately
registered R3; this entry does not silently substitute another protocol.
The gate binds parent artifacts by SHA256.

Only after this gate passes does preparation copy the frozen A implementation
plus the research entries into `runtime/rppm_ostrack_20261103` and write six
configs under `reports/server28_rppm_ostrack_20261103`.

## Training Contract

The arms are fixed60, fixed50, progressive, shuffled, paced, and paced+norm.
All share R1 optimization/data settings, OSTrack initialization, Utility
architecture and probe settings, twenty full epochs, no early stop, and
retained epoch10 and epochs16-20 checkpoints. The selected checkpoint is
the lowest mean 11/01/10 task loss among epochs16-20 on the fixed128-tuple
val97 cache; earlier epoch breaks a tie. This internal loss is not a
reported benchmark metric.

All six arms freeze every BN statistic because the existing Utility
counterfactual probes require it. R1/R2 freeze only head BN. This explicit,
common Utility protocol difference prevents describing the six arms as
exact R replicas.

Each microbatch records accepted sequence/template/search identities and
input tensor hashes. Instrumentation uses the last native template/search
frame-read pair after a valid crop is accepted. Rejected crop reads are
explicitly not recorded. This manifest does not establish that every image
ever read by the loader is covered, and it does not recover missing
historical R1/R2 frame provenance for OHCAS.

The loop uses the shared C-before-mask engine, alternating C/M Utility
probes, gradient accumulation, warmup, LR drop, and finite loss/gradient
checks. Logs include curriculum statistics, mask/input hashes, gradient
norm, learning rates, elapsed time and peak GPU allocation.

Saved checkpoints contain model, optimizer, curriculum state,
Python/NumPy/CPU/CUDA/probe RNG states and consumed loader position.
Worker RNG and prefetched batches are not captured. Resume is unsupported;
there is no resume CLI.

Each formal arm first requires all six actual four-update smokes, strict
model reload with identical outputs, and matching initial-state and input
manifest hashes across those smokes. Smoke evidence does not establish
full optimizer/RNG/loader resume equivalence.

## Executed Checks

Actual server28 command, with CUDA hidden and the existing A objective import:

```bash
cd /home/gaob/Miss-RGBT
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
PYTHONPATH="$PWD/runtime/task_utility_20261018:$PWD" \
conda-envs/miss_rgbt/bin/python research/curriculum_p/check_protocol.py \
--output reports/server28_rppm_protocol_checks_20261103.json
```

Result: PASS, `cuda_initialized=false`.

- Six configs share every field except arm/output.
- Seven invalid protocol mutations are rejected.
- Synthetic accepted-tuple identities are recorded correctly after a
  rejected candidate; input tensors and CPU RNG remain bitwise unchanged.
- The JSONL record reads back exactly.
- Four absent R completion/verification receipts block preparation.
- The actual unfinished R campaign remains blocked.
- Trainer `--help` and preparation `--check-gate-only` execute successfully.
- R `check_sources()` still passes after deploying only the new research files.

Report: `reports/server28_rppm_protocol_checks_20261103.json`.
SHA256: `98afd6e991a6db4751e9b5bdd9ae506b3c5298cbb34c116daec26a68ac9252b4`.
The checker is deliberately tied to the current unfinished R campaign; it
asserts that the actual gate is blocked. Do not rerun it unchanged after R
completion and interpret that assertion as a protocol failure.

## Remaining Work

The real-image six-arm smokes and native-loader parity remain unexecuted.
Positive completed-R gate execution and the snapshot training path remain
unexecuted. The formal saved-run verifier and full official controller for
this new protocol are not implemented. Sixty full evaluations are required:
six arms times five checkpoints times Complete/Miss. No new official score
or module efficacy follows from this CPU receipt.

Keep the healthy current jobs and their existing automatic full evaluation
queues running. Do not allocate another GPU worker to an occupied card.
Existing controller polling is300 seconds. The observed250W caps and
temperatures below85C are not a guarantee of a hard85C temperature limit.
