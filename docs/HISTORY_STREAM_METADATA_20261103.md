# History-Stream Metadata Feasibility

This is a completed metadata check, not a passed OHCAS data gate. No images
were opened, no cache was created, and no diagnostic GPU was allocated.

## Observed Result

The existing server Python 3.7 environment executed
`scripts/audit_history_stream_metadata.py` against LasHeR training metadata.
Under the completed historical diagnostic/cache exclusions inherited from
`train_dense_geometry.exclusions()` and its completed dense cache:

- Fitting partition: 705 sequences; all 705 have at least one candidate.
- Eligible current-search positions: 282,120.
- Eligible history/current pairs: 15,976,753.
- Required diagnostic cohort: 128 tuples; sufficient distinct sequences
  under these partial exclusions.
- Historical age: 1--64 frames.

Each counted pair has two distinct fresh visible templates earlier than the
history frame. Those templates can serve both template contexts. Current
and historical images must be fresh and visible. The previous-current frame,
whose GT supplies shared crop geometry, must also be fresh and visible; it
may coincide with the history frame. Crop acceptance remains unmeasured.

This is a single-search diagnostic condition. It is not the native training
sampler's two-template/three-search eligibility condition, so the older
704-sequence native-sampler result is not contradicted.

## Open Exclusion Boundary

The actual report records `data_gate_passed=false` and
`all_historical_frame_exclusions_verified=false`:

1. At report creation, partitioned A Utility/Random were still training. At
   the 2026-11-04 01:24 CST recheck, both had completed, both completion
   receipts were present, and `verification_v2.json` reported `passed`.
   Their manifests are now frozen. The separate full official evaluation
   queue remains in progress and has not passed its final formal verification.
2. R1/R2 are still training. Neither run has `input_manifest.jsonl`.
   The frozen trainer records tensor hashes for the first optimizer update,
   but no per-frame identities. Waiting for completion alone will not create
   those identities.

Therefore these counts are upper bounds under a partial exclusion union.
Example identities in the report are neither reserved nor approved for an
image cache. A complete provenance solution for R1/R2 is still needed if
their inputs fall within the required historical exclusion scope.
No running source, seed, sampler, loader, or process was modified to repair it.
The checker never treats missing exclusions as permission to launch.

A subsequent real native CPU loader replay recovered frame identities for
four microbatches and matched every recorded R1/R2 first-update tensor hash.
See `docs/PROTOCOL_R_INPUT_WITNESS_20261103.md` and its hash-bound raw report.
This prefix evidence does not establish full-run identities or close the
historical exclusion boundary described above.

## Reproduction

CPU helper arithmetic was compared with exhaustive enumeration on 3,060
small synthetic cases. This establishes counting/indexing arithmetic,
not image acceptance or model parity.

```bash
cd /home/gaob/Miss-RGBT
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 PYTHONPATH="$PWD" \
  conda-envs/miss_rgbt/bin/python -m research.history_stream.check_metadata \
  --output reports/server28_history_stream_metadata_checks_20261103.json
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 \
  PYTHONPATH="$PWD/runtime/task_utility_20261018:$PWD" \
  conda-envs/miss_rgbt/bin/python scripts/audit_history_stream_metadata.py \
  --output reports/server28_history_stream_metadata_20261103.json
```

These exclusive-output commands have already run successfully. Use new output
paths for another observation. The runtime prefix is necessary because the
historical exclusion helpers import `utility_supervision` from the original
A runtime. An initial invocation using only the project root failed that
import before opening metadata or writing a report. No package/code change
was made; the documented successful command selects the existing A runtime.

Reports are copied into the publication checkout. Each metadata report binds
its source files and prior exclusion reports by SHA256. Neither this report
nor the synthetic result is an official tracking metric or a passed Step 4.

## R1/R2 Manifest Recheck

At 2026-11-03 12:40 CST, both R trainer PIDs and the P queue were still live.
The R `steps.jsonl` files were being updated, but neither run had an
`input_manifest.jsonl`. Inspection of the frozen `research/protocol_r/train.py`
confirms that `first_inputs` is populated only while `updates == 0`, and the
input tensor hashes are written only to the first optimizer-update record.
Subsequent step records contain training telemetry, not input hashes or frame
identities. The current logs therefore cannot establish the complete
historical frame-exclusion union for these runs. The existing first-update
hash witness and CPU replay prefix do not extend that evidence to later steps.
At 12:42 CST, the deployed snapshot and published trainer source had identical
SHA256 `1709f566d80cad56405d0601ea0f1301168d559c70f4790086bfaebfebc4d5d3`;
the two `steps.jsonl` files were still growing and all five parent/queue PIDs
were live.

The active run and its frozen snapshot were not edited or restarted. OHCAS/M1
remains unauthorized until complete historical frame provenance is supported
by evidence; do not infer it from the scalar step log or the partial replay.

## Diagnostic Runtime Correction

Read-only source inspection found that the project-root `miss_rgbt/model.py`
is an older model and the root package has no `task_utility.py`. It differs
from the frozen A runtime used to produce Batch10 Utility ep17. The original
history launcher selected that root package, so its actual Utility diagnostic
would fail before useful execution.

`scripts/launch_history_stream.py` now runs from
`runtime/task_utility_20261018`, with that runtime before the project root
on `PYTHONPATH`. The diagnostic's source manifest resolves research/script
files from the project root and native model/config files from its working
runtime. Active trainers, snapshots and queues are unchanged.

The existing server environment imported the corrected diagnostic and
strictly loaded the actual ep17 checkpoint on a fresh CPU model. Its SHA256
matched the registered diagnostic checkpoint; CUDA stayed uninitialized.
The raw receipt is `reports/server28_history_diagnostic_runtime_20261103.json`.
The existing sixteen diagnostic helper checks also passed in this runtime;
their receipt is
`reports/server28_history_stream_diagnostic_runtime_checks_20261103.json`.
A mocked launcher check confirmed the native working directory/import path
and rejection of an unverified cache before any subprocess launch.

These checks establish import/checkpoint compatibility and helper behavior.
They do not create a fresh cache, execute image interventions, decide M1,
recover original R frame provenance or produce official tracking metrics.

## Completed Replay Exclusions

The metadata CLI now accepts `--completed-r-replay` for a completed
epoch-transition witness. It verifies terminal/report/source/run-config and
manifest hashes, then adds the replay's full first-epoch identities and
second-epoch prefix. It does not read a growing replay or mark original R
provenance complete.

Actual CPU metadata execution with the completed real-image replay found
688 eligible sequences out of 705 fitting sequences, 202,523 current-search
positions and 10,496,674 history/current pairs after these additional
exclusions. The earlier partial-exclusion bound was 705/282,120/15,976,753.
No new diagnostic images or cache were created by this metadata pass.

Raw receipt:
`reports/server28_history_stream_metadata_replayed_epoch1_v2_20261103.json`,
SHA256 `645846c461833abd0e4857f85fc6cbf44421ba820fe0c5cd75e05d3e683f1492`.
It retains `data_gate_passed=false`: original A terminal verification and
the complete original R frame history remain unresolved. These counts
still describe a partial-exclusion upper bound, not approved image tuples.
