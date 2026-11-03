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

1. Partitioned A Utility/Random are still training. Their manifests exist,
   but neither completion nor formal verification exists. Their growing
   manifests were not read or frozen.
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
