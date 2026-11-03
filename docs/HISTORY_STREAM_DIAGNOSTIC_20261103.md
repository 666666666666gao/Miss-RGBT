# Frozen History-Stream Diagnostic

Status: executor and CPU intervention fixtures implemented. Fresh image-cache
preparation, real-image parity and actual diagnostic execution remain pending.
This is not a passed Step 4 gate and does not authorize OHCAS training.

## Protocol

Use the frozen Batch10 Utility epoch17 checkpoint:
`runs/server28-recovery-20261031/batch10_utility_seed2027/epoch17.pth.tar`,
SHA256 `fbeaf71a9b4bbfb46e926251b34c7fb1960c4ba41d3da75745342638fd4dc062`.
The executor pins this identity rather than accepting arbitrary candidate weights.
Server28 file hash was rechecked before implementation.

Use 128 fresh tuples from train882 fitting sequences (split-list index modulo
five is nonzero). No val97 or official-test images. Each tuple contains two
current templates, one search, and an earlier visible same-sequence history
frame aged 1--64 frames. History has its own two causally earlier templates.
Current/history search crops share the previous-GT crop geometry and factor four;
the future target annotation must not position the historical crop.

After block three, replace only the unavailable modality's search-token segment.
`true_hist` uses its own modality's historical features without a learned map;
`wrong_seq` uses a deterministic different-sequence donor; `copy_avail` uses the
currently available modality's features. Both missing directions are measured.
The history forward uses its own complete, causal template context.

`zero` is the unmodified native missing-stream path, **not** literal zeros after
block three. The native path starts from positional embeddings and has already
passed through three blocks. Literal post-block zeros would change the control.
Template tokens and available-stream tokens are never replaced. All four modes
must exactly preserve complete-input score/size/offset maps and predicted boxes;
`zero` must exactly match the unhooked native output in all three states.

Use unwindowed native crop-box IoU and native per-sample task loss with batch one.
Report paired, sequence-clustered 10,000-resample 95% bootstrap intervals;
positive loss gain means reduced task loss. Advance only if true-history versus
zero IoU CI lower bound is positive in both directions and true-history mean IoU
exceeds wrong-sequence history in both directions. Task loss is reported, not
an additional selection criterion. Do not tune age on this cohort.

If block three fails this measured gate, retry once at layer zero: after
patch/position embedding, before the first Transformer block. Use the same
checkpoint, cache, configuration, controls and decision rule. The executor
requires the negative block-three report and its hash-bound completion,
recomputes its effects and verifies unchanged source/cache/configuration hashes.
A positive first diagnostic does not authorize the retry. Decisions are
`accepted_block3`, `retry_embedding`, `accepted_embedding`, or
`closed_both_diagnostics_negative`. Missing provenance or an unexecuted
diagnostic is not a negative result and cannot close M1.

## Data Gate Still Pending

`scripts/diagnose_history_stream.py` consumes a frozen `torch.save` list with
128 entries, not raw dataset directories. Each entry contains `templates`,
`history_templates` (two `[1,6,128,128]` tensors each), `search`, `history`
(`[1,6,256,256]`), `target` (`[1,4]`), and:

- `identity`: sequence, template_frames, history_template_frames, search_frame,
  history_frame, history_visible.
- `crop`: current and history equal crop-geometry records plus factor=4.

The JSON cache receipt must contain status=complete, split=train882_fitting,
samples=128, history_max_age=64,
crop_policy=previous_gt_shared_current_history_factor4,
cache/cache_sha256, source_sha256, all_historical_frame_exclusions_verified,
excluded_runs (completed/completed_sha256/manifest/manifest_sha256), and
frame_exclusion_audit/frame_exclusion_audit_sha256. The audit must record
status=passed, overlapping_frames=0, and SHA256 of the sorted-key JSON list
of all cache identities. It must independently check every template, search
and historical frame against the frozen exclusion union.

No such cache or audit was produced in this implementation step. A manifests
are still growing. Before preparing the cache, inventory all historical input
manifests, freeze completed A exclusions and explicitly resolve the scope of
the concurrently running R1/R2 manifests. Do not snapshot a growing manifest
and call it a completed exclusion union. Check metadata feasibility again
before opening images; do not silently weaken freshness if too few frames remain.

## Execution

CPU-only fixture check (does not open images or allocate CUDA):

```bash
cd /home/gaob/Miss-RGBT
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 PYTHONPATH="$PWD" \
  conda-envs/miss_rgbt/bin/python -m research.history_stream.check_diagnostic \
  --output reports/server28_history_stream_cpu_checks_20261103.json
```

After the data gate and a verified free card:

```bash
cd /home/gaob/Miss-RGBT
conda-envs/miss_rgbt/bin/python scripts/launch_history_stream.py \
  --gpu 2 --cache-receipt <verified-cache-receipt.json> \
  --config reports/server28-candidates-20261030/batch10_utility_seed2027/config.json \
  --output reports/server28_history_stream_20261103
```

This is a command template, not an already executed launch. The launcher refuses
occupied cards (memory >=500MiB), temperatures >=85C, or power caps other than
250W. It does not change temperature settings or guarantee a thermal ceiling.
Follow the child log and completed receipt; check long jobs at 180--300-second
intervals near an estimated milestone. Preserve existing healthy jobs.

Only after an actual completed negative block-three diagnostic:

```bash
conda-envs/miss_rgbt/bin/python scripts/launch_history_stream.py \
  --gpu 2 --cache-receipt <same-verified-cache-receipt.json> \
  --config reports/server28-candidates-20261030/batch10_utility_seed2027/config.json \
  --injection-layer 0 \
  --previous-diagnostic reports/server28_history_stream_20261103/report.json \
  --output reports/server28_history_stream_embedding_20261103
```

CPU fixtures cover both hook positions, hook cleanup, four decision outcomes
and rejection of missing/positive/wrong-layer prior experiments, changed cache
or configuration, and inconsistent effects. They do not supply the real-image
or freshness evidence required by either diagnostic.

Server28 Python 3.7 / Torch 1.12 CPU execution passed these fixtures; receipt:
`reports/server28_history_stream_retry_cpu_checks_20261103.json`, SHA256
`f56f3c9ee057619674771b052ec4ec45337175d16e952f136e0eaf8846a62dd7`.
CUDA was not initialized, real images were not executed and the diagnostic
gate remains unpassed.

The executor checks unchanged model parameters/buffers, no gradients, finite
outputs, exact complete parity, and writes report plus SHA-bound completion.
CPU fixture success proves intervention indexing and decision arithmetic only;
it does not prove native real-image parity, useful history, training improvement
or any official benchmark metric. Existing A/R training and evaluator sources
remain untouched.

## Saved Decision Verification

`scripts/verify_history_stream.py` is the consumer entry point for an actual
completed diagnostic. It validates the registered checkpoint, completion hash,
source/configuration hashes, and the existing terminal-cache provenance checks.
It checks all 384 paired observations against the 128 cache identities and
deterministic different-sequence donors, then recomputes the bootstrap effects
and decision from raw observations. Complete-input metrics must remain identical
across interventions. An embedding retry also re-verifies the original negative
block-three report using the same cache and configuration.

```bash
cd /home/gaob/Miss-RGBT/runtime/task_utility_20261018
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 \
PYTHONPATH=/home/gaob/Miss-RGBT/runtime/task_utility_20261018:/home/gaob/Miss-RGBT \
/home/gaob/Miss-RGBT/conda-envs/miss_rgbt/bin/python \
  /home/gaob/Miss-RGBT/scripts/verify_history_stream.py \
  --report <completed-diagnostic/report.json> \
  --cache-receipt <same-verified-cache-receipt.json> \
  --config <same-config.json>
```

This is a command template: no actual diagnostic report/cache currently exists
to verify. `retry_embedding` is explicitly nonterminal; accepted diagnostics or
two measured negative diagnostics are terminal decisions. The output explicitly
does not authorize OHCAS or F training. Accepted OHCAS still requires actual
composition, and F still needs campaign preparation and the other parent gates.

The server's frozen A CPU runtime passed three focused test methods, including
all four decisions, the actual recursive retry-check logic, and rejection of
modified pairs, donors, state/identity layouts, missing rows, nonfinite metrics,
completion hashes, sources, and cache-load failures. Their report is
`reports/server28_history_decision_verifier_cpu_20261103.json`.
Checkpoint identities and cache loading are mocked in these tests; the paired
records are synthetic. These tests do not establish real cache freshness, actual
image interventions, M1 acceptance/closure, or official tracking metrics.
