# Absent Search Token K/V Diagnostic

## Evidence and question

The exact frozen runtime multiplies missing modality patch embeddings by
zero and then adds positional embeddings before all transformer blocks.
Shared blocks do not receive availability. Thus a missing search stream
contains position-derived tokens that still act as keys and values in
self-attention. Existing negative block-bypass did not test attention
key/value isolation: it skipped ordinary residual updates for the entire
stream. This diagnostic asks whether absent search tokens contaminate
other tokens by serving as attention memory, while retaining them as queries
that can read observed template keys and values.

No model weights, head, tracker, optimizer, official benchmark or val97
changes. It is a frozen single-crop train882 diagnostic, not a deployed
policy or module.

## Frozen paired protocol

- Checkpoint: original CADTrack E0 epoch20,
  SHA256`d20c8641e4263df3ceadf9e430d0ac96c181c13ad482bca28c2dddef592a8bb8`.
- Sample128 tuples(seed43101) from existing705 fitting LasHeR train882
  sequence names; first search per tuple; all nine dynamic-template/search
  states. Initial template is always complete.
- Check that tuple identities do not exactly repeat any previous fusion,
  response geometry, head supervision, or modality-head pilot tuples.
  Sequence partitions are reused; this is not an independent-sequence study.
- Three paired forwards share exact same input tensors and native box head:
  `native` is untouched original block forward;
  `sham` uses the custom attention implementation with every key enabled;
  `masked` uses the same implementation but excludes unavailable current
  search tokens from keys/values. Query positions and all template keys stay.
- Custom sham must exactly match native score/size/offset/box tensors on
  every state before interpreting the masked comparison.
- Mask only search-token keys in the unavailable modality stream; never mask
  its queries. Other modality's attention remains unchanged. Weights, QKV,
  residual paths, MLP, and eval-mode dropout are unchanged.
- Record all per-tuple task loss, IoU and output equality. Bootstrap by
  sequence,10000 draws. Cells are exploratory, not multiplicity-adjusted.

Advance only to a separate closed-loop train-only test if both search-missing
directions show at least0.01 sequence-macro IoU improvement masked-vs-sham
with positive95% lower bounds, sham matches native exactly, and complete
search outputs are bitwise equal to native. A pass does not authorize
training or official evaluation by itself. A failed gate closes this exact
K/V hypothesis on this diagnostic cohort; no threshold tuning on it.

Expected under2minutes on GPU2. Check occupancy first. Preserve all caches
and checkpoints; do not fill VRAM with duplicated tuples.

## Execution and disposition

Server28 GPU2 was idle at launch (1 MiB used, 38 C, 250 W power limit).
The single-batch attention witness passed: custom all-key attention exactly
matched native score, size, offset, and box outputs.

The first full attempt (PID53510) stopped before producing metrics because
the sampled target boxes remained on CPU while predictions were on GPU.
The traceback is retained in
`reports/server28_absent_search_kv_20261102_v1.log`. The diagnostic now moves
targets to the prediction device. The failed output directory is preserved
on server28; no weights or runtime files were changed.

The corrected run (PID65200) completed in80.864seconds using128 paired
tuples,1152 rows, and113 distinct train882 sequences. Its report SHA256 is
`192adfe78f833b1764c9ed49410449fd9348994ec33202b3b1a154142fa6b503`.
Independent saved-evidence verification recomputed the report digest,
all source hashes, tuple/state counts, exact exclusion of prior pilot tuples,
all native/sham outputs, complete-state parity, and the sequence bootstrap.
All1152 sham/native output comparisons were bitwise equal; all384 complete
search rows preserved native output. The baseline model state digest was
unchanged and parameter gradients were absent.

Positive IoU delta means masked minus native IoU. Pooling the three dynamic
template states, search-missing RGB (`01`) changed macro IoU by
`-0.182255` (95% CI `[-0.221629,-0.143620]`); search-missing TIR (`10`)
changed it by `-0.062795` (95% CI `[-0.088324,-0.038894]`). Both effects
are negative, so the registered `+0.01` improvement gate failed. Close this
K/V-masking hypothesis on this cohort; do not train, tune, or run official
evaluation for it. This is a frozen single-crop train882 diagnostic, not an
official metric, closed-loop tracking result, or paper-module contribution.

Full report and completion receipt:
`reports/server28_absent_search_kv_20261102_v2/`.
Launch receipts and logs for the witness, failed attempt, and completed run
are retained in the sibling `reports/` files. Independent verifier:
`scripts/analyze_absent_search_kv.py --verify-source-hashes`.
No val97, official test data, optimizer, or checkpoint cleanup was used.
