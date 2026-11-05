# Partitioned Training Fresh-Frame Feasibility

Read-only CPU metadata audit of the705 LasHeR fitting sequences. It uses
only completed historical diagnostic/crop exclusions from the existing
`exclusions()` chain and completed dense-geometry cache identities. It
deliberately excludes the active partitioned-A smoke/formal manifests,
which were still being written. No images, model, GPU, sampler workers or
training/evaluation processes were started by the audit.

The count follows the current native causal sampler. A sequence is eligible
only when length>=20 and visible-frame count>2*(2 templates+3 searches).
The anchor is sampled from `[1, length-3)`; at least one earlier visible
frame and three later visible frames must exist. The native sampler expands
its temporal gap without a fixed cap, so any such earlier/later frames are
eventually reachable. Actual transformed-crop acceptance is not measured.

| Finding | Count |
|---|---:|
| Fitting sequences | 705 |
| Total possible fresh causal anchors | 296,231 |
| Sequences with at least one anchor | 704 |
| Sequences with zero anchors | 1 |
| Median anchors per sequence | 229 |
| Minimum positive anchors | 7 |
| Sequences with fewer than 20 anchors | 7 |
| Sequences below the sampler's visible-frame threshold | 1 |

`aboyleft_1202` has71 frames,5 visible after66 historical exclusions, and
0 legal anchors. This is a concrete sequence-level sampling concern, but
does not mean the full training set is infeasible: 704 sequences remain
eligible and collectively provide296,231 possible anchors before active-A
frame exclusions.

This is an upper bound for the next P run, not its final feasibility result.
The two running A formal input manifests are not included and will add
exclusions. Re-run this audit only after both A manifests and completion
receipts are frozen and
`reports/server28_partitioned_a_20261102_formal/verification_v2.json` has
`status=passed`. Then run `audit_partitioned_fresh_frames.py` with
`--include-completed-partitioned-a` and a new output path. That option checks
both manifest/completion hashes against the paired verifier before reading
the immutable identities. Do not launch P smoke based on this report alone.
The audit creates no
training run or `input_manifest.jsonl`; its only output is the JSON report.

Raw output:
`reports/server28_partitioned_fresh_frame_audit_v3_20261102.json`.
Checker:
`scripts/check_partitioned_fresh_frames.py`; it checks native sampler
boundaries, strict visibility threshold, short-sequence rejection and
exclusion-induced zero-anchor fixtures, plus accepts only paired completed-A
verification receipts with matching completion/manifest hashes. The report
hashes its audit, sampler, dataset and training-source files and all
historical exclusion reports.
