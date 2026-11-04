# Read-Only M1 Decision Gate

`research.feedback_f.protocol_m1.verify_decision(report, cache_receipt, config)`
checks saved M1 evidence before a future F preparation step. It does not
launch jobs, write campaign files, select P*, or authorize F training.

Missing report, terminal receipt, cache receipt or config returns
`WAITING_FOR_M1` with `ready=false`, without loading the image cache. Invalid
existing evidence raises the existing verification errors; it is not
converted into a negative diagnostic.

For present evidence the gate reuses the existing diagnostic cache loader,
checkpoint identity, sequence-cluster bootstrap and retry checker. It checks
the report terminal hash, actual checkpoint hash, config/source hashes,
cache identity, parity/state/gradient flags, ordered sample/state rows and
different-sequence donors, then recomputes effects and the registered M1
decision. An embedding diagnostic must retain the completed negative
block-three predecessor required by the existing retry checker.

The four outcomes are:

| Decision | M1 ready | Next dependency |
| --- | --- | --- |
| accepted_block3 | true | Accepted-OHCAS composition |
| retry_embedding | false | Registered embedding diagnostic |
| accepted_embedding | true | Accepted-OHCAS composition |
| closed_both_diagnostics_negative | true | Continue without accepted OHCAS |

`ready` concerns only this M1 decision. Every outcome retains
`f_training_authorized=false`. P* selection, accepted-OHCAS composition
when applicable, a frozen five-arm F package and five matched real-image
smokes remain caller-owned prerequisites. Saved flags do not independently
replay the original model forwards or prove efficacy.

## CPU Verification

Six unittest methods passed in the server's existing Python 3.7 environment
on 2026-11-04, including a repeat run at approximately10:57UTC. Fixtures cover
all four decisions, missing evidence, propagated cache failure, changed
decisions/rows/parity, source/checkpoint/terminal hashes, and a skipped or
positive block-three predecessor. The bootstrap and retry helpers are real;
the image-cache loader is explicitly mocked and checkpoint constants point
to fixture bytes. The preceding integrated harness confirmed CUDA remained
uninitialized. No real-image M1 result or cache freshness is established.

Only the two new files were staged under
`/tmp/codex_feedback_m1_20261104.Xt0jWT/research/feedback_f/`.
Their SHA256 values match the local files:

| File | SHA256 |
| --- | --- |
| protocol_m1.py | faa81565800cb530347756758dff8483c4ee6ae81b1d7d1f6d306158a16fcf02 |
| check_protocol_m1.py | 5a89d1865d8e2c7ee234f462d61ae5b39aded09cc61a3013c7655cd35f2a3984 |

The readiness probe for
`reports/server28_history_stream_20261103/report.json` and its sibling
`cache_receipt.json`, using the registered batch10 Utility seed2027 config,
returned `WAITING_FOR_M1`: report, completed receipt and cache receipt were
absent. This is a probe of those paths, not an exhaustive cache inventory.
Original R historical-frame provenance remains uncertified. This code does
not repair that data gate, and missing provenance cannot close M1 negatively.

The active server project and frozen R/P runtimes were not changed. The
server-root F package remains the historical four-arm version; this
staging check is not a whole-package deployment. No fresh independent
review was performed, so deployment acceptance is not claimed.
