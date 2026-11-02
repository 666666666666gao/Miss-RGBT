# F Structural Prototype

Implemented independently in research/feedback_f/model.py on November 2,
2026. The current seed2028 trainers, frozen runtime and registered official
queue are unchanged. This is a structural prototype with synthetic checks,
not an image-trained candidate or a validated paper module.

## Computation

FeedbackTrackerModel subclasses MissingTrackerModel without editing it.
Each search is embedded with the native availability/patch mask semantics.
Blocks1-3 execute once; their full token states are saved.
The first pass completes blocks4-12, including A Utility repair at block6,
and obtains the native CENTER prediction.

A tokenwise residual consumes normalized shallow RGB/TIR features,
normalized first-pass deep RGB/TIR features, score/size/offset predictions
and actual availability. Its last linear layer starts at zero.
Only search tokens receive residuals. The saved query/template prefix is
unchanged at the second-pass entry. Blocks4-12 and block6 repair then run
again with shared parameters. The inherited forward returns round2 maps
and nests round1 maps under `round1`.

No complete-view features are passed into the masked-view call.
Feedback is allowed for complete input as well as missing input.
The two-round helper averages the existing C/M task-and-KD objective across
rounds; it does not double its coefficient. Existing utility probe
supervision can consume round2 outputs. Its probes compare repair off/on
with the already constructed second-pass input, not a separate intervention
that recomputes round1 feedback.

Modes:

| Mode | Residual source | Backbone passes |
|---|---|---|
| none | No residual module | Blocks1-3 once; blocks4-12 twice |
| shallow | Shallow features and availability; deep/prediction inputs zeroed | Same |
| deep | Shallow/deep features, prediction and availability | Same |

Shallow/deep have equal residual parameter shapes and perform the same
linear layers. The no-residual control lacks that small module.
Use a separate ordinary single-pass model as the efficiency reference.
The current prototype explicitly excludes random region selection because
its two-round RNG semantics have not been specified. Utility selection
matches the proposed F starting architecture.

## Verified Scope

research/feedback_f/check_structure.py runs synthetic tensors with actual
E0 weights on server28 GPU0, PyTorch1.12.1+cu113, RTX3090. It uses two
128x128 templates and 256x256 searches. The gradient check includes three
searches and both C/M calls, frozen BN, normalized dual-round loss and an
existing utility counterfactual probe.
The repair projection is perturbed away from zero in these checks so that
accidentally skipping repair cannot pass identity through a zero repair.
This is synthetic initialized repair, not a learned A checkpoint.

Current receipt: reports/server28_feedback_f_structure_v3_20261102.json.
Earlier receipts retain their earlier checker-source hashes and are not
evidence for checks added later.

Checks:

- Exact native/round1/round2 parity at zero residual for states11/01/10.
- Blocks1-3 called once and blocks4-12 twice; repair called twice.
- Zero-residual shallow and no-residual branches align with native output.
- Zero-initialized residual projection has finite, nonzero task gradient.
- Nonzero feedback preserves the query/template prefix at insertion,
  changes search tokens and can change the complete-input prediction.
- With nonzero projection, the shallow control ignores changes to deep
  features and predictions; deep feedback responds to each source.
- Changing physically unavailable RGB/TIR values in a later template and
  search leaves both rounds exactly unchanged.
- Changing a masked image patch leaves both rounds exactly unchanged.
- Backbone, feedback and utility selector receive finite nonzero gradients;
  an optimizer step changes the residual projection.
- BN buffers remain exactly fixed during the three-search training check.
- A separately constructed model strictly reloads the saved state and
  reproduces predictions exactly.

This is an implementation smoke check with one optimizer update, not a
training experiment. It does not measure real-image loss improvement,
convergence, correction/destruction rates, end-to-end tracking latency or
official PR/SR/MPR/MSR.
No benchmark images, train882 frame identities, official evaluation,
checkpoint pruning or external notifications were used.
The check's internal saved model is in memory and is not a promoted weight.

## Remaining Work

Preserve P then F then X experiment order. P still needs terminal original
manifests, metadata feasibility, actual image smoke and formal training.
Before image training/evaluation of F, provide the isolated formal trainer
and checkpoint/tracker loading integration, verify one external tracker
state/template update per frame, and compare single-pass/no-residual/
shallow/deep with matched objective and training budgets.
Measure latency and peak memory on the same actual workload and record
correction as well as destruction. Do not infer any of these from block
counts or this synthetic structural check.
