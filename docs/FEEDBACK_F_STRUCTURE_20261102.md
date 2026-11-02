# F Structural Prototype

Latest operational addition: `prepare.py --curriculum-arm progressive`
(or fixed60/fixed50/shuffled) prepares four matched real-image smoke
configs after the existing formal P verification boundary. It does not
launch jobs. Each arm uses20 short epochs,8 samples/epoch,microbatch2,
accumulation2,40 updates,LR5e-5/backbone5e-6 and8 development samples.
The curriculum is an explicit shared input, not an official-score choice.
`check_prepare.py` passed on server Python3.7 with temporary synthetic
metadata for all four curricula and rejected missing formal P before
creating output. Receipt: reports/server28_feedback_f_prepare_20261102.json.
No real F configs/images or formal verification/queue were produced.
Earlier sections below retain the scope at their original recording time.

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

## Tracker Integration Check

`research/feedback_f/tracker.py` adds strict loading for an F checkpoint and
inherits the native `initialize` and `track` methods. It does not fork the
crop, Hann window, box decode, confidence threshold or template-update logic.
`check_tracker.py` uses synthetic uint8 frames and the actual E0 network.
The current receipt is
`reports/server28_feedback_f_tracker_20261102.json`, SHA256
`3b8d6e559b13db3ca43666c45556212340274969ba08e2ad6828c1d2897637f4`.

The check passed strict loading for `none`, `shallow` and `deep`; exact
zero-residual native parity over availability states 11, 01 and 10 across
three frames; one network call and one box/state commit per frame; round2
prediction used for decode; and conditional template append/pop exactly
matching the inherited update rule. This remains synthetic integration
evidence, not an image-trained or benchmark result.

## Shared Training Batch Interface

`training.py` provides the single/none/shallow/deep model factory and shared
C/M batch objective. It strictly copies the native initialized Utility state;
extra module construction preserves the caller's CPU RNG. Two-round task/KD
supervision is averaged over rounds; the existing Utility probe supervises
round2 once. Reverse KD stays disabled. No datasets or jobs are opened.

Actual E0 synthetic check: `reports/server28_feedback_f_training_20261102.json`
(SHA256 `455379b2f90fb0e612acacc0dd1ddf231898e222af57ff9814e4db74886afc62`).
All four arms have exactly equal shared initial state and post-build CPU RNG.
C and M probe cases each reconstruct the objective exactly, perform finite
backward/AdamW updates, update backbone and Utility selector, and preserve BN.
Shallow/deep feedback projections receive nonzero gradients. Eight optimizer
updates took 19.424 seconds; peak allocated memory was 8,434,151,424 bytes.
This is synthetic optimizer evidence, not real-image training, real batch
memory sizing, convergence or efficacy. No persistent weights were produced.

`step_masks` now isolates native CUDA mask draws using `mask_seed + microstep`
and restores the caller's CPU/CUDA RNG. This handles different stochastic
forward draws between single and double passes; an initial seed alone was
insufficient. The future formal loop must call this helper and verify actual
image/input/mask hashes across arms. Training-mode loss equality between
single and double passes is not asserted; zero-residual exact parity is an
evaluation-mode check.

The v2 actual-network synthetic receipt is
`reports/server28_feedback_f_training_v2_20261102.json`, SHA256
`0c4052c1c9eb833114c1522c9610be2931abf5562491882b42e30306b3e14d30`.
It checks native sampler parity at the same step seed, exact caller CPU/CUDA
RNG restoration, different patch draws on consecutive steps, and equal
mask hashes across all four arms for two steps despite their different
forward paths. These masks are used for the eight C/M probe optimizer
updates, with exact objective reconstruction, finite gradients, parameter
updates and frozen BN. Duration19.253s, peak8434151424bytes. V1 retains
its original source hashes. Neither receipt is real-image training evidence.

## Isolated Image Training Entry

`train.py` now supplies the image-training loop for all four F controls.
It reuses the original partitioned loader, frame/tensor identity records,
train882-only development evaluation and frozen BN. All arms use the same
configured P schedule, sample/update budget, per-step mask seeds and probe
sequence. Their learning rate, microbatch and accumulation must be matched
when preparing the four-arm campaign; batch sizing remains unmeasured.
The entry records config/source/initial shared parameter/BN hashes, masks,
losses, gradient norms, learning rates, development history and checkpoints.
It completes the configured 20-epoch P course, retaining every checkpoint;
the minimum development loss within epochs16-20 selects one checkpoint
(earlier tie). It does not select or delete weights using official tests.

Before dataset access or output-directory creation, the entry requires the
full formal P verification report, unchanged verifier/prepared/config hashes,
all four P completion/manifest hashes, and inherited prior completion and
frame-exclusion boundaries. This preserves the P->F->X experiment order.
No F configs, campaign or image workers have been created.

The actual Python3.7 compile/import/CLI and missing-parent rejection check
passed; receipt `reports/server28_feedback_f_train_entry_20261102.json`,
SHA256 `054f3e7874460bbadfd083ac7cfce28108af19464eebf60a71ed293301920736`.
This check confirmed the absent formal P verification causes rejection
without creating an F run directory. It did not execute the loader/epoch/
checkpoint loop or verify the positive-parent path. The earlier optimizer
receipts cover synthetic batch updates only, not this image loop.

`check_parent_contracts.py` also exercises the parent validator against
temporary synthetic metadata: a complete fixture passes, and twelve cases
reject missing verification, wrong phase, missing arm, changed verifier/
prepared/config/run-config files, incomplete parent status (even with updated
hashes), changed manifest, omitted inherited completion/manifest and wrong
verification hash. No GPU, dataset or F run directory is used.
Receipt `reports/server28_feedback_f_parent_contracts_20261102.json`,
SHA256 `8db6fd81b56defe6fae71ce38bae2fdd333cadecbc1ef0bcd6f4f891b3569d44`.
These are metadata contract checks, not validation against real P receipts
or execution of the image-training loop.

## Internal Round-Loss Diagnostics

`validation.py` preserves the original final-round task-loss aggregation and
adds paired round1/round2 losses for F controls, separately for 11/01/10.
For each input tuple it sums task loss over all search frames, then reports
strictly lower/higher/equal loss counts, fractions and mean delta. This does
not measure whether a closed-loop tracker corrected or destroyed a frame.
Single-pass controls keep only the original final-loss fields.

The trainer writes raw vectors to `development_round_samples_epochNN.json`.
The batch index and vector position correspond to the development entries
already recorded in `input_manifest.jsonl`. `development_history.json` keeps
the summary; checkpoint selection still uses final-round mean task loss.
No threshold, selection rule or training objective changed.

Actual E0 synthetic checker `check_validation.py` passes: final loss matches
the original validator exactly for single/none/deep, zero-feedback rounds
tie exactly, CPU/CUDA RNG and selector state are preserved, and nonzero
feedback raw losses/counts/deltas match independent recomputation from
captured network outputs. Its receipt is
`reports/server28_feedback_f_validation_20261102.json`, SHA256
`a32207d49dae468c58a4d9c4f3975774742e9397d58c44d5c74501abea39a665`.
The updated train CLI imports in Python3.7. No real image/epoch, learned F
weight, actual tracking correction/destruction rate or official score was
produced. Earlier receipts retain their historical source hashes.

## Full Official Evaluation Entry

`evaluate.py` provides complete/miss evaluation for single/none/shallow/deep.
It reuses the frozen evaluator's `prepare` and `sequence` functions, including
GT/mask hashes, first-frame initialization, both-missing hold and all-frame
output. Its initializer loads FeedbackTracker and assigns the same sequence
runner globals in each spawned worker. Full inventory counts are fixed at
479 sequences / 337353 frames; the CLI has no subset/limit argument.
It uses the same rgbt metric classes and saves five scores and curves,
all trajectory receipts, checkpoint hash and both native/F source hashes.
No original evaluator or running queue was edited.

`verify_official.py` independently recomputes scores/curves from all saved
trajectories, checks every individual receipt against the aggregate, verifies
all GT/mask hashes, model/F/native/runtime source hashes and full coverage.
The original verifier hardcodes the original evaluator's source hash, so it
cannot validate this F entry unchanged. This separate verifier identifies the
actual F entry instead; it does not call any development evaluation main.
It is not yet integrated with a verified F campaign or final-five queue.

Fixed a concrete loading mismatch: single-arm training saves native
MissingTrackerModel weights with feedback mode `single`; the earlier F loader
only constructed a two-round FeedbackTrackerModel and would reject that mode.
The loader now explicitly constructs the native model for `single` and
strictly loads its weights. The updated actual E0 synthetic tracker check
confirms single/native boxes, templates and availability match over three
frames/states, and preserves the existing two-round checks. Receipt:
`reports/server28_feedback_f_tracker_v2_20261102.json`, SHA256
`4aab14a1ded19e1d7b71bc176e1c73aed72e3159c563c0cbcaf7b69b4839a468`.

Actual Python3.7 compile/import/CLI help pass for both new entries; `--limit`
is rejected before dataset access/output creation. Receipt:
`reports/server28_feedback_f_official_entry_20261102.json`, SHA256
`d4c1df7722c0a378ffa0b8e4a715e2d29c8f2c0cfa0e91f01ee9dc783d1804bb`.
Neither spawned evaluation, actual F trajectory verification nor any official
benchmark was run. These are implementation checks only.

## Remaining Work

Preserve P then F then X experiment order. P still needs terminal original
manifests, metadata feasibility, actual image smoke and formal training.
Before image training/evaluation of F, complete positive-parent validation,
prepare matched configs, measure batch memory and perform image smoke plus
saved-evidence verification of the new training loop. Synthetic tracker
loading and one external state/template update have passed. Compare single-pass/no-residual/
shallow/deep with matched objective and training budgets.
Measure latency and peak memory on the same actual workload and record
correction as well as destruction. Do not infer any of these from block
counts or this synthetic structural check.
