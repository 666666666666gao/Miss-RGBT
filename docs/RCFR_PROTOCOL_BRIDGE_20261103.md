# RCFR OSTrack/RPPM batch bridge

## Implemented

`research/feedback_f/protocol_training.py` provides a construction and
microbatch interface for `single`, `none`, `shallow`, and `deep`.
It does not launch jobs or authorize an experiment.

Construction uses the new P OSTrack initializer and native Utility construction.
The initializer SHA256 is checked. Extra feedback construction preserves CPU
RNG; shared backbone/head/Utility parameters and buffers are identical across
the four arms. Returned initialization evidence describes these shared native
parameters; it is not a complete feedback parameter manifest.

All six P curricula are supported. C runs first; its final-round task loss
updates the complete-loss EMA and supplies pacing decisions. Mask draws use
`mask_seed + microstep` inside a CPU/current-CUDA RNG fork. Independent pacing
generator state stays in `CurriculumState`. Equal hard flags and generator
states give paired masks despite extra model forwards. Once learned C losses
differ, paced hard flags and therefore availability may legitimately differ.
Patch draws remain paired. This keyed-mask protocol differs from the current
P trainer's global RNG stream: F single must be retrained as its matched
control, not substituted with an existing P score.

For two-pass arms the objective is the mean of round1/round2 C/M objectives.
With normalization off it calls the existing native objective for each round.
For `paced+norm`, the state EMA is updated once from all round/search/sample
observations; both rounds use the same updated denominator. This is a new,
explicit F normalization convention. It avoids updating EMA twice or silently
doubling the missing-task loss scale. Single-pass normalization retains the
one-round convention. C/M outputs are returned for caller-owned Utility
supervision on the final round; probe routing adds no extra branch forward.
Optimization, accumulation, Utility loss weighting and checkpoint writing
remain the caller's responsibility.

## Executed Check

On server28's existing Python3.7/Torch1.12 environment, from
`/home/gaob/Miss-RGBT/runtime/task_utility_20261018`:

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
PYTHONPATH=/home/gaob/Miss-RGBT/runtime/task_utility_20261018:/home/gaob/Miss-RGBT \
/home/gaob/Miss-RGBT/conda-envs/miss_rgbt/bin/python -u \
-m research.feedback_f.check_protocol_training \
--initializer /home/gaob/rgbx-initializers/OSTrack_ep0300.pth.tar \
--output /home/gaob/Miss-RGBT/reports/server28_feedback_protocol_training_checks_20261103.json
```

PASS: 72 synthetic CPU batch cases (6 curricula x 4 F modes x epochs1/6/16),
independent loss/gradient/state-EMA calculations, six keyed-mask/RNG and
checkpoint-state roundtrips, final-round probe routing, and four actual
OSTrack CPU constructions with identical shared state and post-build CPU RNG.
CUDA remained uninitialized. The report binds the new checker/bridge and
existing F model sources; it is not a complete training-runtime manifest.

There was no actual-network forward/backward, image loading, GPU smoke,
full training, tracking evaluation, FPS measurement or efficacy evidence.
The test model is deliberately small and exercises the batch objective only.
No current P/R source or frozen training runtime was changed.

## Fixed Development Validation Interface

`research/feedback_f/protocol_validation.py::validate` accepts the four F
modes explicitly and requires all 128 cached development tuples, with two
templates and three searches per tuple. Model selection uses final-round
native task loss averaged across complete, RGB-missing and TIR-missing states.
This is internal selection evidence, never an official benchmark result.

For two-pass arms it saves both rounds' per-tuple task losses and per-search,
per-sample crop IoU. Boxes use the tracker's Hann-windowed native CENTER
decoder, normalized crop coordinates, and unclipped prediction/GT. Success
means IoU >= 0.5. Reports include corrected/destroyed counts, fractions over
all 384 searches per state, conditional rates over originally failed/successful
crops, higher/lower/equal IoU counts and mean IoU changes. A conditional rate
with no eligible crops is null. No closed-loop tracking, template updates or
full-image clipping is represented.

Validation preserves the Utility selector buffer, torch CPU/current-CUDA RNG,
and each module's training flag, including frozen batch-normalization modules.

Executed on server28 in the same CPU-only environment:

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
OPENBLAS_NUM_THREADS=1 \
PYTHONPATH=/home/gaob/Miss-RGBT/runtime/task_utility_20261018:/home/gaob/Miss-RGBT \
/home/gaob/Miss-RGBT/conda-envs/miss_rgbt/bin/python -u \
-m research.feedback_f.check_protocol_validation \
--initializer /home/gaob/rgbx-initializers/OSTrack_ep0300.pth.tar \
--output /home/gaob/Miss-RGBT/reports/server28_feedback_protocol_validation_checks_20261103.json
```

PASS: synthetic 128-tuple diagnostics in all four modes and three availability
states; independent final-loss calculation; known IoU correction/destruction
counts; threshold-boundary and zero-denominator cases; Hann decoding with raw
argmax deliberately different; parameter/BN-buffer, selector, RNG and module-mode
preservation. Actual OSTrack CPU forwards on synthetic full-size inputs also
passed in all four modes and three states with three searches each. Zero-init
two-pass predictions matched round1 and the shared single-pass reference.
CUDA remained uninitialized. The report records source hashes.

This check did not use real images or learned feedback, perform backward or
GPU smoke, evaluate a val97 model, measure FPS, or produce official metrics.
It establishes diagnostic arithmetic and native forward compatibility only.

## Isolated Training Loop

`research/feedback_f/protocol_runner.py` provides a library loop for the new
OSTrack protocol. It is not a CLI or campaign authorization. Its caller must
verify the selected P curriculum, accepted or closed M1 decision, source
snapshot and, for formal training, all four matched real-image smokes.
The retired P evidence checks must not be passed unchanged as F checks.

The loop uses microbatch2/accumulation2 and native AdamW/clipping, 500-update
warmup, epoch10 LR drop, and a final-round Utility probe every four
microbatches, alternating C/M and cycling the three searches. It records
accepted tuple identities, tensor/mask hashes, shared and feedback-specific
initialization, source manifest and checkpoint RNG state. Smoke mode stops
after four optimizer updates and checks fresh CPU strict loading plus
bitwise complete-view predictions for every round/search on the live model.
That reload check is not an all-availability-state or resume check.

Formal mode performs 20 full epochs, saves epoch10 and16-20, writes complete
internal round-diagnostic sidecars and selects among the final five by fixed
internal mean loss, with the earlier epoch winning a tie. Its completion
receipt explicitly does not establish official evaluation completion.
No resume, early stopping, pruning or active P/R source edits are introduced.

`check_protocol_runner.py` compares the CPU synthetic step against a separate
explicit optimizer loop: 6 curricula x 4 feedback modes, 24 microsteps and
12 optimizer updates per case. It checks accumulation boundaries, exact
model/AdamW/curriculum/RNG state, gradients, masks, alternating probe views
for all three searches, learning-rate warmup/drop and detached all-round
prediction snapshots. Synthetic counterfactuals have a known zero Utility
target. This does not establish actual-network backward, real image
smokes, checkpoint reload, full-run correctness, learned feedback or scores.

Executed on server28 in the existing Python3.7/Torch1.12 environment:

```bash
cd /home/gaob/Miss-RGBT/runtime/task_utility_20261018
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
OPENBLAS_NUM_THREADS=1 \
PYTHONPATH=/home/gaob/Miss-RGBT/runtime/task_utility_20261018:/home/gaob/Miss-RGBT \
/home/gaob/Miss-RGBT/conda-envs/miss_rgbt/bin/python -u \
-m research.feedback_f.check_protocol_runner \
--output /home/gaob/Miss-RGBT/reports/server28_feedback_protocol_runner_checks_20261103.json
```

PASS, exit0: 24 mode/curriculum cases, 576 microsteps, 288 updates, five
learning-rate boundary cases and prediction snapshots in all four modes.
CUDA remained uninitialized. The report SHA256 is
`7a4940cc8d8458a42248f2a3215c37b3dfb3d544c6a4c209523ae6169290b427`;
it binds the checker/runner/bridge/fixture source hashes, not an entire
training snapshot. No F training was launched.

## Saved-Evidence Verifier

`research/feedback_f/protocol_verify.py` adds a library verifier for this
training loop. `verify_run(config, verify_evidence, verify_smokes)` requires
the same caller-owned F parent/source and four-smoke checks. It audits
20 epochs, all accepted train882 tuple identities, input/mask hashes,
round-specific objective terms, Utility schedule, accumulation and learning
rates. It retains the native RNG/optimizer/BN/native-parameter checks for
epoch10 and16-20 and checks feedback shape/state changes for shallow/deep.
Single/none must have no feedback parameters.

Every internal diagnostic sidecar is hash-bound to its history row. The
verifier recomputes correction/destruction counts and rates from all384
crop IoUs per availability state, checks all128 final-round sample losses
against the selected-loss summary and rejects missing searches/tuples.
`verify_pairs(configs)` audits four matched configurations, shared native
initialization/data, shallow/deep feedback initialization and per-step masks.
Paced modality draws may differ with actual model-dependent difficulty;
patch masks and equal-probability control masks must match.

Executed in the existing server28 CPU environment:

```bash
cd /home/gaob/Miss-RGBT/runtime/task_utility_20261018
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
OPENBLAS_NUM_THREADS=1 \
PYTHONPATH=/home/gaob/Miss-RGBT/runtime/task_utility_20261018:/home/gaob/Miss-RGBT \
/home/gaob/Miss-RGBT/conda-envs/miss_rgbt/bin/python -u \
-m research.feedback_f.check_protocol_verify \
--output /home/gaob/Miss-RGBT/reports/server28_feedback_protocol_verify_checks_20261103.json
```

PASS, exit0: 24 curriculum/mode cases,576 actual synthetic-loop step records;
four diagnostic modes including JSON round trips and feedback-weight fixtures.
Checks reject incorrect mode/round count/objective/normalization counts,
nonfinite rounds, altered summary/crossing counts, invalid IoU, missing
searches/tuples and wrong batch indices. CUDA remained uninitialized.
Report SHA256:
`543c92ea6db3b69b38a1334d76f35d7ce3b645a85ed1e01c46992f859f0fa236`.

This is synthetic checker evidence. The full `verify_run` orchestration,
actual saved checkpoints/native optimizer audit, caller parent/smoke gates
and four full-run pairing have not been executed. There is no F CLI or
authorized campaign yet, no learned feedback/FPS evidence and no new score.

## Read-Only P Parent Gate

`research/feedback_f/protocol_parent.py` is a read-only CLI for the new
six-arm campaign. It requires queue completion, all six saved-training
verifications, paired-training verification and all sixty full official
evaluations before returning `P_PARENT_VERIFIED`. It binds parent/source
hashes, checks the final-five internal-loss selection for every arm,
recomputes every full official result with the existing official verifier,
and reruns six-arm pairing verification. Missing prerequisites return
`WAITING_FOR_P` without constructing a model or creating a campaign.

The plans request the best P curriculum without defining cross-arm
selection. Before any P result exists, this implementation specifies:
choose the arm with the lowest internally selected final-five mean native
task loss. Epoch ties choose the earlier epoch; arm ties choose the
registered `ARMS` order. Official Complete/Miss scores do not select an
arm or checkpoint. The selected Complete/Miss metrics use that same
checkpoint. This defines a parent candidate, not evidence that it passes
the plans' module-promotion comparisons.

Executed on server28:

```bash
cd /home/gaob/Miss-RGBT/runtime/task_utility_20261018
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
OPENBLAS_NUM_THREADS=1 \
PYTHONPATH=/home/gaob/Miss-RGBT/runtime/task_utility_20261018:/home/gaob/Miss-RGBT \
/home/gaob/Miss-RGBT/conda-envs/miss_rgbt/bin/python -u \
-m research.feedback_f.check_protocol_parent \
--output /home/gaob/Miss-RGBT/reports/server28_feedback_protocol_parent_checks_20261103.json
```

PASS, exit0: six synthetic arms and sixty stubbed official evaluations;
fixed50 wins an internal-loss tie with progressive, and epoch16 wins the
epoch tie despite higher synthetic test scores later. Twelve altered
scope/selection/checkpoint/output/evidence cases are rejected. The official
and paired verifiers are explicitly stubbed in this fixture. CUDA remained
uninitialized. This checks the outer gate only, not actual full-run or
official-result verification.

The production invocation replaces the module with
`research.feedback_f.protocol_parent` and omits `--output`. It executed
successfully and returned `WAITING_FOR_P` with twenty missing prerequisites.
No F configs, image smokes or training jobs were created. The gate always
returns `f_training_authorized=false`, even after P succeeds: an actual
accepted/closed M1 decision, accepted-OHCAS composition when applicable,
frozen F preparation and four matched real-image smokes remain necessary.

## Remaining Dependencies

The old `feedback_f/prepare.py` and `train.py` still bind the retired four-arm
P campaign, E0 continuation and partitioned data. They cannot consume the new
six-arm campaign by changing a path. This bridge does not repair them.

Next work requires a coherent new F preparation/evaluator/dispatcher
path using the selected P curriculum and the accepted or closed M1 decision.
The isolated training loop and verifier require those caller-owned gates before use.
It must preserve native train882/fixed val97 selection, common data identities,
20 full epochs without early stopping, epoch10 and16-20 retention, four real
image smokes, strict model reload, and ten full official evaluations per arm.
Every evaluation must verify479 sequences/337353frames. Full final metrics
must use one internally selected checkpoint/config; no test-selected maxima.

New F runs use microbatch2/accumulation2, effective batch4, LR5e-5 and
backbone LR5e-6 as previously profiled, subject to real image smoke. Learned
round IoU correction/destruction evidence and matched latency remain required;
the new diagnostic interface alone supplies neither.
OHCAS's unresolved historical R frame-provenance/data gate is unchanged.
The persistent P queue does not dispatch F or OHCAS.

## Four-Arm Official Reporting

`protocol_official.verify_campaign` now joins the four completed single-arm
controllers. The caller must provide the same actual parent/source and
matched real-image smoke checks required by `protocol_runner.run`; this is
a library entry point, not a new training CLI or M1 authorization.

It reruns the four saved-training verifiers and full-run pairing verifier.
For every controller it checks the original frozen source/training/checkpoint
hashes, exact ten-job registration, successful subprocess exit records,
saved metrics and terminal receipt. It independently recalculates every
one of the forty complete-dataset results through the existing F verifier,
including the 245/220703 LasHeR and 234/116650 RGBT234 coverage checks.
Every Complete/Miss pair must use the same checkpoint.

Only after that check can `write_campaign_reports` create a fresh output
directory containing `completed.json`, `all_epochs.csv` (twenty rows) and
`selected_epochs.csv` (four rows). Both CSVs carry all ten official metrics
per row. The latter preserves each arm's internal-development selection;
larger official scores in other epochs do not alter the selection. This
does not decide which module or final model deserves promotion.

The server's existing Python 3.7 environment ran eight unittest methods
successfully, with CUDA uninitialized. The new campaign fixture exercises
forty mocked evaluations and rejects missing results, test-based selection,
altered metrics, wrong checkpoint hashes, reordered jobs, nonzero exits,
changed frozen sources and incomplete dataset coverage. Existing single-arm
failure/source checks and all 240 curriculum/mode job plans still pass.
The fixture training and pairing verifiers are mocked: no actual F campaign,
image smoke, learned weight, real official score or M1 decision is established.

Raw check receipt:
`reports/server28_rcfr_campaign_reporting_20261103.json`, SHA256
`7c06dcc3dcb8a37d6c0b9020cca6dd8391a4eed25dd2f25b361a10675e5eccb8`.
Frozen R/P sources and their active processes were not changed.

## Five-Arm Amendment

The preceding four-arm reporting and test counts are historical evidence for
the earlier protocol. The current isolated F protocol adds
`deep-no-response`, retaining both shallow and deep features while zeroing
the first-pass response-map input. This separates the planned deep-feature
contrast from the response-conditioning contrast. The current mode list is
`single`, `none`, `shallow`, `deep-no-response`, `deep`; reporting now
requires five arms, 50 full evaluations and 25 final-five epoch rows for one
selected curriculum. It does not launch or authorize an F campaign.

See `docs/FEEDBACK_F_FIVE_ARM_RESPONSE_CONTROL_20261103.md` and the raw
checks under `reports/server28_feedback_five_arm_checks_20261103/` for the
amendment rationale, current CPU verification and scope limits.

## Read-Only M1 Decision Verification

`protocol_m1.verify_decision` now supplies a separate saved-evidence M1 gate.
It recomputes the existing diagnostic decision and checks its terminal,
cache, source, checkpoint and retry bindings. It never authorizes F training.
Six CPU fixture tests pass with the image-cache loader explicitly mocked.
See `docs/FEEDBACK_F_M1_DECISION_GATE_20261104.md` for exact scope and staging
identities. Original R historical-frame provenance, actual M1 evidence,
P* selection, accepted-OHCAS composition where applicable and five real-image
smokes remain unresolved prerequisites for future F preparation.
