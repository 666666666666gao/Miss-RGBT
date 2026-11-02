# Curriculum P and Feedback F

User proposal received November 2, 2026. This document registers the next
implementation direction, not a successful module or a formal training launch.
Keep the seed2028 partitioned A trainers and official queue unchanged.

## P: Missing-Frequency Curriculum

Keep the A Utility model, initialization, objective, BN policy, patch masking,
optimizer and effective batch fixed. Change only whole-modality frequency.
Use four arms with twenty full epochs each:

| Arm | Epochs 1-5 | Epochs 6-15 | Epochs 16-20 |
|---|---:|---:|---:|
| fixed60 | .6 | .6 | .6 |
| fixed50 | .5 | .5 | .5 |
| progressive | .2 | .5 | .8 |
| shuffled | Same multiset as progressive, deterministic epoch permutation | Same permutation | Same permutation |

At missing frequency p, state probabilities are [1-p, p/2, p/2] for
complete, RGB missing and TIR missing. Patch masking remains .1 per
modality, independently, using the existing native sampler.
The native sampler preserves the first template and also samples availability
for subsequent templates, in addition to searches. It is incorrect to
describe its current whole-modality masking as search-only.

The progressive and shuffled schedules have identical twenty-epoch mean .5
and identical stage-count histograms. fixed50 controls average strength;
shuffled controls ordering. fixed60 is a same-budget reference, separate
from the current independently stopping seed2028 training.
The shuffled permutation uses isolated Python RNG seed65101.
Do not interpret a comparison against fixed60 alone as a curriculum benefit.
Do not use early stopping: epoch16 must actually enter the .8 stage.
Record development trajectories to describe observed convergence and
distinguish a fixed training budget from convergence.

Implementation is isolated in research/curriculum_p/schedule.py and calls
the existing native sampler. No active runtime or frozen script is edited.
Synthetic checks compare native outputs, stage transitions, probability
mass, no whole-modality [0,0], patch shape/binary values, identical patch
draws and RNG consumption across arms.
These checks establish implementation behavior only, not efficacy.

Before image smoke or formal launch, obtain terminal receipts from both
current trainers and freeze their complete input manifests as exclusions.
Use the existing 705 fitting / 88 development / 89 reserved audit boundary,
all previous exclusion chains plus active replication smoke/formal inputs,
and log actual frame identities. Do not race a growing input manifest.
Use common fitting/dev inputs and initialization across all four P arms.
Preserve all twenty checkpoints initially; report final-five full official
Complete/Miss results, and identify the same internal-dev-selected
checkpoint for both conditions. Never clean using official-test ranking.
No val97 access. A synthetic PASS is insufficient to launch formal image
training: the image gradient/update smoke must pass first.

The separate train.py entry point is implemented using the frozen A
loader, validation, objective and initialization helpers. It runs20epochs
without early stopping, records actual inputs and masks, saves all weights,
and selects within epochs16-20 by internal development loss.
prepare.py creates four configs only after all original smoke/formal
completion receipts exist; missing original receipts stop preparation.
Formal preparation also requires a four-arm image-smoke verification.json
with status PASS, matching completion digests and unchanged verifier sources.
verify.py and official.py are implemented separately; neither has verified
real P image training or performed a benchmark run yet.
No new image training has been launched. Compile and sampler checks alone
do not establish model gradients, convergence, image-manifest pairing or
inference compatibility.

CPU synthetic checks passed in the actual server28 Python3.7 environment;
the receipt is reports/server28_curriculum_p_sampler_20261102.json.
The full trainer imports and --help succeeded; all four source files compiled.
prepare.py --phase smoke failed at the expected missing original formal
completion receipt, before creating any P campaign directory.
Before image smoke, check remaining train882 frame/clip feasibility from
metadata after exclusions; previous crop experiments demonstrated that a
requested fresh cohort can be infeasible. No fallback sampler is added.

### P Execution And Verification

launch.py starts one prepared arm on physical GPU0 or1, after checking free
memory, temperature and250W cap with the existing queue helper. Original
GPU2/3 training and its registered official queue remain separate.
Run at most two P arms concurrently, then the remaining two.
Smoke uses20tuples/epoch,8development tuples and20epochs/40updates;
it exercises all schedule stages but is not formal training.
Formal uses20000tuples/epoch and128development tuples.

verify.py checks all four configurations, partitions/exclusions, every
input identity/hash, every logged loss/gradient norm, learning-rate stages,
all20development rows, selected epoch within16-20, retained20weights,
finite final-five weights, actual backbone/head/utility updates and BN
hashes against initialization. Inputs and patch masks must match across
arms; modality masks must also match at equal missing frequencies.
It compares all six arm pairs and refuses unequal/truncated evidence.
This is saved-evidence checking, not independent image/optimizer replay.

After actual smoke PASS, prepare formal configs and train all four arms.
After actual formal PASS, official.py can report each arm's five checkpoints
in both complete/miss conditions:10full evaluations per arm,40total,4workers
per GPU. Each result is independently recomputed by isolated verify_official.
No development main is called, no --limit argument is used, and no weights
are deleted or selected by official metrics. Same dev-selected checkpoint
is identified for both conditions.

Synthetic contract checks exercised rejection of a14epoch course, early
stopping, selection outside final-five, nonfinite loss/gradient, wrong
probability/learning rate, wrong final-five/update count and mismatched
input/patch/modality records. Synthetic job planning checks40unique full
official output paths; this does not mean40benchmark jobs were executed.
The native-sampler parity, shared patch draws and RNG-consumption checks
also passed on CUDA GPU0. No images, optimizer or benchmark data were used.
Current contract receipt:
reports/server28_curriculum_p_contracts_v3_20261102.json.
CUDA receipt:reports/server28_curriculum_p_sampler_cuda_20261102.json.
Earlier contract receipts are retained with their own source hashes.
All eight P Python files compile under the actual Python3.7 environment;
launch/verify/official CLI imports pass. No P campaign directory or image
worker was created by these checks.

## F: Same-View Deep-To-Shallow Feedback

First implement a separately versioned structural candidate after P is
isolated. Round one saves search features after block3 and computes the
ordinary deep prediction. A small residual module uses that view's shallow
features, deep features, prediction and actual availability. Re-run blocks4-12
with shared backbone weights and predict again.
Only search tokens receive feedback. Keep template policy unchanged and
commit one box/template state update after round two per frame.
Both rounds receive task supervision. Permit a complete-input residual;
do not gate all complete-input feedback to zero.

At inference, use only the physically available view and its own feedback.
Complete-view activations must never enter the missing-view forward path.
Detach does not remove information leakage. Cross-view interaction is
training supervision only in this first version.
Check zero-residual parity, gradient paths, strict checkpoint reload,
actual unavailable-stream input invariance and one template update per frame.
Compare a single pass and a computation-matched second pass without
deep-to-shallow feedback, with matched loss and training budgets.
Record measured latency/memory, round1-to-round2 correction and destruction
rates, separately by availability. Block counts alone are not latency.

## X: Cross-View Supervision

Only after P and F are isolated, compare ordinary response KD, deep-to-shallow
response supervision and matched auxiliary task supervision. Ground truth
supervises both views. Do not label ordinary KD, dual views, recursive
backbones or a simple epoch schedule as new contributions without evidence.
Prior negative diagnostics do not establish universal impossibility;
they also do not justify recycling a rejected fixed signal as a new module.

Relevant author repositories identified in the user's proposal:
hoqolo/SDSTrack, CroitoruAlin/CBM, joe-siyuan-qiao/DetectoRS,
sundw2014/DCM. Pin and inspect relevant source commits before borrowing code;
no source from those projects was copied into the P sampler.

Author-repository HEADs verified with git ls-remote on November2:

| Repository | HEAD |
|---|---|
| hoqolo/SDSTrack | 822d9852fee66bdd6d9df4dcb679eebecd85e91c |
| CroitoruAlin/CBM | ca907b03df2db6f37441a036f5da9779f9636fa7 |
| joe-siyuan-qiao/DetectoRS | 612916ba89ad6452b07ae52d3a6ec8d34a792608 |
| sundw2014/DCM | df2f5f68e6f7eb50f27fe77c5047d1ffd87c2820 |

HEAD lookup proves repository identity at this observation, not a code review,
module portability or a novelty finding. Relevant implementation reading is
still required before the F/X designs borrow from these repositories.

## Original Replication First Epoch

At18:05:48CST original PIDs1288280/1288281 remained live, each with
epoch1/2000updates completed. Fixed128development mean task loss was
4.356546024481456 Utility and4.3102924053867655 Random. Complete/RGBmissing/
TIRmissing losses were3.654450/5.315351/4.099837 and
3.579864/5.275305/4.075708. These are train882 internal losses, not PR/SR,
not closed-loop benchmark metrics and not evidence of convergence.
Elapsed to first development result was6389.62/6368.71seconds.
At similar throughput14-20epochs would take roughly25-35.5hours total;
approximately23-34hours remain at this observation, before official evaluation.
Do not stop/restart or retune using this single-epoch comparison.

## Reporting Scope Correction

FINAL5_OFFICIAL_RECALC_AUDIT_20261102.md summarizes the recent forty-run
candidate campaign plus E0. It does not prove global historical maxima.
The migrated historical A Utility complete result records RGBT234 MPR
91.45071185356501, above the recent candidate maximum91.043051.
The original artifact was read on server28 at17:56:50CST:
artifacts/server25-data2/results/a_utility_full20_complete_20261020/metrics.json.
Its recorded coverage is245/220703 and234/116650; this read did not
recompute all trajectories or prove a fresh server28 evaluation.

The historical E4 epoch05 Miss MSR61.797040 is recorded in
docs/TASK_EVIDENCE_SOURCE_FINDINGS_20261014.md and HANDOFF.md.
Its complete artifact and provenance require direct checking before calling
it a newly verified global maximum. Report migrated historical observations
separately from the fully recomputed recent campaign.
No historical candidate is silently dropped; no per-metric mixture is
reported as a single checkpoint. Neither scope currently proves the
requested complete-input improvement or four Miss targets in one model.
