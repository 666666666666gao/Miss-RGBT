# R3: Matched Complete-Only And Complete/Missing Objectives

## Purpose And Exposure

This is a separate follow-up protocol, not an amendment to the frozen R/P
campaign. R1's development-selected epoch16 Complete result has already
exposed the official benchmarks: LasHeR PR/NPR/SR fell by
0.897504420/0.775185510/0.621649534 percentage points relative to E0.
All three fail the original E0-minus-0.1pp gate. Preserve that outcome;
neither R2 nor this new study can substitute for R1 at that gate.

The follow-up asks whether adding the existing missing-task and forward-KD
package changes complete-input localization under a common training policy.
It does not isolate each change from historical R1, establish "gradient
dilution", or prove a new module effective. It is benchmark-motivated;
future choices use development evidence, with official results for reporting.

## Registered Pair

| Setting | Both arms |
| --- | --- |
| Arms | `complete_only`, `complete_missing` |
| Initialization | Fresh OSTrack, never A/R/P checkpoint continuation |
| Initializer | `/home/gaob/rgbx-initializers/OSTrack_ep0300.pth.tar` |
| Initializer SHA256 | `8e01d6251569ac84dbb53fdd062cd938551be2d889582f789aa3070196f42f41` |
| Network | Native CADTrack plus `MissingTrackerModel`, repair and U1 disabled |
| BN policy | All BatchNorm running statistics frozen in both arms |
| Seed | 2031 for initialization and native data sampling |
| Training pool | Full LasHeR train882, native two-template/three-search tuples |
| Budget | 20 epochs, 20,000 tuples/epoch, no early stopping |
| Microbatch / accumulation | 4 / 2, effective batch8 |
| Optimizer | AdamW, weight decay0.0001, gradient clipping0.1 |
| LR | Head0.0001, backbone0.00001 |
| Schedule | Linear warmup500 updates; multiply by0.1 after epoch10 |
| Workers | 2 per loader |
| Development | Existing fixed128-crop cache from the registered R development split |
| Checkpoints | Save epoch10 and16-20; retain every produced checkpoint |
| Resume | Not supported by this registration |

Each arm processes400,000 accepted native tuples,100,000 microbatches and
50,000 optimizer updates. A tuple has three search images; this is not one
pass over every training frame. Complete-only makes100,000 model forwards;
Complete/Missing makes200,000. Equal data and update budgets deliberately
do not mean equal compute or wall time. Runtime estimates must come from
the paired real-image smokes, not synthetic CPU timing.

Load OSTrack with the existing exact missing-key allowlist in
`research/protocol_r/common.py`. Do not call that module's `build` helper:
it binds YAML loading to the old R snapshot. The new study needs its own
isolated snapshot and config paths. Initial parameters and BN buffers
must match between arms before the first update.

## The Only Objective Difference

`complete_only` uses `task_loss(C, boxes, cfg).mean()` and one model forward.
`complete_missing` uses the existing native
`training_loss(C, M, boxes, cfg, 0.1, 0.0, 0.1)` and two model forwards.
The native objective is the sum of mean C task loss, mean M task loss and
0.1 times forward response KD; it is not an average of C and M losses.
Reverse KD has zero weight. Task loss retains the native sum over the
three searches and its GIoU/L1/focal implementation.

Both arms draw identical candidate native masks: availability probabilities
0.4/0.3/0.3 for complete/RGB-missing/TIR-missing, plus patch-drop
probability0.1. Complete-only records but never applies these masks and
does not execute a dummy M forward. Its absent M statistics are null.
Template availability follows the existing native sampler, including its
always-present first template.

For zero-based microstep k, the objective's disjoint role seeds are:

- Complete stochastic forward: `74101 + 3*k`.
- Missing stochastic forward: `74101 + 3*k + 1`.
- Candidate masks: `74101 + 3*k + 2`.

Each role forks/restores the CPU and active CUDA generator. Model/mask
draws must not advance the outer loader RNG stream. Matching seeds do not
guarantee equal later predictions after the arms' parameters diverge.
Use existing accepted-tuple provenance recording to compare actual frame
identities and transformed tensors; the record does not cover rejected
loader attempts. Do not read its manifest while growing.

## Development Decision And Reporting

The primary paired development contrast is
`C+M epoch20 complete task loss - C-only epoch20 complete task loss`.
Report the signed difference and both values. A positive value is worse
on this development endpoint, a negative value better, and zero a tie;
one paired seed does not support a statistical significance claim.
Also retain all epoch curves for11/01/10 availability and their mean.
This diagnostic does not establish the cause of the historical R1 decline.

For each arm, choose the official reporting checkpoint by minimum mean
11/01/10 development task loss among epochs16-20, breaking ties toward the
earlier epoch. Freeze this choice before inspecting its official scores.
The selected-checkpoint comparison is secondary to the fixed-epoch20
development contrast. Reuse the existing development cache only after the
permitted stage; do not create extra val97 tracking, partial or subset runs.

Evaluate all five final weights in both Complete and Miss conditions:
two arms x five weights x two conditions =20 full evaluations.
Each evaluation uses LasHeR245 sequences/220,703 frames and RGBT234234
sequences/116,650 frames, totaling479 sequences/337,353 frames.
Complete and Miss use the same checkpoint. Independently recompute full
saved-output metrics against ground truth with the existing verifier.
Report every registered row and the development-selected row; never choose
per-metric official maxima or delete weights based on official scores.

## Execution Prerequisites And Actual State

The A terminal audit must pass before any new GPU experiment is dispatched.
R1/R2 controllers continue their original full evaluation sets; the old P
queue retains its frozen gate. R3 is independent of a positive P result
and does not authorize P. Prefer GPU2/3 when they are released and truly
available, with temperature/power/memory checks at dispatch.

Before formal training, finish the isolated trainer/config integration,
bind the existing cache and frozen source/configs, perform the fresh code
review required by the experiment-bridge workflow, and pass paired
real-image smokes including loader pairing, BN invariance, finite
gradients, memory fit and strict checkpoint reload. Reviewer delegation is
currently unavailable; preparation can proceed, deployment acceptance
cannot be claimed.

Currently implemented: `research/localization_r3/objective.py` and its
synthetic CPU check. The final three unittest methods passed on Server28's
existing Python3.7/Torch1.12 environment in0.333s, including the explicit
CUDA-uninitialized assertion.
The tests check native C loss/gradient, native C+M loss/statistics and
eight paired unchanged-weight microsteps with24 unique role seeds and
unchanged outer CPU RNG. Their small synthetic model is not CADTrack
real-image, CUDA, optimizer-convergence or benchmark evidence.

CPU stage: `/tmp/codex_r3_objective_20261104.INqkMq`.
Objective SHA256:
`c1c9be07bb6c5ad042c281e578bbccbd7201bcb12b34626ebdb95c9fc93e2ed5`.
Test SHA256:
`2795d9fd69b8b4ed9d6b2b22543881e7a83e420c7135f65db52be6a1c17511d1`.
No R3 trainer, launcher, prepared campaign, real-image smoke, formal run
or metric result exists yet. This document fixes the intended paired
comparison before execution; it is not a completed deployment receipt.
