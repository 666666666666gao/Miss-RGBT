# RPPM Implementation Boundary

## Implemented

`research/curriculum_p/pacing.py` supplies the fifth `paced` schedule and
per-sample masking. The complete branch must run first. Its detached per-sample
task losses update a global EMA with momentum0.99; the first observation
initializes the complete-loss EMA to its microbatch mean. Hard samples exceed
the updated EMA. Missing probability is max(0.2, p_epoch-0.3) for hard samples,
and p_epoch otherwise. Log ema_c, hard_fraction and mean_missing_probability.

Native categorical draws at p_epoch are thinned using an independent caller
owned generator: retain each missing state with probability p_sample/p_epoch,
otherwise restore it to complete. This exactly samples the specified
[1-p_sample,p_sample/2,p_sample/2] distribution. It also preserves native
patch masks and global RNG consumption for every hardness pattern; all-easy
masks are bitwise equal to progressive. The first template stays complete.
The caller must seed and save this separate generator and the EMA in training
checkpoints. No extra draw changes the data/model RNG stream.

`research/curriculum_p/objectives.py` supplies state normalization separately
from the frozen native loss. Three EMAs (11,01,10) start at1 and use
momentum0.99. For each microbatch, update each observed state from all matching
search/sample losses, then divide each loss by that state's updated EMA.
Unobserved states retain their previous EMA. Sum search losses per sample,
then average over the batch, preserving the native objective's scale.
The native focal calculation retains its whole-batch reduction per search.
Complete task loss and KD remain unchanged; reverse reliability uses raw
task losses. With state_norm=False, delegate directly to the native objective,
preserving loss, status and gradients bitwise.

## Deliberately Not Launchable Yet

These are isolated helpers, not completion of plan Step6. The old schedule.py
ARMS and all old four-arm prepare/train/verify/official entrypoints remain
unchanged. In particular, their E0-continuation initialization and internal
development split do not implement the new protocol. Do not add paced to those
old campaigns or restart the stopped old P controller.

Before a six-arm P campaign, choose the protocol from completed R1/R2 evidence,
integrate C-before-mask execution and logs, serialize both EMA states and the
independent generator, share model/data RNG ordering across all six arms, and
extend preparation, source manifests, saved-run verification and launch gates.
All arms must then use the chosen initialization/BN/LR/batch policy,
20 full epochs, fixed val97 task-loss selection and final-five full official
Complete/Miss evaluation. GPU smoke and saved-weight evidence are still pending.

## CPU Check

Run from the server root without allocating any GPU:

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
PYTHONPATH=/home/gaob/Miss-RGBT \
/home/gaob/Miss-RGBT/conda-envs/miss_rgbt/bin/python \
research/curriculum_p/check_pacing.py \
--output reports/server28_curriculum_paced_checks_20261103.json
```

Checks cover six schedule boundaries, all-easy exact native state/patch/RNG
parity, all-hard and mixed categorical frequencies, easy samples unchanged,
EMA update ordering, per-search state assignment, absent-state preservation,
normalization-off native loss/gradient parity and finite normalized gradients.
This is synthetic implementation evidence, not actual image training,
convergence, module efficacy or an official tracking score.

## Execution Receipt

Server28 CPU execution passed on2026-11-03 at approximately09:04CST, using the
existing Python3.7/Torch1.12 environment, with CUDA_VISIBLE_DEVICES empty and
cuda_initialized=false. Receipt:
`reports/server28_curriculum_paced_checks_20261103.json`, SHA256
`682ba2c861285ddc2e466db5fb6bd86ecdbbc7cc572e5013bde49ea4a55f8fd1`.

The first synthetic invocation failed because the checker had not loaded the
CADTrack YAML: default search size320 disagreed with the synthetic16x16 output.
Loading the actual256px YAML corrected this fixture mismatch; native/runtime
code and training configuration were not changed.

At09:05CST all four trainers remained alive. R1 was at epoch1/update275 and R2
at epoch1/update100. The A Utility/Random pair continued onGPU2/3. Temperatures
were82/79/77/74C and all reported power limits250W. No new official score.
