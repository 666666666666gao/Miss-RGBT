# RPPM Six-Arm Microbatch Engine

## Implemented Boundary

`research/curriculum_p/engine.py` implements one microbatch for fixed60,
fixed50, progressive, shuffled, paced and paced+norm. It does not replace any
existing trainer or launch a campaign.

All six arms execute the complete branch first, update the detached complete
task-loss EMA, sample availability and patch masks, then execute the missing
branch. The four control arms use the existing schedule sampler. Both paced
arms use the existing independent-generator thinning sampler. Only paced+norm
enables per-state missing-task-loss normalization. Forward/reverse KD weights
and margin are read from the supplied configuration.

The helper returns the loss, status, both branch outputs and masks. Utility
probe arguments route to the designated existing C or M forward; the caller
must still add utility supervision, perform accumulation/backpropagation and
update the optimizer. The engine itself does not add a utility objective.

`CurriculumState.state_dict()` captures the arm, processed microbatch count,
complete-loss EMA, three missing-loss EMAs, their momenta, and the independent
mask generator state. Tensor values are copied to CPU and detached. Reload
restores these values onto the target state device.

This state is only part of a future full checkpoint. The trainer must also
save model/optimizer state, native CPU/CUDA and Python/NumPy RNG states,
utility probe/selection state and trustworthy data-loader position or replay
information. The CPU check does not establish resumability of a full trainer.

## Actual Server28 CPU Evidence

The check was executed successfully on2026-11-03, approximately09:24CST:

```bash
cd /home/gaob/Miss-RGBT
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
PYTHONPATH="$PWD/runtime/task_utility_20261018:$PWD" \
conda-envs/miss_rgbt/bin/python research/curriculum_p/check_engine.py \
--output reports/server28_rppm_engine_checks_20261103.json
```

The report uses exclusive creation; use a new output path for a deliberate
repeat. No GPU allocation or real image loading occurs.

- Eighteen stochastic synthetic executions cover all six arms at epochs1,6,16.
  Their model draws random values in both forwards, so C-before-mask ordering
  affects and is checked against an explicit reference execution.
- Outputs, availability, patch masks and native RNG endpoints match the
  explicit reference bitwise.
- The five normalization-off arms match native loss, status fields and
  parameter gradient bitwise. For paced+norm the field
  `norm_off_native_loss_gradient_bitwise=false` means that this comparison is
  inapplicable: normalization is intentionally enabled. Its gradient is finite.
  Independent normalized-loss arithmetic is covered by `check_pacing.py`.
- All six arms serialize through torch.save/load; restored EMA/generator
  state reproduces the next microbatch loss, status, outputs, masks,
  parameter gradient and resulting curriculum state bitwise.
- C/M utility-probe routing uses exactly two forwards.
- `cuda_initialized=false`.

Report: `reports/server28_rppm_engine_checks_20261103.json`.
SHA256: `d07398233cda85b6dc2a5962651fa7fac79360c7f257993ca834a5f2631da1ff`.

## Remaining Gates

This is implementation evidence, not Step7 preparation, real-image smoke,
saved model reload, module efficacy or an official tracking score.

After completed and independently verified R1/R2 full official results
establish the training protocol, a separate campaign must integrate this
engine into the OSTrack-initialized Utility trainer. It must freeze sources,
record per-frame training identities and the prescribed initial microbatch
hashes, share data/model initialization and RNG ordering across arms, log and
checkpoint curriculum state, use fixed val97 task loss for epoch selection,
and verify actual saved training artifacts. Preserve twenty full epochs and
all final-five full479-sequence Complete/Miss evaluations.

The old E0-continuation P entrypoints are unchanged and remain incompatible
with this new protocol. Do not restart their retired controller.
The R1/R2 missing frame-provenance issue and the OHCAS data gate are unchanged.
