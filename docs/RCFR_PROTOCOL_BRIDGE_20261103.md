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

## Remaining Dependencies

The old `feedback_f/prepare.py` and `train.py` still bind the retired four-arm
P campaign, E0 continuation and partitioned data. They cannot consume the new
six-arm campaign by changing a path. This bridge does not repair them.

Next work requires a coherent new F preparation/trainer/verifier/evaluator
path using the selected P curriculum and the accepted or closed M1 decision.
It must preserve native train882/fixed val97 selection, common data identities,
20 full epochs without early stopping, epoch10 and16-20 retention, four real
image smokes, strict model reload, and ten full official evaluations per arm.
Every evaluation must verify479 sequences/337353frames. Full final metrics
must use one internally selected checkpoint/config; no test-selected maxima.

New F runs use microbatch2/accumulation2, effective batch4, LR5e-5 and
backbone LR5e-6 as previously profiled, subject to real image smoke. Actual
round IoU correction/destruction and matched latency remain required.
OHCAS's unresolved historical R frame-provenance/data gate is unchanged.
The persistent P queue does not dispatch F or OHCAS.
