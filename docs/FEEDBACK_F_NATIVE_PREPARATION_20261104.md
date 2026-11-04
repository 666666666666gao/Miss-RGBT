# Native-Protocol F Preparation

`research/feedback_f/protocol_prepare.py` supplies the five-arm preparation
entry missing from the native OSTrack/RPPM F bridge. The historical
`prepare.py` remains unchanged: it binds the retired four-arm P campaign
and E0 continuation, so changing its campaign path would not implement
the current protocol.

## Actual Gates

The entry first calls the existing six-arm P* selector, which verifies the
completed campaign and selects the curriculum using internal development
loss. If P is unfinished it returns `WAITING_FOR_P` without opening M1
cache evidence or creating campaign directories.

After P, the entry calls the actual M1 saved-evidence verifier. Missing
M1 or a required embedding retry waits. An accepted M1 decision returns
`WAITING_FOR_OHCAS_COMPOSITION`: the current native F builder has no accepted
OHCAS composition, so silently dropping that module would change the
registered branch. The explicit closed-both-negative M1 outcome allows
preparation without accepted OHCAS. Missing provenance cannot manufacture
that outcome.

`ready` here means preparation may proceed. It is not permission for formal
training. Every preparation result retains `f_training_authorized=false`;
the runner still needs all five matched four-update real-image smokes.

## Frozen Configuration

When the actual gate passes, preparation creates new, separate report and
runtime directories. It copies native lib/miss_rgbt/cadtrack sources from
the frozen A runtime and the five-arm F plus curriculum/R protocol sources
from its own complete source checkout. No active runtime is modified.

All five arms use the internally selected P curriculum and its fixed
development cache, but initialize from OSTrack rather than the selected P
checkpoint. This is a curriculum transfer, not continuation training.
The prepared settings are:

- Seed2030 and paired mask seed65403, following the existing F seed convention.
- Microbatch2, accumulation2, effective batch4; LR5e-5 and backbone LR5e-6.
- 20 complete epochs,20,000 tuples per epoch,100,000 optimizer updates per arm.
- Native train882 and the inherited fixed128-sample val97 development cache.
- All BN frozen, all five modes paired on optimization and data settings.
- Epoch10 and16-20 retained; weights selected by the inherited final-five
  internal loss rule, never by official metrics.
- Ten full evaluations per arm,50 total: each479 sequences/337353 frames,
  comprising LasHeR245/220703 and RGBT234234/116650.

Changing effective batch from P's8 to F's4 doubles the optimizer-update
count for the same tuple/epoch budget. All five F arms share the new budget;
the single-pass F control is necessary before attributing changes to feedback.
The real-image smokes still need to verify actual memory suitability.

`check_evidence(config)` validates the exact generated configuration,
frozen source and parent bindings, initializer, development cache and train
names before a future runner caller uses it. Existing directories are never
overwritten. This entry creates no training or official-evaluation process.

## Verification And Limits

Six CPU unittest methods passed in Python3.7 on Server28. They cover:
waiting P without M1/cache access or writes; waiting/retry/accepted M1 paths;
all30 curriculum/mode configurations; paired settings and OSTrack retention;
isolated source/config creation and evidence checks; changed learning rate,
parent or frozen source rejection; and refusal to overwrite an existing
preparation. The P*/M1 decisions, checkpoint/cache bytes and copied source
trees in these fixtures are synthetic. This is not real parent acceptance
or a whole-production-package deployment test.

The three files were staged under
`/tmp/codex_feedback_prepare_20261104.bDfteB/research/feedback_f/`, using
the earlier five-arm staging package for dependencies. The live readiness
probe returned `WAITING_FOR_P` with20 missing prerequisite paths. It confirmed
the five mode names, CUDA uninitialized, and the designated real campaign,
snapshot and runs directories absent.

| File | SHA256 |
| --- | --- |
| protocol_prepare.py | 09c497c4afedcc667e37c564467199d3a25af2ec9f1176e12a6522bb7e1d69a0 |
| check_protocol_prepare.py | 39cc5ca37a7b582dd407541133aa5b3982ee1ed45d9ebcddc88fcf4fc5671d6e |

No fresh independent reviewer is available through the exposed native
reviewer tools, so independent deployment review remains unfulfilled.
No deployment acceptance, actual F preparation, real-image smoke, model
training, memory profile or official score is claimed. A complete five-arm
source checkout must be frozen at authorized preparation time; the small
temporary test stage is not that deployment.

The remaining execution work is the native-protocol smoke/formal dispatcher
and campaign reporting integration, plus accepted-OHCAS composition when
the actual M1 decision requires it. Existing R/A/P jobs and their gates are
unchanged.
