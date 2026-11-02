# P/F/X Primary Source Audit

Inspected on November 2, 2026 using author Git repositories and `git show`
at the commits below. This is implementation evidence, not a novelty review
or an experiment showing that these designs improve Miss-RGBT.
No reference source was copied, installed or executed.

## Repository Pins

| Repository | Inspected commit | Relevant source |
|---|---|---|
| hoqolo/SDSTrack | 822d9852fee66bdd6d9df4dcb679eebecd85e91c | lib/models/sdstrack/vit_ce.py; lib/train/actors/sdstrack.py; lib/models/layers/attn_blocks.py |
| CroitoruAlin/CBM | ca907b03df2db6f37441a036f5da9779f9636fa7 | ablation/ablation_curricula.py; data_handlers.py |
| joe-siyuan-qiao/DetectoRS | 612916ba89ad6452b07ae52d3a6ec8d34a792608 | mmdet/models/detectors/rfp.py; mmdet/models/backbones/resnet.py |
| sundw2014/DCM | df2f5f68e6f7eb50f27fe77c5047d1ffd87c2820 | wrn/main.py; wrn/KL_divergence.py; wrn/resnet.py |

Repository names identify GitHub author repositories. For reproducibility,
inspect the pinned commit, not a later default branch.

## SDSTrack: Complementary Patch Masking And Shared Paths

`vit_ce.py:331-347`, `apply_mask`, uses RGB mask probability .3 and
`dte_mask = ~rgb_mask & (dte_prob < .3)`. The same token position cannot be
masked in both modalities by this function. Under independent uniform
draws, TIR's marginal masked probability is .7 * .3 = .21, not .3.
These are patch masks; they do not establish whole-modality missing-frame
behavior.

`vit_ce.py:210-245` iterates the same blocks over fine/masked paths,
skips the masked path during inference, and collects fusion features at
zero-based block indices [3, 6, 9, 11].
`attn_blocks.py:95-145` chooses adapters by submode (rgb/dte/rgb_dte);
the fine/masked path label does not instantiate separate adapters.
Shared backbone does not mean every modality uses identical adapters.

`lib/train/actors/sdstrack.py:124-133` averages fusion-stage feature MSE
and uses .6 * fine task + .2 * masked task + .1 * feature MSE.
Neither feature argument is detached at this MSE call.
This is evidence of two-view loss coupling, not direct clean-to-missing
activation transfer or a deep-to-shallow second inference pass.

Our `miss_rgbt/model.py:forward_tokens` applies independently sampled
patch visibility before positional addition. P retains this native masking.
Do not silently change P to complementary masks: that would change a second
factor alongside the frequency schedule. A separate later comparison could
test the masking difference, but no present result identifies it as the cause
of RGBT234 weakness.

License inspected: MIT, copyright 2024 hoqolo.
Blob IDs: vit_ce.py `5697691feb7a3355b7e53cc35d5c7f20350c1f7e`;
actor `d95962357c52ac17a21a6281ccec130b03eda80d`;
attn_blocks.py `2c36f3a882487eb9c59948e73dac9fd870bf4e42`.

## CBM: Patch Curriculum Is Not Whole-Modality Curriculum

`ablation/ablation_curricula.py` implements static, linear, logarithmic,
exponential, reversed linear and repeated schedules. Linear repeats each
value for four epochs; anti_linear reverses that list.
`data_handlers.py:29-52` computes Sobel gradient magnitudes, averages
spatial patches and normalizes those values into sampling probabilities.
This is image patch masking for classification.

P changes the probability of an entire modality being missing while keeping
patch masking fixed. Borrow the controlled schedule comparison principle;
do not describe a generic increasing mask schedule as a new contribution.
The fixed50 and shuffled controls remain necessary to separate average
strength and ordering from the progressive schedule.

No LICENSE file was present in the inspected repository tree. This does not
establish permission to copy its implementation; P uses our native sampler.
Blob IDs: curricula `1dad95fc8744eee805a29b158142997d7e88ffb6`;
data_handlers.py `beb51ef88e6fae146456af711396badf02ad0bd5`.

## DetectoRS: Actual Recursion And Two Different Zero Initializations

`mmdet/models/detectors/rfp.py:RecursiveFeaturePyramid` defaults to two
steps and `rfp_sharing=False`. It builds additional backbones unless sharing
is explicitly enabled. ASPP transforms pyramid features; `rfp_forward`
recomputes backbone features and the neck is called again.

`mmdet/models/backbones/resnet.py:210-217,269-309,595-607` defines
zero-initialized feedback projections and adds projected pyramid features
inside the recomputed residual stages. This is genuine second-pass
computation, not a feedback arrow without another forward.

The separate `rfp_weight` convolution in rfp.py is also initialized to zero,
but it is followed by sigmoid: its initial blend coefficient is .5, not zero.
Do not cite this gate initialization as evidence of zero feedback/output
parity. F must test its own identity behavior with our actual model.
Our proposed shared blocks4-12/search-token feedback differs from RFP's
default separate backbones and stage-level pyramid feedback.

License inspected: Apache 2.0, copyright 2020 Siyuan Qiao.
Blob IDs: rfp.py `f0274aacbcc3c759fa220a9a7075f147ebe81e66`;
resnet.py `0b4e37e878d1850139bd15bc4a50c8260d1e1ef9`.

## DCM: Detached Targets Couple Losses, Not Inference Activations

`wrn/main.py:130-166` obtains three classifier outputs from each of two
networks, applies hard cross entropy to all six, and uses a six-by-six
connection map to couple all heads across networks through KL losses.
Targets are explicitly detached via `KL_divergence(o2.detach(), o)`.
`wrn/resnet.py:31-39,69-82` defines auxiliary branches and returns their
logits along with the final classifier logits.
`wrn/KL_divergence.py` normalizes on classification dimension1.

This supports X as training supervision without privileged activations in
the missing-view inference path. It does not supply a drop-in tracking
loss: tracking response maps and box geometry require their own objective
definition and auxiliary-task control. Ordinary multi-head KD is prior art,
not sufficient novelty.

No LICENSE file was present in the inspected repository tree; do not copy
implementation without establishing permission.
Blob IDs: main.py `c33fb820034aff4453c5264f0badcfe3aa814e31`;
KL `6a168e79278456beeee43d11e9106761fc15d0c4`;
resnet.py `49cef4aac9f48d08ab10146f4425a301776b4314`.

## F Integration Constraints From Our Actual Code

Inspected local model.py, task_utility.py, tracker.py and vit_care.py at
parent commit 0f8a733c7d6d4885e493a2df0286612637f9497c.
No active source was modified.

1. `MissingTrackerModel.forward_tokens` prepends a query token, then
   templates, then search tokens. Save the block3 state with this ordering.
   Inject residuals only into the last lens_x tokens of each stream.
   Preserve the saved query/template tokens when beginning round two.
2. Blocks share weights across RGB/TIR but call self-attention separately.
   A same-view residual may use the view's available RGB/TIR tokens;
   it must not import complete-view activations into a masked forward.
3. A Utility invokes utility_repair after block6. Re-running blocks4-12
   must also execute this repair at block6. Calling predict_tail on block3
   states would instead start after repair_layer=6 and omit blocks4-6 and
   repair. It cannot implement the proposed second pass.
4. `TaskUtilityRepair.select` advances a registered RNG buffer only in
   random mode. Default utility selection is deterministic top-k.
   Zero-residual parity must compare evaluation mode and matching initial
   selection state. If random mode is compared, explicitly define/replay
   selection state rather than consuming a second independent random draw.
   No generic fallback or active RNG-policy change is warranted.
5. Backbone drop path is stochastic during training, and two head calls
   can update BN twice unless frozen. Establish exact parity in evaluation
   mode; retain the current frozen-BN training policy. Do not mistake
   training-time randomness for a structural identity failure.
6. `MissingTracker.track` commits state/template updates after the network
   returns. Keep both rounds inside one network call and use round2 maps
   for the single existing external update. Returning two frame decisions
   would change tracker dynamics.
7. Both rounds' task losses require explicit normalization. Compare
   single-pass, a matched second pass without deep feedback, and actual F
   with matched objective scale and budget. A zero-residual deterministic
   second pass should reproduce round1; it alone is not evidence that
   extra computation improves predictions.

Next implementation remains isolated from the frozen active runtime.
P image smoke/formal execution still waits for original formal manifests
to stop growing. F/X have no implemented or trained efficacy result.
The three desired paper modules remain hypotheses, not validated claims.
