# Updated Research Priorities And Source Audit

## Evidence And Interpretation

This records the user's second review and a read-only source check on
2026-11-04, after the 14:57 UTC handoff. It is not an independent deployment
review, new training, or a claim that any proposed module works. The pasted
review's sandbox download links were not available as local artifacts; no
Excel/PDF/ZIP content was inspected or reproduced.

- R1 trained20 epochs from OSTrack but selected epoch16 remains below E0
  on Complete LasHeR PR/NPR/SR by0.897504420/0.775185510/0.621649534pp.
  Preserve the failed original E0-minus0.1pp gate. Do not substitute R2 or17.
- A Utility14 versus Random15 is not a stable selector win across datasets.
  Comparisons to historical Utility17 also differ in partition/stopping/selection,
  not just seed. Preserve the original selected weights and every checkpoint.
- R2 selects17. Its epoch16 pair is a verified but unselected reporting row.
- Full-pair center-error diagnostics show a heavier missing-condition tail;
  geometric oracles are not official scores, causal proof or predicted gains.
- Published SCDT numbers are external comparisons, not certified same-scorer
  reruns. No per-metric model splicing or benchmark-maxima weight selection.

## Source Findings

1. U1 is enabled in the R3 evaluation path, contrary to earlier prose.
   `research/localization_r3/control.py:evaluate` calls the root official
   evaluator. `evaluate_candidate_official28.py:worker_init` loads the native
   task-utility snapshot YAML and `MissingTracker`. The actual server YAML
   sets two templates, update interval5 and threshold0.65. The tracker appends
   an available-modality template after one final state update; it preserves
   the initial template. `task_utility=None` disables A, not template U1.
   Both-missing frames skip track and hold the box, so the interval counts
   effective track calls, not all real frame numbers. This is source evidence,
   not a newly executed tracking test. Native tracker SHA256:
   `1d05274d2827de9a4e08044e5837fedaba971400162a72cb90e98c0d1679efa4`.
2. Fresh fusion parameters/statistics are an explicit initialization difference.
   `research/protocol_r/common.py:MISSING` lists fusion convolution, BN affine
   and running mean/variance. R3 `train.build` verifies that exact list and
   records initial tensor identities. `fix_bn(model, True)` sets all BN to
   eval but does not freeze convolution/affine gradients. The two arms share
   this policy. Existing native CPU evidence matches211 parameter tensors
   and39 BN buffers; it does not establish calibrated fusion statistics or
   prove that allBN causes the observed decline. A statistics-policy contrast
   must be a separate registered experiment, not an edit to this snapshot.
3. R3's native task loss sums searches to[B], intentionally preserving the
   existing objective. It does not claim state normalization. The existing
   `research/curriculum_p/objectives.py` state-normalized path first reconstructs
   `[search,B]`, stacks matching availability, and only then normalizes. Do not
   report a hypothetical shape bug as an observed defect or duplicate this code.
4. Native response KD is KL(teacher C || student M): teacher values are detached
   before `kl_div(log_softmax(student), softmax(teacher))`. Size/offset are not
   separately distilled. Reverse weight is zero in R3. These are not reasons
   to reverse the existing implementation or add an unregistered geometry loss.
5. Original F protocol still depends on P terminal evidence. The user's suggested
   history-free route is a future independent registration; this audit does not
   remove the existing P prerequisite or authorize F/OHCAS deployment.

## Execution Order And Limits

1. Finish original R1/R2 final-five full reporting with their existing controllers.
2. Obtain genuine R3 predeployment review, then both paired four-update real-image
   smokes. R3 is an objective-package comparison, not equal compute:100000 versus
   200000 forwards at matched400000 tuples/50000 optimizer updates.
3. If interpreting an R3 effect, register C+C and loss-scale matched controls
   before claiming a missing-learning mechanism. Define BN/init and retain U1.
4. Test development crop-loss versus closed-loop tracking association only on
   independently eligible development sequences. No extra val97/subset official
   runs and no retroactive selection changes. RGBT234 development independence
   remains unresolved; pre-registering a grid does not erase test tuning exposure.
5. Register curriculum controls matching overall severity/distribution, including
   a paced probability-to-sample shuffle. Ordinary fixed/progressive schedules
   alone are not evidence for sample-conditioned novelty.
6. Test history-free RCFR using single/none/shallow/deep-no-response/deep under a
   newly registered parent path. Require correction AND corruption rates plus
   measured FPS, tail latency and GPU memory. Only the final prediction commits
   state/template once. No C-to-M hidden current-modality activation transfer.
7. Add history only after source, crop transform, real frame-age and validity
   provenance are demonstrably correct. Two supported mechanisms are preferable
   to retaining an unsupported third contribution.

Every formal reporting checkpoint still needs Complete/Miss on LasHeR245/220703
and RGBT234234/116650:958 sequence runs,674706 frames, ten metrics per pair.
Same weight and inference policy within each dataset pair; retain all weights.

## Segmented Backward: Isolated Candidate, Not Deployed

A separate optimization candidate is C forward/backward followed by M forward/
backward with detached C targets, no parameter step between views, and one clip/
AdamW step after accumulation. C must retain task supervision, not no_grad.
Its acceptance requires native per-parameter gradients, actual optimizer updates,
BN buffers, mask/DropPath draws and peak memory compared to the current joint
graph. It is restricted to reverse_weight=0 with no cross-view activation path.
No memory-saving ratio is claimed. Do not apply it to the frozen R3/runtime
or call synthetic equivalence native evidence.

An isolated implementation now exists in `research/localization_r3/segmented.py`.
It is not imported by the registered trainer or controller. The narrow contract
is native C/M task plus0.1 forward response KD, zero reverse weight, no utility
auxiliary supervision or cross-view activation exchange. Three synthetic CPU
tests compare stochastic masks, outer RNG, accumulated per-parameter gradients,
BN buffers, once-clipped AdamW updates and optimizer state under frozen/updating
BN. They and the three existing objective checks passed together in0.571s with
CUDA uninitialized. Tests ran from isolated staging
`/tmp/codex_segmented_20261104.rutu5k`; frozen files were not edited. A first
test attempt incorrectly compared metadata keys as numeric loss diagnostics;
the test was corrected to compare the five actual objective diagnostics.
This first test set is not native-model/image equivalence, a memory measurement
or launch acceptance.

The subsequent `check_segmented_native.py` executes the actual103138437-parameter
OSTrack-initialized network on synthetic batch1 CPU images: two128px templates
and three256px searches, one microstep, all BN statistics frozen. It compares209
non-null gradients (max absolute difference2.384185791015625e-7),39 exact buffers,
211 updated parameter tensors (max5.960464477539063e-8) and627 AdamW state tensors
(max5.820766091346741e-11). Both preclip norms are60.93133544921875. It passed in
56.633560s with CUDA uninitialized. The full result is
`reports/server28_segmented_native_synthetic_cpu_20261104.json`, SHA256
`26ee1eb0e5323b5864a644ff43de6925a099e2af3f3288abfb5417c94e77bfd7`,
matching the server staging output. This extends the network scope, but still
does not prove native batch4 real-image/accumulation equivalence or GPU memory,
speed, launch acceptance or efficacy. No frozen trainer imports this candidate.

## Actual Review Attempt And Blocking Condition

The resumed run called native spawn_agent with fresh context, model gpt-6-astra
and reasoning_effort=max as required by the installed policy. Actual response:
`Reasoning effort max is not supported for model gpt-6-astra. Supported reasoning
efforts: low, medium, high, xhigh`. No reviewer started and no acceptance record
was created. This is a model/effort compatibility barrier, not absent spawn API.
No waiver has been received. Changing the required effort/model needs explicit
human instruction; resuming the goal does not waive review.
