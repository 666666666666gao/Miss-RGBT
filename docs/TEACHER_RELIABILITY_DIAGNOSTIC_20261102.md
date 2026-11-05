# Complete-Teacher Reliability Gradient Diagnostic

## Observed Source Question

training_loss always applies complete-to-missing response KL when its
configured forward_weight is positive. Its reliability condition controls
only the optional reverse loss. Complete output is detached as a response
target,but the complete branch itself receives task gradients on shared
parameters. It is not a separate frozen teacher during formal training.
Earlier repair-scope crops included cases where complete task loss exceeded
missing task loss. This motivates measuring KD/task directions,not assuming
harm or treating all complete predictions as trustworthy.

The preceding raw/Hann pilot failed despite improvements over E0 because it
did not exceed its matched training control or protect recovery states.
Do not tune or promote either of those selected heads.

## Frozen Setup

- Original E0 SHA d20c8641e4263df3ceadf9e430d0ac96c181c13ad482bca28c2dddef592a8bb8.
  Warm GPU2 runtime. No optimizer or parameter update.
- Fresh128 tuples,seed55101,existing705 fitting names from train882.
  Original sampler/augmentations/order,complete initial template,all9
  dynamic-template/search states,first search only.1152 feature rows.
  Exclude exact tuples from every preceding pilot including chronology
  and corrected raw/Hann training. Reused sequences are not untouched data.
- Cache frozen fused768-channel features with native head replay. Pair
  each missing view with its own all-complete11_11 feature row,never another
  tuple. This isolates eval-mode KD; it does not measure formal training's
  DropPath noise,full-backbone gradient,AdamW/clipping or closed-loop effects.
- Native task loss and existing response_kd unchanged. Teacher detached.
  A copied native head is frozen except its final1x1 response projection
  temporarily allows autograd; autograd.grad never populates parameter.grad.
- Measure per-row task/KD gradients with respect to fused features and
  pre-sigmoid response logits. Native CENTER uses sigmoid_(),so a forward
  hook returns a clone while preserving the original pre-sigmoid witness.
  Complete-view maps/boxes must still match its teacher bitwise and KL
  must have absolute value<1e-6.
- Form exact per-example final-response-projection weight/bias gradients
  from logit gradients and projection inputs. Verify their sums against
  direct parameter autograd (atol/rtol2e-5 for different reduction order).
  Save all projection gradient vectors,norms,dots and feature cosines.
  E0/copied head state digests unchanged; no parameter.grad tensors.

## Registered Diagnostic Criterion

Primary population is six missing-search cells,768 rows.
Unreliable teacher means its native supervised task loss>=student loss;
reliable means strictly smaller. This uses training labels for diagnosis,
not an inference confidence signal. Require>=32 rows/>=16 sequences in
each group and>=16 shared sequences for a within-sequence group contrast.
Sequence bootstrap10000draws,seed55301 for conditional gradient cosines,
55302 for reliable-minus-unreliable mean fused-gradient cosine on shared
sequences. Positive cosine means KD descent locally agrees with task descent.

Support a next matched reliability-filtered KD experiment only if:

1. Unreliable-group fused cosine mean<=-.05 and upper95% bound<0.
2. Reliable-minus-unreliable paired fused cosine mean>=.1 and lower95%>0.
3. Unreliable-group final-response-parameter cosine upper95%<0.
4. Coverage,finite values,complete parity,state/gradient assertions,
   identity exclusions and source/base/cache/report hashes pass.

If supported,this only authorizes preregistering fresh matched training,
including a random/count-matched gate to distinguish reliability from weaker
KD. No training efficacy,paper-module novelty or official benchmark gain
is established. Failure closes this gating explanation on this cohort;
do not tune reliability margins or gradient cutoffs afterward.
This differs from the earlier regression/focal conflict probe: the contrast
here is native task versus complete-response KL.

Source hashes are resolved from actual imported module paths and persisted
before feature caching. All caches/original/final-five weights preserved.
No val97,RGBT234,LasHeR test or official missing masks. Estimate2-4min,
inspect near completion with180-300s milestone checks;no automatic restart
on observation timeout. Full complete+2,RGBT234-Miss and three-module goals
remain unmet.

## Result

GPU2 PID728997 completed55.655seconds.128fresh tuples,1152rows.
Missing-search population768rows:unreliable177rows/61sequences,
reliable591rows/114sequences;paired group contrast57shared sequences.

| Gradient cosine statistic | Sequence mean | 95% CI |
|---|---:|---|
| Unreliable,fused features | .004752 | [-.058037,.066890] |
| Unreliable,response projection parameters | .492476 | [.372922,.600634] |
| Reliable,fused features | .447424 | [.414153,.480243] |
| Reliable,response projection parameters | .432975 | [.348347,.515169] |
| Reliable minus unreliable,fused,paired sequences | .432528 | [.350958,.512308] |

Registered diagnostic_supported=false. Reliability separates fused-feature
alignment,but unreliable examples do not have established negative average
fused alignment,and their response-projection parameter alignment is
strongly positive. Larger supervised teacher loss is not sufficient to
infer harmful KD gradients. Do not launch reliability-filtered training
from this diagnostic or tune margins/cutoffs on these rows.
This does not prove that every KD update helps or rule out all reliability
methods;full-backbone parameter gradients and optimizer effects were not tested.

Server and local saved-evidence analyzers checked actual fitting partition,
exact prior-tuple exclusions,nine-cell coverage,finite values,all saved
projection-vector cosines and fused-dot/norm cosines,group coverage,
sequence bootstrap/decision and report/protocol hashes. Server verified
base/source/cache hashes. Separate-process native-head construction replayed
all1152cached gradient records exactly,using the same measurement function;
this is not independent image/backbone replay or external scientific review.
Producer verified complete parity,per-example projection-gradient sums
against autograd,unchanged copied-head/baseline states and absent .grad
tensors. Saved replay re-exercises those head assertions,not optimization.

Report SHA256:
`f46bbd0342729be6d3d8b15523908fcf7a6e3f26410b0352694b4614098382d6`.
Evidence `reports/server28_teacher_reliability_20261102/`.
No optimizer,val97,official evaluation or checkpoint cleanup.
GPUs idle38/37/42/37C,all250W. Official maxima unchanged;full goal unmet.
