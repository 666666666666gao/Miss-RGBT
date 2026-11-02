# Template Age And Missingness Binding

## Code Evidence And Question

VisibleErrorSampler in miss_rgbt/data.py emits template_ids=anchor+previous,
so template0 is more recent than template1. sample_masks in miss_rgbt/train.py
always keeps template0 complete, masking later templates. In contrast,
MissingTracker keeps the initial template at0, appends recent dynamic
templates at1, and stores their actual availability. Thus missing-template
augmentation is bound to an older crop in training and to a recent crop in
inference. This is an observed data contract difference, not a proven bug
or proof that changing training improves benchmarks.

Native templates share positional embeddings without a temporal role
embedding; swapping images and availability together should primarily be
permutation-invariant. Therefore image order alone is not a sufficient
explanation. The experiment explicitly measures that control.

Earlier independent-anchor cosine and learned-anchor pilots already failed
to establish robust identity acceptance; do not repeat/relabel them.
This question concerns temporal mask assignment rather than descriptors.

## Registered Protocol

- Original frozen E0 checkpoint SHA256
  d20c8641e4263df3ceadf9e430d0ac96c181c13ad482bca28c2dddef592a8bb8.
  No optimizer or source-model changes.
- Train882705 fitting names,128fresh tuples,seed52101,first search.
  Exclude exact tuples from prior pilots including candidate-quality and
  learned-quality-ranker. Reused names; not untouched sequences.
- Require actual ids older<newer<all search ids, record age gaps and
  input tensor digests. Batch16,float32,eval,deterministic CUDA settings.
  All3 dynamic-template states crossed with3search states,1152rows.
- Native training-like: [newer complete,older dynamic-state].
  Chronological inference-like: [older complete,newer dynamic-state].
  Matched permutation sham: [older dynamic-state,newer complete].
  Search image and availability identical; native Hann decoder unchanged.
- Record all3 boxes, GT and IoUs per tuple/state. Original weights/buffers
  unchanged, gradients absent. No val97, test frames, RGBT234 or official masks.
- Primary effects: chronological-minus-native sequence-macro IoU,
  dynamic states01/10 pooled, separately for search11/01/10.
  10000sequence-bootstrap samples,seed52102. Complete dynamic11 rows
  and permutation sham are controls, not separate improved-model claims.
- Data-contract effect supported only if any primary mean abs>=.01 with
  a95% interval excluding zero, and all per-row sham IoU deviations<=1e-4.
  Measure map/box numerical permutation deviations without assuming exact
  equality. This is diagnosis of either-sign effect; a negative result
  does not authorize deploying chronological inputs as a better tracker.
  Report all primary categories; no tuning on this cohort.
- Even a supported diagnostic only authorizes designing a new matched
  train882 training experiment; no automatic full training, official
  evaluation or validated module. If unsupported, close this exact question.

## Execution

GPU2free<500MiB,temperature<85C,cap250W; warm existing runtime.
Estimate1-3minutes,first check180seconds. Keep weights. Independent
saved-box arithmetic, tuples/source/protocol hashes and bootstrap checks.
No neural replay is claimed without saved/replayed input tensors.

## Result

Completed GPU2 PID538386 in79.312seconds:128tuples/118fitting sequences,
1152rows. Either-sign diagnostic criterion supported, not an improvement
criterion or module promotion.

| Search state, dynamic missing01/10 pooled | Chronological minus native IoU | 95% sequence-bootstrap CI |
|---|---:|---|
| Complete search11 | -.005529 | [-.020217,+.008644] |
| RGB-missing01 | -.018251 | [-.035117,-.001065] |
| TIR-missing10 | -.015967 | [-.031889,-.000592] |

Each primary category has256rows from118sequences. Both missing-search
effects exceed.01 in magnitude and intervals exclude zero. The
complete-search interval crosses zero. This says an older complete /
newer masked template pair is harder for this frozen E0 on these tuples
than the opposite availability assignment. It does not show why, prove
that current training caused the official gap, or show training on the
harder assignment improves performance.

Permutation sham mean IoU delta-.000000234,
CI[-.000000922,+.000000420], all per-row IoU deviations<=1e-4.
Maximum box component difference.0000352263, map difference.0141708255.
The outputs are not bitwise invariant; small decoded effects satisfy only
the registered sham IoU condition. Do not claim identical attention or maps.
The sham and chronological outputs for dynamic-complete rows matched each
other exactly as they use the same inputs.

Server and local saved-evidence analyzers passed temporal ordering/age gaps,
actual fitting names, exact prior tuple exclusions,9-cell identities,
3456box-IoU checks, sequence-bootstrap recomputation, decision, source/
base/protocol/report hashes. Producer reports original state unchanged and
gradients absent. No raw image tensors or maps persisted for this run;
input digests alone do not provide neural replay or independent verification
of every producer-state assertion. No external scientific review.

Report SHA256:
`20cb927876ec418518b6d196e859f6e861b756c26284ce994817ae9fbb94ae79`.
Evidence: `reports/server28_template_chronology_20261102/`.
No optimizer, val97, official test evaluation or checkpoint cleanup.
Official maxima unchanged; Complete+2,RGBT234-Miss and three modules unmet.

## Next Experiment Boundary

The next matched training question should compare native newer-complete
mask assignment against older-complete assignment using the same E0,
images, mask states, parameter budget and update order. Changing template
order while keeping masks with their images is only a sham. New fitting,
stopping and delayed-audit tuples are required, with chronological inputs
for both arms at evaluation. It must retain complete11_11 and both missing
search endpoints, then test fresh prediction-driven trajectories before any
official evaluation. Do not tune on this diagnostic cohort or interpret
its negative delta as a reason to change deployed tracking history.
