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

Pending.
