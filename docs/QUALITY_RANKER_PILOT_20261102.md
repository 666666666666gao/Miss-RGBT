# Matched Quality Ranking Pilot

## Anchor And Scope

Fresh fixed-top8 Hann candidate diagnostic passed its headroom gate.
Its oracle is not deployable. Now test whether fixed native candidates can
be ranked from available features using box-quality supervision, compared
with the same network trained to reproduce native response probabilities.
Ordinary quality alignment is prior art; see pinned TOOD source review.
This pilot cannot establish a novel paper module or official gains.

## Preregistered Setup

- Original frozen E0; checkpoint SHA
  d20c8641e4263df3ceadf9e430d0ac96c181c13ad482bca28c2dddef592a8bb8.
  All backbone/fusion/head parameters and buffers frozen, model eval mode.
- Existing train882705/88/89 names, fresh512/128/128 tuples with seeds
  51101/51102/51103. These sequences were used in prior pilots, not
  independently untouched sequences. Historical E0 exposure uncertified.
  Exclude exact tuples from prior pilots and candidate-quality diagnostic.
- Complete initial template, all3 dynamic-template states crossed with all3
  search states, first search only.4608 fitting rows,1152 development,
  1152 audit. Audit not sampled until both arms stop/select.
  No val97, RGBT234, LasHeR test or official missing masks.
- Fixed native top8 Hann candidates with stable lowest-index ties.
  Store candidate geometry/IoU/scores/indices and fused768-channel features,
  coordinates and4 availability flags. Initial native Hann box equality
  required. Cache inputs are detached; no geometry change.
- Ranker: LayerNorm over768 features, concatenate4 geometry values,
  2coordinates,2scores,4availability flags to780; Linear780->64,
  GELU, Linear64->1, zero final projection. Add residual to log Hann score.
  Initial chosen candidate must exactly equal native Hann peak.
  Both arms identical state, parameter count and epoch permutations.
- Quality targets: softmax(native candidate GT IoU /0.1), detached.
  Response control targets: native Hann scores normalized across8.
  Cross-entropy training in both arms. Neither receives GT as input
  during inference. Same target temperature is not meaningful for these
  different target scales; this control changes supervision, not only labels.
- Batch256 candidate sets,18updates/epoch, AdamW8e-4 (linear reference
  batch32/LR1e-4),wd1e-4,clip.1. These tiny cached features cannot usefully
  fill24GB; do not replicate data to inflate memory. Record actual peaks.
  Max20epochs, LR8e-5 from7, fresh stopping reference7, patience3 without
  .2% improvement, min10. Each arm selects minimum own-target9-cell
  development cross entropy among final5, ties earlier. Different absolute
  objective values cannot be compared across arms. Preserve all weights.
- One optimizer seed; no multi-seed robustness claim.

## Gate

After selection, evaluate once on fresh audit. Report sequence-macro IoU
gain quality-minus-native for full complete input11_11 and separately each
missing search direction averaged across3 dynamic-template states.
10000resample sequence bootstrap seed51301. All3 means must be>=.01 and
lower95% bounds>0. Additionally quality-minus-response over all9cells must
have lower95%>0,seed51302. All losses finite, baseline state unchanged,
no baseline gradients and source/cache/weight/replay verification required.
Pass authorizes only a fresh train882 closed-loop test. Failure closes this
exact pilot; do not tune its audit or enlarge/run official evaluations.
Selection uses no official test scores. Keep all five final checkpoints.

## Execution

Reuse2028 warm runtime. GPU2 free<500MiB,temperature<85C,cap250W.
One GPU runs both tiny rankers sequentially sharing immutable caches.
Estimate3-8minutes; refine from cache/epoch speed; checks180-300seconds.
Source freeze during execution. Independent saved-evidence statistics,
box arithmetic and separate-process cached-ranker replay before promotion.
Full Complete+2, RGBT234-Miss88.1/64.3 and three-module goals remain unmet.

## Result

Completed GPU2 PID499198 in231.174seconds. Both51585-parameter arms
stopped10epochs/180updates. Quality selected9,response10 from final-five
6-10 using their own development cross entropy, before audit sampling.
All epoch weights retained. Peak allocated quality441560576bytes,
response441975296bytes; reserved1086324736bytes each. This tiny cached
ranking experiment does not establish full-tracker batch size/convergence.

Audit128tuples/66actual sequences,1152rows per arm. Sequence-macro
crop-IoU changes (not official tracking scores):

| Comparison | Mean delta | 95% sequence-bootstrap CI |
|---|---:|---|
| Quality minus native / full complete11_11 | +.004284 | [-.033706,+.046220] |
| Quality minus native / RGB missing | +.022927 | [-.013764,+.064605] |
| Quality minus native / TIR missing | -.004925 | [-.039010,+.028868] |
| Quality minus response / all9cells | +.011954 | [-.014616,+.043682] |

All intervals cross zero, and two categories miss the.01 mean requirement.
Registered advancement=false. Do not enlarge this exact ranker, retune
its audit, run official evaluation or label it a validated paper module.
Earlier fixed-candidate oracle headroom remains valid on its own sample;
this learned pilot does not establish that those gains can be recovered.
It also does not prove that all quality-ranking methods fail. The shared
reused sequence partitions, one seed, static crops and no temporal identity
evidence limit generalization and causal claims.

Server and local saved-evidence analysis passed partition/exact prior-tuple
exclusions, row pairing, box-IoU arithmetic, stopping/LR/final-five
selection, bootstrap/gate and report/protocol hashes. Server additionally
checked listed sources, original E0, three candidate caches and selected/
all10 final-five weight hashes; each actual weight dictionary had51585
finite parameters and matching epoch metadata.
Separate-process cached selected-ranker replay reproduced all2304rows
exactly and independently checked9216audit candidate box IoUs. This reuses
ranker/evaluation implementation and cached fused features; no neural
image/backbone replay or independent scientific review. Cache candidate
ordering was checked; full native dense candidate reconstruction was done
in the preceding diagnostic, not independently rerun for every training
cache. Producer baseline digest unchanged and gradients absent.

Report SHA256:
`686d472292a7eb8a72e3a7ee168818b880cb601a0a353824f129cb73d5628bb7`.
Immutable initial protocol SHA256:
`7302d5775ba34a3741dac22c138499f5131d42816c202a6374995e8f71272499`.
Evidence: `reports/server28_quality_ranker_20261102/`.
No val97, official evaluation or checkpoint cleanup. Official maxima
unchanged; full Complete+2,RGBT234-Miss and three-module goals remain unmet.

Next investigation must bring distinct evidence, such as measuring whether
identity information in initial/dynamic templates can distinguish native
candidate ambiguities on fresh train882 trajectories. Do not infer such
information is useful from this failure or add a verifier without first
testing its discriminative signal under complete and both missing states.
