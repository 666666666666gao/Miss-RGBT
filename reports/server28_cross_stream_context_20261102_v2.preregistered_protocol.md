# Cross-Stream Context Pilot

## Question And Boundary

Frozen late-fusion deletion did not establish a useful absent-stream removal.
Do not train or tune that control. This new question tests whether missing
stream search queries can use available-stream spatial context through a
learned residual before the fixed fusion/head. It retains native search
features and template-derived missing-stream representations.

This is a bounded matched optimization diagnostic, not a novel paper module.
Cross-attention is ordinary prior art; DETR's official
`facebookresearch/detr/models/transformer.py` uses query/key/value attention.
The official `supertyd/FlexTrackV2` repository already describes incomplete
modality reconstruction and MoE fusion. No external implementation is copied.
An improvement here would not establish novelty or official benchmark gains.

## Registered Design

- Original E0 epoch20 checkpoint SHA256:
  `d20c8641e4263df3ceadf9e430d0ac96c181c13ad482bca28c2dddef592a8bb8`.
- Existing train882 sequence partitions: 705 fitting names, 88 stopping
  names, 89 final-audit names. These partitions were used in earlier
  investigations; fresh tuples are not untouched independent sequences.
  Original E0 historical exposure remains uncertified.
- Fresh seeds48101/48102/48103 sample256/128/128 tuples. First search only,
  complete initial template; all3 dynamic-template states and both
  missing-search states in fitting (1536rows). Development and audit use
  all9 states (1152rows each). Exclude exact tuples in all preceding
  fusion/head/geometry/KV/position/late-fusion pilot records. No val97,
  RGBT234 or LasHeR test/official masks.
- Cache float32 pre-fusion maps from unchanged E0. Record identity,
  input tensor digests, cache/source hashes, and native fusion/head replay
  equality. Audit data are sampled only after both arms stop and select.
- Adapter: LayerNorm,64-wide Q/K/V,4-head scaled dot-product attention,
 64-to768 output projection initialized zero. Missing-stream queries read
  available-stream keys/values; add residual only to absent-stream channels.
  All complete-search states preserve native inputs/maps exactly.
- Aligned arm keeps key/value spatial correspondence. Shuffled control
  permutes only value-token positions using one fixed seeded256-position
  permutation; keys, queries and native features remain unchanged.
  Both arms have equal parameters, initialization, minibatch order, and
  attention/gather shapes. This is a correspondence control, not inert noise.
  A fixed permutation could be partly learned; no sole causal claim.
- Baseline fusion/head/backbone weights and BN buffers frozen; baseline
  gradients absent. Native task loss:2GIoU+5L1+focal, native inference.
- Each arm trains sequentially on GPU2: batch32, AdamW LR4e-4
  (linear reference batch8/LR1e-4), decay1e-4, clip0.1.48updates/epoch.
  Maximum20epochs; LR4e-5 from epoch7; fresh stopping reference at7;
  stop after3 post-drop epochs without0.2% improvement, minimum10epochs.
  Choose minimum six-cell development task loss among final5, ties earlier.
  Preserve all weights through verification; no official-test selection.
- Record update counts, histories, actual peak VRAM, and convergence time.
  This finite cached pilot does not establish full-tracker convergence or
  full-model batch sizing; do not duplicate rows to inflate VRAM use.

## Gate

After both arms select on development loss, evaluate audit once:

1. Aligned-minus-E0 sequence-macro IoU >=0.01 in both missing-search
   directions, averaged over dynamic-template states, with positive95%
   lower bootstrap bounds.
2. Aligned-minus-shuffled IoU across six missing-search cells has positive95%
   sequence-bootstrap lower bound (10000draws).
3. Native/initial replay and complete-search maps bitwise equal, baseline
   whole-state digest unchanged, all losses finite, gradients absent.

Pass authorizes only a new train882 closed-loop diagnostic. Failure closes
this exact pilot; do not retune its audit sample or run official evaluation.
Full Complete approximately+2 points and three validated modules remain
separate unmet goals; a complete-search bypass cannot deliver that gain.

## Execution

Check server28 GPU2 occupancy, temperature<85 C and250W power cap.
Estimated5-20minutes for caching and both matched arms; revise from measured
epoch throughput. Inspect near estimate, then poll at180-300second intervals.
Keep raw caches and weights on server; only small reports go to GitHub.

## Result

Pending.

### Execution Correction

First launch PID307361 stopped before optimizer updates during initial
Complete-output parity checking; final audit had not been sampled.
`check_cross_stream_layout.py` replayed16 actual cached Complete inputs.
Native and adapter features were numerically identical, but their strides
were `[393216,256,16,1]` versus `[393216,1,24576,1536]`. Native repeated
forwards matched exactly; changed-layout outputs differed (maximum offset
map difference0.000800133). Making adapter output contiguous restored exact
score/size/offset/box equality. The adapter now explicitly returns the
native contiguous layout. No tolerance or gate change. Retain the failed
directory and run corrected code in a separate `v2` output directory.
