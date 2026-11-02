# Template attention candidate evidence

## Question and boundary

The prior fixed-top8 oracle found candidate headroom, but quality ranking
without template evidence failed. Independent crop cosine and historical
small identity-MLP controls also failed; do not promote or rename them.
This distinct frozen diagnostic tests whether native template foreground
attention carries candidate-specific localization information beyond
template background attention.

Official OSTrack candidate_elimination at commit
33b5e12586216b7fd0e95d255bd01ba44cbec759 reads template-query/search-key
attention, masks template queries, averages heads/queries and ranks search
tokens (lib/models/layers/attn_blocks.py lines31-53). Root license MIT was
previously reviewed. Foreground-conditioned attention is prior art.
No external code/dependencies/checkpoints copied or installed.
https://github.com/botaoye/OSTrack/blob/33b5e12586216b7fd0e95d255bd01ba44cbec759/lib/models/layers/attn_blocks.py

Native template tokens are jointly contextualized with the search, so this
is NOT an independent initial-target identity verifier. Native causal
sampler template0 is its annotated anchor, not necessarily sequence frame0.
Results concern static augmented fitting crops, not natural trajectories.

## Fixed protocol

- Original E0, all weights frozen/eval/no gradients. Unchanged server28
  Python3.7/Torch1.12 warm runtime, GPU2<500MiB,<85C,250W.
-128 new fitting705 tuples, seed60101, batch16, all9 dynamic-template/search
  states, first search only. Initial template always observed. No optimization,
  val97, RGBT234, LasHeRtest or official masks.
- Mask dataset sampling visibility to exclude every frame in prior tuple,
  burst, crop, full-image fitting/development/amended-audit and last-observed
 32clip records. Original image/box/physical visibility data unchanged;
  exclusions affect only this sampler instance. Save all sampled identities.
- Read qkv output at one-based block6, without changing output. Native
 2stream ordering must be RGB thenTIR;385tokens=CLS+2x64template+256search.
  Reconstruct native scaled qk softmax over ALL385keys, then extract
  template0query rows1:65 and searchkey columns129:385.
- Template0 cropped normalized annotation defines fractional intersection
  of its target box with each8x8patch. Foreground query weight=patch coverage,
  background=1-coverage; normalize each group by its weight sum, average
  heads. Require positive finite sums/maps. This avoids empty center-point
  masks for thin objects without inventing a fallback region.
- Average stream maps over CURRENT observed search modalities only; no
  position-only missing stream is treated as current visual evidence.
  Foreground/background computation/candidate budget identical.
- Fixed stable native top8 Hann candidates and unchanged size/offset maps.
  Primary signal readout: all unordered candidate pairs with absolute GTIoU
  difference>=.05. Concordance=1 if attention ranks the higherIoU candidate,
 .5 exactties,0otherwise. Pool pairs within each sequence then macro-average
  sequences. Rows without informative pairs contribute0counts, no fake label.
  Report eligible sequence/pair counts and all9cells descriptively.
- Supplementary fixed deployable heuristics: argmax(nativeHann*foreground)
  versus argmax(nativeHann*background) among the SAME8. Report selected IoU
  deltas versus native; no heuristic tuning/promotion based on this sample.
- Main categories: complete11_11; RGBmissing across3template states;
  TIRmissing across3template states. Bootstrap10000 seed60301 bysequence.
- Hook/nohook fullmaps/boxes bitwise parity on firstbatch in all9states.
  Wholemodel state unchanged and gradients absent. Save normalized first-
  template/search images on server, native maps/attention maps/dense boxes,
  initial template labels, tensor hashes. Save firstitem rawqkv percell for
  separate NumPy attention reconstruction; no large caches in GitHub.

## Gate

Authorize only a separately registered matched train882 learning pilot if
EACH of3main categories has >=64 informative sequences and>=100pairs,
foreground concordance lower95%>.5, and foreground-minus-background
meanconcordance>=.05 with lower95%>0. This is an information gate, not
attainable tracking gain. Failure closes this fixed attention signal; no
layer/weight/mask/budget search on its results. Passing does not authorize
official evaluation, count a module or override failed earlier pilots.

Expected1-3min plus verification; firstinspect at180s, then180-300s only
if actual process live. Separate analyzer checks box/pair/sequence arithmetic,
frame exclusions, checkpoint/source/cache hashes, and NumPy attention
reconstruction. Preserve immutable source/protocol snapshots. Full official
goals and three validated modules remain unmet.
