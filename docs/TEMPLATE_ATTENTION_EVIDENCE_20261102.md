# Template attention candidate evidence

## Verified result

GPU2 PID1070949 completed51.497seconds normally.128fresh tuples from117
fitting sequences produced1152nine-state rows. All input template/search
frame ids are disjoint from the independently reconstructed prior exclusions.

Sequence-macro candidate-pair concordance (not official tracking metrics):

| Condition | Foreground | Background | Foreground-background [95% CI] | Informative sequences/pairs |
|---|---:|---:|---|---|
| Complete11_11 | .598906 | .583816 | +.015090 [-.030979,.060233] |117/1986|
| RGB missing | .646492 | .602461 | +.044031 [.012752,.076987] |114/6573|
| TIR missing | .623651 | .641900 | -.018250 [-.055731,.018017] |115/6292|

All foreground concordance lower bounds exceed.5, but the required.05
advantage over background is absent in all3point estimates; complete and
TIRmissing difference intervals span zero. Registered gate=false. Directional
RGBmissing evidence does not satisfy the all3condition gate. Background
attention also carries localization information; this does not prove that
either map independently verifies original-target identity.

Fixed foreground*Hann heuristic selected-IoU minus native, complete/
RGBmissing/TIRmissing, is-.018731[-.051021,.011491],
-.015895[-.036386,.004185],-.007360[-.027912,.011773].
None establishes improvement. Do not promote background as an alternative
from its exploratory scores. Close this fixed block6 attention signal;
no learned ranking expansion, layer/mask/weight search on this cohort,
official evaluation or module claim. It does not reject every possible
attention/identity method.

Server and local independent arithmetic verifiers passed. Checked all9216
candidate boxes/IoUs, stable top8 membership/order, foreground/background
pair counts, sequence bootstrap/gate, input identities/frame exclusions,
listed source/checkpoint/cache hashes and physical visibility.
Independently reconstructed query coverage from stored template annotation.
Nine first-item rawqkv cases reconstructed in NumPy float64 have maximum
attention error1.4117185e-9. Separate normalized-image firstbatch replay
covered144inference rows (16x9); original score/size/offset maps matched
bitwise and baseline state/gradients were unchanged. Hook/nohook parity
passed9firstbatch calls. Replay shares the native network, does not decode
raw dataset images independently or replay all128input images. No external
scientific reviewer acceptance is claimed.

First verifier attempts stopped at an overly strict equality comparison of
prior-report hash dictionaries. The independent verifier additionally hashes
11older reports; producer-listed hashes all matched and none were missing.
Corrected to require EVERY producer-listed hash match, retain all21
independently reconstructed prior hashes, and still verify all frame
exclusions. Local/server_v2 checks then passed; original failure log retained.
Producer source, data, sampling, gate, protocol and outputs never changed.
This was a verifier correction, not a new experimental trial.

ReportSHA256:
`a7e8afa27171abd35ea53d963ae11d71f25c2f7b8145d4b2cef0528d81bb18eb`.
WorkerSHA256:
`22cb23b74a388da0496860bd01c303a749a50ae4c9a146f545c4e9b2fc4de05b`.
Evidence reports/server28_template_attention_20261102/analysis_server_v2.json,
analysis_local_v2.json, report.json and original verifier failure log.
Normalized image/map caches stay on server, not GitHub. No optimizer,
new weights, checkpoint cleanup, val97 or official benchmark access.
Final GPUs idle38/37/42/37C,all250W. Official best metrics unchanged;
complete-input approximately+2,RGBT234-Miss targets and3modules remain unmet.
Future fresh samplers must additionally exclude ALL template/search frame
ids in this report's input_manifest; do not reuse these tuples as a new audit.

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
