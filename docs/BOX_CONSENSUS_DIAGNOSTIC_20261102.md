# Fixed Candidate Box Consensus Diagnostic

## Completed and verified result

GPU2 PID1131961 launched2026-11-02T07:25:43Z, completed50.036seconds.
128fresh fitting tuples from117 sequences,1152 state rows. No optimizer,
new weights or official data access. The fixed estimator failed its gate.

| Population | Weighted minus native IoU | 95% sequence CI | Uniform minus native IoU |
|---|---:|---|---:|
| Complete11_11 | -.061691 | [-.091699,-.034760] | -.158898 |
| RGBmissing,3template states | -.075441 | [-.104439,-.047579] | -.108520 |
| TIRmissing,3template states | -.117869 | [-.152202,-.085691] | -.184741 |
| Recovery01_11 | -.078797 | [-.112600,-.048098] | -.171067 |
| Recovery10_11 | -.079187 | [-.112631,-.049165] | -.183629 |

Each category includes117 sequences. Pooledmissing weighted-minus-uniform
is+.049976 CI[.040661,.059431], but weighted-minus-native is-.096655
CI[-.121425,-.072691]. Beating a worse averaging control is insufficient.
Close this exact fixed top8 mean estimator; no candidate-count/temperature/
cluster search on these outcomes, uniform-control promotion or module claim.
Results do not reject every box-fusion method or identify a sole causal
reason for degradation. Geometry and response modes can differ; this
experiment did not isolate multimodality from off-peak regression error.

Local/server saved-evidence analyzers reconstructed all9216 candidate
box IoUs and1152 aggregate boxes, seeded sequence bootstraps/decision,
all paired identities/exclusions and protocol/launch linkage. Server checked
all saved dense score/size/offset maps, stabletop8 membership/order, physical
visibility, source/checkpoint/cache hashes. Separate native construction
replayed first16normalized input tuples in9states (144inference rows):
all three native maps bitwise equal, unchanged full state/no gradients.
Replay shares native network and normalized inputs; not independent raw
image decoding/full128-input replay or external scientific review.

ReportSHA256:
`dedaec020a61dea618dd8320bec459fc3ec03956e67631ac811f08eae150806b`.
Frozen workerSHA256:
`a077431dbb73485da14e8cd70e4e0970c08f0be5ed686b79dd7847520d7d3df4`.
Immutable protocolSHA256:
`7840251b4c0403300cb553d1183182d7ab5389e4b5d40ae523b3ce01d3357da5`.
AnalyzerSHA256:
`c3768afd1cd4d374294c83ae9d8b1e87576a3ceaf35249527a53b852c78c6e85`.
Small evidence reports/server28_box_consensus_20261102/; large maps/first
images stay server-only. Future fresh samplers must additionally exclude
ALL template/search frame IDs in this report's input_manifest.
No val97, official evaluation, cleanup or validated paper module.
Official scores and full goal remain unchanged/incomplete.

## Question

Prior top-eight GT-oracle headroom did not transfer through the learned
quality ranker or template-attention selectors. Those methods choose one
existing box. This distinct control tests whether averaging their geometry
reduces localization noise. It can also merge separate response modes and
harm accuracy; no gain is assumed.

Confidence-weighted box coordinates are established prior art:
ZFTurbo/Weighted-Boxes-Fusion commit
796d4b1b6125ebef795cf14363dae00da78d5a1c,
ensemble_boxes/ensemble_boxes_wbf.py, get_weighted_box.
The inspected source sums confidence times coordinates, divided by summed
confidence. Here there is one tracker and no IoU clustering or multi-model
confidence rescaling. This is a simple fixed control, not WBF replication
or a novel paper module. No third-party code or dependencies are installed.
https://github.com/ZFTurbo/Weighted-Boxes-Fusion/blob/796d4b1b6125ebef795cf14363dae00da78d5a1c/ensemble_boxes/ensemble_boxes_wbf.py

## Prospective protocol

- Original frozen E0, unchanged server28 Python3.7/Torch1.12 runtime; GPU2
  must be idle, below85C, existing250W limit. No optimizer or new weights.
-128 fresh fitting705 tuples from train882, seed61101, batch16, original
  augmentations and template order, complete initial template, first search,
  all nine dynamic-template/search states.
- Exclude every template/search frame from all predecessor records,
  including all128 template-attention input manifests, by sampling-only
  visibility masking. Physical images, boxes and visibility are unchanged.
  Assert actual sampled IDs do not overlap. Reused sequences are not
  independent data; historical E0 exposure remains uncertified.
- Fixed stable top8 native score-times-Hann candidates. Native is rank0.
  Weighted estimate is the arithmetic mean of cx/cy/w/h weighted by those
  positive Hann scores, normalized to sum1. Control is the unweighted mean
  of the same eight boxes. No temperature, threshold, clustering, candidate
  count, mask or parameter search. GT only scores outputs.
- Record all candidate boxes/scores/IoUs, target, aggregated boxes/IoUs and
  per-row differences. Require candidate0 equals native CENTER Hann decode,
  finite positive geometry and unchanged model state/no parameter gradients.
- Primary categories: complete11_11 and each missing-search direction
  averaged over three dynamic-template states. Also inspect recovery cells
  01_11 and10_11 separately. Sequence-macro means and10000 bootstrap draws,
  seed61301. Report uniform and weighted outcomes without promoting control.
- Save all native maps on server; first16 normalized input tuples enable
  separate native-image-forward replay. Independently reconstruct candidate
  membership/order, dense decode, weighted arithmetic, boxes/IoUs, effects
  and decision. Caches stay server-only. Replay shares native network and
  normalized inputs; it is not independent raw-image decoding/full replay.

## Advance criterion

Authorize only a fresh train882 closed-loop diagnostic if:

1. Weighted-native IoU mean>=.01 and lower95%>0 for BOTH missing directions,
   with>=64 sequences in each category.
2. Weighted-uniform pooled-missing IoU lower95%>0, with>=64 sequences.
3. For EACH complete-search cell11_11,01_11,10_11, weighted-native IoU
   mean>=0 and lower95%>-.005, with>=64 sequences.
4. Saved-evidence arithmetic, exclusions, checkpoint/source/cache hashes,
   native decode, frozen-state/gradient and replay checks pass.

A pass is not official metric attainment or training/module validation.
Failure closes this fixed estimator; do not search weights/count/clusters
on these outcomes or promote the uniform control. No val97, official
LasHeR/RGBT234 data/masks, checkpoint selection or cleanup.
Expected1-3min plus verification. Inspect around180s only if still needed;
use180-300s observations for live long jobs, no timeout-based restart.
Full complete+2, RGBT234-Miss thresholds and three-module goals remain unmet.
