# Fixed Candidate Box Consensus Diagnostic

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
