# Matched Local Dense Geometry Training

## Question and prior evidence

Native task_loss regresses the single predicted box, gathered at one score
peak. Target-peak-only supervision already failed; do not repeat it.
Fixed top8 box averaging also harmed native localization. Neither result
proves off-peak regression is the cause. This distinct paired optimization
test supervises a fixed3x3 target neighborhood while retaining single-peak
native Hann inference. It never reuses or tunes the failed averaging rule.

Dense target-supported regression/center sampling is prior art, not a new
module: official FCOS commit a773f1281275cf1e1cdaa0ca8c4a06b33036bb71,
fcos_core/modeling/rpn/fcos/loss.py implements get_sample_region,
location targets and positive-location regression. This pilot uses the
existing CENTER cx/cy/w/h geometry, not FCOS l/t/r/b or its implementation.
No third-party source/dependency is copied or installed.
https://github.com/tianzhi0549/FCOS/blob/a773f1281275cf1e1cdaa0ca8c4a06b33036bb71/fcos_core/modeling/rpn/fcos/loss.py

## Fixed prospective protocol

- Original frozen E0 SHA d20c8641e4263df3ceadf9e430d0ac96c181c13ad482bca28c2dddef592a8bb8.
  Unchanged server28 Python3.7/Torch1.12 runtime, GPU2 idle/<85C/250W.
- Reuse train882705/88/89 fitting/development/audit name partitions.
  Fresh256/128/128 tuple seeds62101/62102/62103; original augmentations,
  complete initial template, first search, all9 template/search states.
  Mask sampling visibility only to exclude every historical template/search
  frame, including all box-consensus input IDs. Physical images/GT unchanged.
  Reused development/audit names are not untouched data. E0 exposure unknown.
- Cache frozen float32 fused maps and verify native head output bitwise.
  Two copies of pretrained CENTER head, all6,474,117 parameters trainable,
  BN eval; same initialization, order and optimizer schedule.
- Native arm regresses at current score*Hann argmax. Dense arm replaces
  that regression with the arithmetic mean over the fixed3x3 neighborhood
  of the original GT heatmap peak, restricted to valid16x16 positions.
  Each selected position predicts the same clipped GT box. Coefficients
  remain2GIoU+5L1+focal; regression strength is not multiplied by9.
  Both use the SAME original full-map focal loss. No GT is used at inference.
- Witness on first16 fitting rows: dense counts4..9, onehot regression at
  native Hann peak matches native regression, native-vs-dense map gradients
  are finite, and dense offset gradients vanish outside fixed support.
  Save actual native maps/targets/losses/masks/gradients for independent
  arithmetic. This checks the objective, not training effectiveness.
- Profile one dense training update at batch64, including Adam moments.
  Estimate80% GPU allocation from actual incremental peak; choose largest
  multiple32 divisor of2304 not exceeding estimate or2304. No fake row
  duplication, arbitrary cap or OOM retry. LR=1e-5*batch/8, AdamW decay1e-4,
  clip0.1. Record actual train allocated/reserved peaks and updates/epoch.
- Max20epochs, LR divided10 from epoch7. Fresh stopping reference at7,
  improvement0.2%, patience3, min10epochs. Select minimum all9-cell
  native-Hann development task loss among last5, ties earlier.
  All checkpoints retained through verification. This finite cached-head
  pilot does not establish full-backbone convergence.
- Audit sampled only after BOTH arms stop/select; measure E0 and both
  selected heads with native single-box Hann decoding. Save per-row boxes,
  targets, losses and IoUs. All original model state/gradients and head BN
  buffers unchanged; finite weights, source/base/cache/weight hashes.

## Advancement

Only authorize fresh train882 closed-loop validation if dense minus E0
sequence-macro IoU>=.01 with lower95%>0 in BOTH missing directions,
dense minus native-training pooledmissing lower95%>0, and dense minus E0
mean>=0 with lower95%>-.005 in EACH11_11/01_11/10_11 cell.
Use10000 sequence-bootstrap draws, seeds62301(missing),62302(control),
62303(complete cells). Each contrast must include>=32 sequences.
Verification must pass identities/exclusions, objective arithmetic,
LR/stopping/final-five/selection, hashes, finite weights and selected-head
cached replay. No training control promotion on its own.
Failure closes this exact treatment; no neighborhood/weight/audit retuning.
Passing is not an official gain or novel paper module.

No val97, RGBT234/LasHeR test images or official masks, test-based selection,
checkpoint cleanup or official evaluation. Large caches/weights server-only.
Expected5-15min including cache collection and verification; first estimate
from measured epoch throughput, inspect near completion at180-300s intervals
for live long jobs. An observation timeout never authorizes restart.
Overall complete+2, RGBT234-Miss targets and3modules remain unmet.
