# Matched Local Dense Geometry Training

## Completed result

GPU2 PID1178458 completed normally in280.274seconds. Batch2304/LR.00288,
one full-batch update/epoch; native ran20epochs and hit the maximum with
continued development improvement. Dense stopped14epochs at patience3.
Last-five development-loss selections: native epoch17 of16-20; dense
epoch13 of10-14. Both final-five sets were independently rehashed and all
10 checkpoint state dicts loaded finite with BN buffers matching E0.
Actual peak allocated memory was9440MiB for each arm. This is a small
cached-head pilot, not complete-network convergence or VRAM saturation.

Audit128 tuples covered74 sequences; all nine conditions in each arm.
Selected heads always use original single score*Hann argmax at inference.

| Contrast | Mean IoU delta | 95% sequence-bootstrap interval |
|---|---:|---|
| Dense minus E0, RGB-missing search | -.006127 | [-.028993,+.015274] |
| Dense minus E0, TIR-missing search | +.022116 | [+.001969,+.046957] |
| Dense minus native-trained head, pooled missing | -.010682 | [-.019862,-.001967] |
| Dense minus E0,11_11 | +.011744 | [-.012965,+.044125] |
| Dense minus E0,01_11 recovery | +.022820 | [-.012885,+.065286] |
| Dense minus E0,10_11 recovery | +.037759 | [+.003800,+.079068] |

Advancement=false: RGB-missing benefit absent; dense is worse than the
matched native-training control; complete protection interval crosses the
registered lower bound. TIR-only gain does not override these failures.
Close this exact3x3 treatment; do not tune neighborhood or weights on this
audit. It neither proves dense geometry universally ineffective nor
establishes the cause of the RGB/TIR asymmetry.

Server/local analyzers passed complete sequence pairing and seeded bootstrap,
all historical template/search frame exclusions and physical visibility,
training partitions, native-head map replay, full source/base/cache/report/
weight hashes, training history/update/LR/stopping/selection arithmetic,
and independent CPU reconstruction of all16 objective witnesses. Maximum
independent offset-gradient difference was0. First-five audit inputs were
not replayed; selected-head replay covered all3456 cached audit rows using
the shared native head/task-loss and a separately implemented Hann decode.
It is not raw-image/full-tracker replay or external scientific review.

ReportSHA256:
`de33c5b1912be26b4527df5bbdf579f995596680e1459d548ee675b21f308a70`.
WorkerSHA256:
`b23858e5f3f679af547047873ab4f96d85f8ba3b501833759fedea4f3880f217`.
ProtocolSHA256:
`8847da6ad48e2219336b0fe1c0554a11714cd4bd5c4f6f0fb04f0acbee686b21`.
AnalyzerSHA256:
`f5ac820966ef3aadf0a249c67b5d7e9509b634eae187debc8ebc1c1238019c3d`.
Small JSON/report/history evidence published; large feature caches and all
20 trained checkpoints remain server-only and retained. No val97, official
test, test-based cleanup or baseline mutation. Official scores unchanged.
Future fresh samples must exclude all manifest frames from this run.

## Actual launch and sizing

GPU2 PID1178458 launched2026-11-02T07:40:46Z. Worker immutable SHA256
`b23858e5f3f679af547047873ab4f96d85f8ba3b501833759fedea4f3880f217`;
original protocol SHA256
`8847da6ad48e2219336b0fe1c0554a11714cd4bd5c4f6f0fb04f0acbee686b21`.
At elapsed2:40, native arm reached epoch9/update9, GPU14396MiB/65C/250W.
Actual64-row dense update: static443497984bytes, peak720177152bytes,
total25296044032bytes. Registered estimate proposes4578 rows; actual
finite2304-row fitting cache bounds batch2304, LR.00288,1update/epoch.
No input duplication to fill remaining GPU memory. Peak training memory
is reported separately at completion. A few epochs with one update each
must not be described as full-model convergence. Expected3-5min remaining
including delayed audit; next observation after180-300s if still live.
No result or promotion was established at this observation.

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
