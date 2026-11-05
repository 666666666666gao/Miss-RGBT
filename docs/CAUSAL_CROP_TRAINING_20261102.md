# Matched Full-Image Crop Exposure Training

## Verified amended result

The original16x96 audit was infeasible and remains incomplete. The separately
registered24x64 audit completed in346.833seconds without retraining. All24
unique audit sequences contributed. Each model produced3024 tracking rows;
9072 rows across E0/native/causal. This is train882 internal IoU, not official
PR/NPR/SR/MPR/MSR, and not96-frame or full-sequence robustness.

| Amended audit contrast | Sequence-macro IoU delta | 95% bootstrap interval |
|---|---:|---|
| Causal minus E0, RGB missing | +0.005559 | [-0.008252, +0.020946] |
| Causal minus E0, TIR missing | -0.003614 | [-0.033199, +0.024765] |
| Causal minus native-training, pooled missing | -0.002265 | [-0.016638, +0.009103] |
| Causal minus E0, complete trajectory | -0.002447 | [-0.019433, +0.018133] |

Amended promotion gate=false: neither missing direction clears the
effect/uncertainty threshold, matched-control superiority is absent, and
complete protection fails. Close this exact pilot; do not enlarge or retune
its audit, train an expanded version based on it, or count it as a module.
The previous-GT crop oracle benefit did not establish that this305pair
offline full-image exposure treatment transfers to deployment. These results
do not prove all trajectory-exposure learning ineffective.

Server and local analyzers independently recomputed stored box IoUs,
selection/stopping, bootstrap statistics, partitions/exclusions and hashes.
Server additionally regenerated the seeded24clip selection/schedules from
actual metadata, checked actualGT/visibility, image-cache tensors, source/base
hashes and all10final-five finite full-model checkpoints. BN buffers equal
E0. Each selected model changes151backbone and58head/fusion tensors; both
parts were actually trained. A separate-process native-image first-audit-clip
replay reproduced378boxes/confidences exactly across3models/2trajectories.
This shares native tracker/crop utilities; not full-audit image replay or
independent scientific reviewer acceptance. Earlier first-fitting-clip
reconstruction reproduced10accepted pairs and1rejection.

Report SHA256:
`9502da9b3fcbaf14caf5c604fe2a29a7b1d308afa8563f1bb1fac32728147a02`.
Evidence: `reports/server28_causal_crop_training_20261102/analysis_server.json`,
`analysis_local.json`, `report.json`, amendment/launch snapshots and failure
log. All original/final-five weights remain on server28; no test-based cleanup.
All GPUs idle at final check,250W limits unchanged, temperatures38/38/46/37C.
No val97 or official tests. Official scores unchanged; complete-input gains,
RGBT234-Miss targets and three validated paper modules remain unmet.

## Basis and scope

The registered48clip causal previous-GT crop diagnostic passed, mixedmissing
IoU delta+.042196 CI[.022036,.066595]. This is an unavailable oracle, not a
trained policy. Complete-input oracle gains also occur. Test the distinct,
trainable question: does supervision on crops from actual frozen-E0
prediction trajectories help a trainable full-image tracker more than
ordinary GT-jittered crop supervision with the same images and templates?
This is an ordinary trajectory-exposure control, not a novel paper module.
Prior-art boundary: CAUSAL_CROP_PRIOR_ART_20261102.md.

## Preregistered design

- OriginalE0 checkpoint SHA d20c8641e4263df3ceadf9e430d0ac96c181c13ad482bca28c2dddef592a8bb8.
  Reuse unchanged server28 runtime/environment. All original model parameters
  trainable, all BN statistics frozen; native train-mode DropPath, native
  task loss, no adapters or parameter-count changes.
- Train882 existing705/88/89name fitting/development/audit partitions.
  New32 fitting96frame clips(seed58101),8 development48frame clips(58102),
  and16 audit96frame clips(58103). All unique within split, finite positive
  visible GT. Exclude frames in previous tuple/rollout diagnostics including
  causal-crop48. Shared names were used before, not pristine sequences.
  Audit clips selected/loaded only after both arms stop and select.
- Fitting collection runs one frozen-E0 mixed IID trajectory per video,
  .4/.3/.3 complete/RGBmissing/TIRmissing,initial complete.
  Every8th tracked frame supplies one candidate paired training sample.
  Snapshot native templates, availability, previous prediction, actual
  search image and current GT before native tracker update.
  Two crops from the same current image: native factor4 crop centered on
  previous E0 prediction; ordinary factor4 crop around current GT jittered
  with center3/scale.25 as native processor. Both output256pixels.
  Templates128pixels from the same native history.
  Require full GT extent in BOTH crops and positive valid labels; reject
  the candidate jointly otherwise and report reasons/counts. At least128
  accepted samples. No GT-driven trajectory resets, crop resizing policy
  change, label clipping as a rescue, or fabricated padding samples.
  GT is used only as normal fitting supervision/inclusion, never inference.
- Save normalized float32 image tensors (not backbone features), target
  boxes, identities, crop boxes and input hashes in a server-only image
  cache. Template availability and same current state shared across arms.
  Every training forward reruns full network from image tensors.
- Native arm: complete branch alltemplate/searchavailability1 on GT-jitter
  crop; auxiliary branch native storedtemplate/currentavailability on SAME
  GT-jitter crop. Causal arm: exactly the same complete branch, auxiliary
  branch on predicted crop. Optimize mean(taskC)+mean(taskM).
  No patch drop or KD in either arm. Different crop coordinates make direct
  response-position KD invalid; no synthetic response alignment is added.
  Thus these controls differ from historical A full campaigns.
- Sequential branch backprop before one shared clip/AdamW step reduces
  live activation memory. Same per-epoch sample order and paired forward
  RNG for both arms. OriginalE0 initialization identical; BN buffers fixed.
  No cached backbone/head features. No gradient through frozen collector
  trajectories: this is offline exposure, not online policy aggregation.
- Profile full-image optimizer step at batch4 using disposable model.
  Estimate largest multiple4 <=32 fitting within80% totalGPUmemory,
  including Adam states, record actual peak. No OOM retries.
  Batch reference8/LR1e-5, linear scale actual batch, backbone multiplier.1,
  AdamW wd1e-4, clip.1. Shuffle/drop_last, identical accepted samples;
  record dropped remainder per epoch, no duplicate padding.
- Max12epochs, LR x.1 startingepoch5. Minimum8epochs; reset stopping
  reference atepoch5, stop after3epochs without .2% improvement.
  Every epoch native closed-loop tracking on8development48frame clips,
  complete andIIDmixed schedules fixed before fitting.
  Score is mean of sequence-macro complete IoU and sequence-macro mixed
  missing IoU. Larger is better; select best among final5, earlier on ties.
  Keep all weights until verification; retain all final5 and original.
  This finite pilot does not estimate full train882 convergence.
- After both select, sample16audit96frame clips once, fixedcomplete/mixed
  schedules. Run originalE0 and both selectedmodels in closed loop with
  native crop/Hann/templateupdate and no GT after initialization.
  Persist allframe boxes/confidence/state/GT and input hashes.
  This is internal train-split IoU, NOT official PR/NPR/SR/MPR/MSR.

## Promotion gate

Sequencebootstrap10000draws seed58301. Causal-minus-E0 audit IoU must
be>=.01 withpositive95% lowerbound in BOTH mixed RGBmissing andTIRmissing
conditions. Causal-minus-native-training pooledmixedmissing lower95%>0.
Complete-trajectory causal-minus-E0 mean>=0 andlower95%>-.01.
All16sequences must contribute in each main readout. Finite model/gradients,
identical initialization, fixedBN, metadata/hashes/partitions also required.
A pass authorizes a fresh larger image-training protocol, not immediate
module validation or target claims. Failure closes this exact pilot; no
retuning its audit, cropping policy or threshold based on result ranking.

## Execution

OneGPU2 collector and sequential full-image fitting arms; may use GPU3 for
read-only audit if idle but not required. Check occupancy<500MiB,temp<85C,
existing250Wcap atlaunch. CPUimage caches consumeRAM/disk, not published.
Estimate20-40min includingcollection/train/development/audit; refine ETA
from actual epochs. Observe180-300s near useful milestones; do not relaunch
on observationtimeout. Preserve rawfailure evidence. No val97,LasHeRtest,
RGBT234test,officialmasks or official test-based selection/cleanup.
Officialmetric goals and threevalidatedmodules remain unmet.

## Input verification during the live run

The producer source remains frozen at launch SHA
430a563f5db57d56c25ab170a78b0ad2c9926f27a9b85a99a8f061e968204fd3.
352candidate fittingpairs across32videos yielded305accepted pairs,
125complete/87RGBmissing/93TIRmissing. Of47joint rejections,46were
GTjittercrop invalid and1predictioncrop invalid. All32videos contribute.
Batch32,LR4e-5/backbone.1,9updates/epoch;17samples dropped after the
shared per-epoch shuffle. Native full-image peak allocated/reserved
22,169,819,648/22,636,658,688bytes;observed occupancy23,350MiB/74C/250W.

GPU3 independently reconstructed the first fitting clip `whiteboy242`,
frames117-212,without updates. Its10accepted paired search-image tensors,
template tensors and availability matched the saved cache bitwise; the
one rejection was reproduced. Predicted and seeded jitter crop boxes
matched exactly. Independently written NumPy target-coordinate arithmetic
had maximum error1.7149784e-7. This uses the shared native tracker, image
decoder/crop/preprocessor; it is first-clip input reconstruction, not
full-cache or optimizer replay, performance evidence or external review.
Receipt: `input_replay_first_clip.json`. Verifier source SHA
080541c736588da0df8f3bce9a830e73fe8a9f8fbab16f3b1bdafc47ceaa18c3.

At worker19:20 native completed12epochs/108updates and choseepoch11
fromfinal5=8-12. Causal reachedepoch7/63updates and remained live.
Native development scores fluctuate; max12 is a pilot cap, not proof of
full-dataset convergence. Audit remains delayed until both selections.

## Original execution failure and prospective recovery

Original PID877938 subsequently stopped: native12epochs/108updates,
selected11(final8-12); causal9epochs/81updates,selected6(final5-9),
stopped after3postdrop stale epochs. Both103,138,437parameter networks
were fitted from images; their initial state hashes match.
The worker then asserted in audit fresh_clips, before saving audit_clips
or reading audit images/predictions. Original16x96 protocol is incomplete.

Metadata-only inventory under the same89auditnames and exclusions finds
12 eligible96frame sequences,24 at64frames,29 at48frames. Inventory's
initial NumPy integer JSON serialization failed; partial file/log retained,
v2 corrects Python integer serialization. Neither inventory used images,
model predictions, official data or performance.

Before audit outcomes, registered separate
CAUSAL_CROP_AUDIT_AMENDMENT_20261102.md:24x64,same seed58103, unchanged
selections/partitions/exclusions, same thresholds requiring24contributing
sequences. Audit-only resume PID949348 onGPU2 at06:22:25Z. Original worker
source unchanged; all weights and failure evidence preserved. This amended
shorter-horizon readout cannot establish original96-frame success.

Operational lesson: before expensive fitting, check metadata-only audit
feasibility under existing exclusions, without choosing/loading audit
images or observing predictions. Delaying audit outcomes need not delay
this feasibility check. Future protocols must preregister feasible sample
counts and lengths; do not add result-driven fallback sampling.
