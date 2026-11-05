# Matched Full-Image Crop Exposure Training

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
