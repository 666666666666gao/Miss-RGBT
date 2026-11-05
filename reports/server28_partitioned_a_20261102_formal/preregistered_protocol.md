# Train882-only paired A replication

Registered before execution, 2026-11-02. This tests the reproducibility of
existing Utility versus Random selection under a new seed and a corrected
data boundary. Historical official-test maxima motivate no parameter choice.
It is not an exact historical replication: fitting now excludes development
and audit names and previously registered diagnostic frames.

## Fixed protocol

Unmodified CADTrack E0 initialization, checkpoint SHA256
`d20c8641e4263df3ceadf9e430d0ac96c181c13ad482bca28c2dddef592a8bb8`.
Retain original A architecture/objective, all BN frozen, two templates and
three searches. Only routing selection differs: utility versus random.
Shared parameter initialization and training seed2028, selector seed2028.
Native complete/missing probabilities .4/.3/.3, patch mask .1, forward KD .1,
reverse weight0, utility supervision .1 every4microsteps.

Use existing LasHeR train882 order: index modulo5 nonzero gives705 fitting
names; remaining first88 development and last89 reserved audit. Never open
val97, LasHeR test, RGBT234, or official masks. Exclude ALL prior registered
frames via the existing exclusion chain plus dense-geometry train/dev/audit
manifests. Smoke inputs also excluded from formal runs. Log actual sampled
sequence/template/search IDs and tensor hashes for every microbatch. Only
sampling visibility changes; image content and physical annotations do not.
This is internal development; E0 exposure to all train882 is not denied.

Native augmentation retained for the fixed128development tuples, seed63102,
batch4. Cache once, evaluate without patch dropout with complete templates
and searches11/01/10; equal-weight mean native task loss. Fork Torch RNG and
restore selector RNG after each validation. Separate fixed development cache
from705fitting. It is not PR/SR or a closed-loop generalization guarantee.
Epoch-specific RNG reset makes training data independent of model RNG use.

Physical GPU2 Utility and GPU3 Random; occupancy<500MiB, temperature<85C,
existing250W power caps required at launch. Microbatch5, accumulation2,
effective10, LR1.25e-4, backbone multiplier.1, AdamW weight decay1e-4,
gradient clipping.1; this batch occupied approximately22.7GiB historically.
20000tuples/epoch, maximum20, LR drops after10. Reset early-stop reference
at11; relative improvement.002, patience3, earliest stop14.
Budget limit is not evidence of convergence.

Preserve all checkpoints for this campaign. Select the same checkpoint for
all final conditions by minimum fixed development task loss within the last
five epochs, ties earlier. Do not use official scores for selection/cleanup.
No evaluator is invoked automatically by this launcher. After independent
training/selection verification, report official FULL complete and Miss
metrics for final-five checkpoints as a disclosed post-hoc comparison.

## Short-run gate

Before formal launch run both arms for6epochs,20tuples/epoch, LRdrop2,
development8tuples seed63202, training seed63201. Exactly12optimizer updates
per arm. Check finite loss/gradient, parameter updates, paired input/mask
hashes, disjoint partitions/prior exclusions, final-five hashes and selection
arithmetic. Preserve failed evidence and do not silently restart.
No favorable smoke loss is required or interpreted as method benefit.

Expected smoke2-4min; formal roughly80-100min/epoch,19-34h until14-20epochs
on two GPUs, to be revised using measured timing. Poll actual processes near
milestones at180-300s intervals. Remote workers continue independently; no
claim of a launched formal run until PID/GPU/log state is verified.

## Interpretation

Compare internal paired development trajectories, stopping behavior and later
full official reports. One additional seed cannot establish robust method
superiority. If Utility fails to improve, report that result without tuning
on official tests or promoting Random as a novel module. Existing complete
approximately+2 and RGBT234-Miss targets and three validated paper modules
remain unmet.
