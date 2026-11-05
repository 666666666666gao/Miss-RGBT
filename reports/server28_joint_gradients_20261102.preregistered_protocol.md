# Joint Complete/Missing Gradient Diagnostic

Previous late-checkpoint CPU single-update audit did not show negative raw
complete/auxiliary gradient alignment. Teacher-reliability cached-head
diagnostic also failed its conflict criterion. Neither justifies projection.
This prospective diagnostic tests a distinct question: original E0 full
parameter gradients on16fresh train-mode batches of4 train882 tuples.

Seed56101,64tuples from existing705fitting names. Native two templates,
three search frames,DropPath in train mode,allBN frozen,unchanged raw
task loss. Native availability probabilities.4/.3/.3 and patch dropout.1;
per-batch forward RNG56200+index. Complete and missing forwards execute
sequentially as formal training; teacher maps detached. No optimizer.
Complete gradient is grad(mean task_C); auxiliary is
grad(mean task_M+.1mean response_KL(M,C)). Compute both using autograd.grad,
record squared norms/dots/cosines for all parameters,backbone,head/fusion.
Assert unused parameter identities match and all gradients finite.
The three searches share templates and are not independent observations.

Input image/box digests,actual frame identities,availability arrays,patch
keep counts and mask digests retained. Exclude exact tuples of all prior
pilots including teacher-reliability. Existing name partitions are reused;
64tuples are not an independent test set. Baseline state hash unchanged,
all parameter.grad absent; no AdamW/clipping update is inferred from raw
gradient geometry.

Advance only to preregistration of a matched joint-training optimization
control if all-parameter mean cosine<=-.05,upper95% bootstrap bound<0,
and at least25% batches have negative cosine.10000batch bootstrap
draws,seed56301. Batch bootstrap is descriptive:sequences can recur and
this does not estimate independent sequence generalization.
Failure closes this raw-conflict explanation for this cohort. Do not tune
criteria,project gradients anyway,or claim a paper module.

GPU2 after occupancy<500MiB,temp<85C,250W check,warm runtime unchanged.
Memory diagnostic batch4 matches formal microbatch; no LR/no training.
Estimated2-5min,180-300s milestone checks. No val97,official test data,
new checkpoints or cleanup. Independent saved Gram/bootstrap/protocol/
source/base verification; no raw gradients/images stored for full replay.
Complete+2,RGBT234-Miss thresholds and three validated modules remain unmet.

## Result

Pending.
