# Paired Head Supervision Pilot

Previous fresh crop diagnostic found missing-input errors remain even at
the GT response peak. It did not support increased gradient conflict.
Compare ordinary predicted-peak regression versus target-peak regression
supervision, not a new module. Do not promote the failed fusion adapter.

Frozen E0 checkpoint SHA256 d20c8641e4263df3ceadf9e430d0ac96c181c13ad482bca28c2dddef592a8bb8.
Reuse705/88/89 train882 fitting/development/audit name partitions; new
seeds40101/40102/40103,256/128/128 tuples, all9 template/search states.
Initial template always complete; first search only, native crop augmentations.
Record input identities/hashes and check no exact tuple overlap with both
preceding pilots. Independence refers to adaptation name partitions only;
historical E0 exposure is uncertified. No val97 or official tests.

Cache frozen fusion features, verify native head replay. Two copies of
the pretrained native CENTER head, equal initialization and capacity.
Baseline model and BN buffers remain unchanged. Copied heads train all
parameters with BN held in eval; identical cached features and batch order.
Native arm uses predicted argmax boxes for original2GIoU+5L1+focal loss.
Target arm supplies the native heatmap only during training box decode;
original score-map focal term remains unchanged. Evaluation always uses
predicted response argmax and never supplies GT to inference.

All9 conditions included equally in fitting and development selection.
Batch64,AdamW LR8e-5 (reference batch8/LR1e-5),decay1e-4,clip0.1.
2304 fitting rows,36updates/epoch,max20;LR drops after6. Fresh stopping
reference epoch7,0.2% relative loss improvement,patience3,min10epochs.
Select lowest all9-cell native-inference development task loss among last5.
All small copied-head checkpoints preserved through audit; no official
checkpoint deleted. Cached-head plateau is not full tracker convergence.

Final audit only after both arms stop/select; all9 native inference cells.
Record loss and IoU per tuple and bootstrap by sequence(10000 draws).
Advance only to a fresh train882 closed-loop test if target arm:

1. Improves macro IoU by at least0.01 over frozen E0 in each missing-search
   direction, with positive95% lower bound, averaging template states.
2. Has positive95% lower bound on IoU versus native training across six
   missing-search cells.
3. Has nonnegative point IoU improvement versus E0 in each complete-search
   cell(11_11,01_11,10_11), with95% lower bound above-0.005.
4. Finite optimization/inference, native initial replay, baseline state
   digest unchanged and gradients absent; copied-head BN buffers unchanged.

Failure means no promotion, no official evaluation, no claim of a module.
Even a pass does not prove approximate+2 complete improvement or benchmark
thresholds. This bounded paired experiment tests supervision, not novelty.

GPU2 sequential arms; check occupancy before launch. No new dependencies.
Record actual peak allocated/reserved memory rather than duplicate finite
cache rows to fill VRAM. Expected3-8minutes, inspect near completion; if
still running poll180-300seconds. No automatic restart on observation timeout.
