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

## Completed result

GPU2 PID4161273 launched2026-11-01T23:47:52Z and completed274.387seconds
normally. Native and target arms each stopped at epoch10/360updates;
development selected epoch6 from final-five6-10 in both arms.
Both heads have6,474,117 parameters. All fitting2304 rows and each1152-row
development/audit cache passed native head replay. Audit128 tuples sampled
70 actual sequences. Both arms began with identical pretrained head states.

Registered sequence-macro IoU effects, positive means target arm improves:

| Comparison | Mean IoU change | Sequence-bootstrap95% interval |
|---|---:|---|
| Target versus E0, TIR-only search averaged templates | -0.000202 | [-0.017034,0.017395] |
| Target versus E0, RGB-only search averaged templates | 0.001481 | [-0.015394,0.019496] |
| Target versus native training, six missing-search cells | -0.009119 | [-0.017295,-0.000888] |
| Target versus E0, full history11_11 | -0.006657 | [-0.024239,0.013553] |
| Target versus E0, recovery01_11 | -0.003889 | [-0.019001,0.015083] |
| Target versus E0, recovery10_11 | -0.017905 | [-0.033137,-0.005049] |

Registered advancement gate=false. Missing-direction gains do not reach
0.01 and their intervals span zero; the target arm is worse than native
training across missing cells. Complete and recovery protection fail.
Do not launch full training or official evaluation from this pilot.
The native arm is a control, not independently approved for promotion.
No ordinary target-index regression contribution is established.

For context only, tuple-mean audit IoU(native inference, not sequence-macro):

| Dynamic/search | E0 | Native training | Target training |
|---|---:|---:|---:|
| 11_11 | 0.813392 | 0.810294 | 0.803137 |
| 11_01 | 0.619726 | 0.632082 | 0.613738 |
| 11_10 | 0.754999 | 0.755372 | 0.751457 |
| 01_11 | 0.798621 | 0.798023 | 0.793576 |
| 10_11 | 0.806343 | 0.797997 | 0.788571 |

The target-only supervision changes where box gradients are applied.
Global clipping can also alter branch update scales; the experiment does
not identify a sole failure cause. Shared sequence partitions have been
used in earlier investigations, so fresh tuples are not untouched sequences.
Static cached crops omit tracker Hann-window decoding and prediction-driven
template updates. This is a head-objective control, not full convergence
or a closed-loop benchmark. Do not tune repeatedly against this audit cache.

## Verification

`analyze_head_supervision.py --verify-server-files --replay-native-heads`
passed. It independently recalculated histories, LR/stopping/final-five
selection,70-sequence bootstrap effects and the failed gate; checked source,
base/cache/selected/final-five hashes; directly loaded all10 final-five
weights and verified finite tensors and unchanged BN buffers against E0.
It independently checked705/88/89 name partitions and prior tuple exclusions.

The analyzer constructed native CENTER heads via a separate loader,
loaded E0/native-selected/target-selected weights and directly replayed
1152 audit feature rows for each. All3456 losses/IoUs matched exactly.
No GT map was supplied to these inference calls. This is actual cached-head
replay, not a backbone/image replay or external scientific review.
Baseline whole-state digest remained
4d64b467b82ff76963a8c32d13cb0e60980fef236df14546e7bceaddffdb6eb6;
baseline gradients absent. Source hashes do not cover all transitive imports.

Peak allocated GPU memory: native777,229,312bytes,target835,023,872bytes;
reserved1,090,519,040/1,092,616,192bytes. The finite cached-head diagnostic
does not saturate24GiB VRAM. No replicated data or larger LR was added merely
to fill memory. Full-model batch/LR profiling remains necessary if a method
clears its evidence gate.

Report SHA256:
5e67287881365117c978b89d382911d02d595fd7514ac344f0d00b845f95d4f9.
Small evidence:reports/server28_head_supervision_20261102/.
Initial protocol preserved aspreregistered_protocol.md,
SHA256b1904a570372563701ba0b26abb4073e81c1260a3bb217b5c2b4af6e0cb4a039.
All pilot/original/final-five checkpoints remain on server; no cleanup,
val97 or official-test evaluation. Official scores unchanged. Complete+2,
RGBT234-Miss thresholds and three validated modules remain unmet.
