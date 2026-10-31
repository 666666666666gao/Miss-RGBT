# Server Reboot Recovery

Two actual reboots interrupted all original training/evaluation processes.
At16:25UTC+08 uptime was about1minute and boot ID was
`b605e4d9-1b12-4f55-96a1-c33b32a26603`.
At16:31 boot ID changed to `8212e607-b051-467d-a468-86436b0b702e`.
The first checkpoint audit became0bytes after the second reboot.
This is evidence for explicit flush/fsync in the replacement audit;
it is not evidence of the reboot's cause. No temperature/power changes made.

## Verified Checkpoints

`audit_reboot_checkpoints28.py` actually deserialized all four checkpoints
on CPU, checked finite model tensors and nonempty optimizer state, and
recorded SHA256/size/epoch/updates/LRs. Report:
`reports/reboot-checkpoint-audit-20261031.json`.

| Campaign | Restart After Epoch | Saved Updates | GPU | Controller / Trainer |
|---|---:|---:|---:|---|
| Fixed Utility | 10 | 25000 | 0 | 49952 / 50004 |
| Fixed Random | 10 | 25000 | 1 | 49953 / 49999 |
| Batch10 Utility | 9 | 18000 | 2 | 49955 / 50013 |
| Batch10 Random | 10 | 20000 | 3 | 49956 / 50008 |

The final boot remained unchanged through16:59. Recovery launched16:45:53.
At16:59 all four trainers were live and actual updates exceeded the
checkpoint by hundreds, with finite logged losses. GPU memory usage was
19346/19346/22732/22806MiB. No mere constructor/import success claim.

## Continuation Contract

`resume_training28.py` is separate from frozen original source/runtime.
It strictly loads model and AdamW state, checks saved LRs against the epoch,
and sets StepLR epoch accordingly. A real optimizer/scheduler witness
checked restored epochs9 and10 both produce the expected next LR.

The original checkpoint lacks RNG and scheduler serialization. New epoch
RNG streams use seed+100000+epoch consistently across each matched pair.
This is not bitwise uninterrupted continuation; loader draws and utility
probe history cannot be reconstructed. Candidate Utility replays epoch10
from epoch9; Random starts epoch11 from epoch10. Compare matching completed
epochs with this provenance, not simultaneous wall-clock progress.

Outputs are isolated in `runs/server28-recovery-20261031/<label>`.
Verified old epoch files are hardlinked there; unlinking a recovery link
does not delete the original file. Original logs/weights are preserved.
Do not describe recovery-link cleanup as global original-weight cleanup.
Original interrupted campaigns/controllers are terminal, not still running.

`recovery_campaign28.py` restores full final-five selection for all four
campaigns: candidate early-stop policy unchanged, fixed campaigns run20;
all final5 get full97 complete/outage metrics, one shared weight selected,
four recovery links removed with receipts, then official complete/Miss
evaluation. Thus the fixed controls now also get final-five selection.
Future evaluation success and cleanup remain unverified until execution.

## Development Evaluation

Controller72348 launched16:59, workers72352/72353 observed17:00:21.
`recover_state_offset28.py` checks source/spec/checkpoint hashes and prior
progress receipts, reuses only saved full trajectories with exact file
hashes and independently recomputed metrics, and starts unfinished videos
from initialization. Original partial files remain untouched.

Recovered directory:
`reports/server28-state-offset-replication16-recovered-20261031`.
The original full independent analyzer will verify all96 trajectories,
including reused past-offset traces and native/sham/complete equality.
This final report is not available yet.

`recover_evaluation_chain28.py` then runs the registered geometry4 and
gap4 specifications in new `-recovered` directories, using unchanged
original tracker/worker/analyzer code. It checks registered source hashes,
predecessor report receipts and >=3000MiB free on GPU0/1 before stages.
Original registrations and cohorts are retained; original PID fields are
historical and the recovery launch records current process provenance.
No new cohort/outcome selection or parameter tuning.

## New Validation Before Reboot

Random completed epoch10: complete4.2394807264, TIR-only5.9841034710,
RGB-only4.7672082037, mean4.9969308004.
Epoch9mean4.9724204093, so this proxy worsened. Utility epoch10 did not
complete before interruption. This is neither tracking metrics nor
convergence/official attainment.

Next useful nodes: recovered center16 after remaining long trajectories;
candidate Random epoch11 approximately18:15-18:35 and Utility epoch10
approximately18:20-18:40 UTC+08, revised from actual resumed throughput.
All official thresholds, three validated modules, final5 selection and
global nonselected-weight cleanup remain incomplete.

## Recovery Execution Evidence at17:09

All four training controllers/trainers and the evaluation controller/workers
were alive17:08. Recovered logged losses remained finite.
Worker logs contain both independently checked `REUSED` full trajectories
and a newly completed `TRACK` for whitegirl2 alternating outage.
Progress is82/96 tracks: GPU0 has24/30, GPU1 has58/66. Frame-balanced
assignment has5/11 videos, so48 tracks per GPU would be an incorrect count.
The6158-frame leftconergirl corrected complete and outage tracks remain;
do not omit this sequence or infer a full result from the partial cohort.
Provisional center completion17:45-18:15 UTC+08 plus up to300s queue wakeup.
Geometry/gap results remain unavailable until their registered stages run.
