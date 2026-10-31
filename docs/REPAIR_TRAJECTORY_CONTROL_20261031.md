# Closed-Loop Repair Intervention

## Question

The completed epoch6 fixed128-tuple intervention finds Utility native-minus-
off RGB-only mean task loss-0.007465, but only62/128tuples improve.
Between-model differences largely persist when compensation is disabled.
That diagnostic does not establish a closed-loop localization benefit.

Test the retained same-epoch Utility and Random models under native,
recording-only sham and compensation-off inference. Off still executes
selection/correction and returns the exact pre-compensation RGB/TIR tensor
objects to the remaining network. It preserves selection RNG advancement
and computation overhead while removing the correction's output.
It is an inference intervention, not matched retraining, a selector-only
ablation, a performance optimization or proof of a novel module.

Both checkpoints are epoch6/update12000, already retained in the registered
original-weight inventory. They must remain intact until this analysis
finishes as well as the original final-verification cleanup conditions.
Utility SHA256:
`ec9b4b087f2f44f71831ada45c3c34ee82efaad4d5344e048aaf9a7ce526a527`.
Random SHA256:
`66fa8175a6f000e711da8577903eba386c2a8b03c94b504e68e7a82c5f4577ee`.

## Prospective Design

Sequential fresh development76..83, no outcome/duration filtering:

| Index | Video | Frames |
|---|---|---:|
| 76 | oldwoman | 265 |
| 77 | blkboy | 329 |
| 78 | Aboydownbike | 986 |
| 79 | girl2trees | 223 |
| 80 | thefirstexcersicebook | 673 |
| 81 | higherthwartbottle_quezhen | 4221 |
| 82 | boyrun | 168 |
| 83 | whitegirlatbike | 194 |

Total7059frames. GPU0 receives the4221-frame video; GPU1 receives2838frames.
All videos and arms use both complete and existing32-frame alternating
11/01/11/10/11/00conditions:2models x3arms x8videos x2conditions
=96trajectories. Initialization/nonpositive GT exclusion and PR20/SR use
the existing development metric protocol. No official test score selects
cohort, weight, branch, threshold or parameter.

## Checks And Scheduling

The seven-observed-call CPU witness uses11/01/10 inputs on each model.
It requires native/sham box and confidence identity; off output identity
to a separate run with the utility module disabled; exact hook-call counts;
and full frozen-model digest identity after restoring the selection RNG.
This limited CPU witness does not certify CUDA or long-trajectory behavior.

GPU workers record source, checkpoint, cohort, GT, trace and box hashes.
The analyzer independently recomputes all96trajectory metrics, checks
full native/sham equality, observed-call hook counts and blind holds,
then reports native-minus-off for each model plus between-model native
and off contrasts. Paired-video bootstrap uses10000draws/seed2027.
Frozen-network restoration is a worker witness, not independent replay.
No training-seed uncertainty or isolated selector causality follows.

New isolated entrypoints: `run_repair_trajectory28.py` and
`launch_repair_trajectory28.py`. They import unchanged helpers from the
registered state-template runner; keep both experiments' dependencies
unchanged through final analysis.
Output `reports/server28-repair-trajectory8-retry-20261031`, sibling `.log`.

Controller waits on the actual preceding state-template controller PID
every300seconds. Its exit must be followed by the verified48trajectory
report and controls. Only then, after observing at least3000MiB free on
GPU0/1, does it start one worker per card beside the existing trainers.
No new GPU worker is authorized merely by queue registration.
Estimate60-120minutes after predecessor release plus up to300seconds
observation delay. Refine from the longest full native trajectory.
Do not poll early, restart on observation timeout, or remove the long video.
No trained weight or deletion is produced by this diagnostic.

## Queue Registration

Python3.7 compilation passed. The seven-call-per-model/arm CPU witness
confirms native/sham trajectory identity, off/directly-disabled-module
identity, restored selector RNG and unchanged frozen-network digest.

The first CPU witness passed, but controller366251 exited before GPU
allocation. The witness recorded `__file__` as a relative path when directly
invoked, while the launcher uses an absolute script path; exact manifest
comparison therefore failed on the path key. The original launch receipt
and failure log are preserved; no GPU worker or output directory was created.
The corrected `sources()` resolves its own path before hashing, and the retry
uses a distinct output/launch path.

The retry controller waits on the actual
state-template controller PID342664 every300seconds. No repair-control
GPU workers start until that process exits, its full48-trajectory report
hash and completion receipt agree, and GPU0/1 each have at least3000MiB
free. Source SHA256
must be read from the retry witness and launch receipt.
Queue and sources remain frozen through final analysis.
Retry controller393144 registered at20:37:03 UTC+08. It was verified alive
and waiting on controller342664 at20:37:58 and20:44:51; no repair-control
GPU workers existed during that wait. After controller342664 wrote the
verified48-track report/completion receipt and exited, the retry checked
both GPUs for at least3000MiB free and launched workers418802/418803 at
20:52 UTC+08. At20:56:32 controller393144 and both worker command lines were
actually alive. The completed predecessor report SHA is
`66b3b776e0152f01bf0115e322e371ca7e15ec0c7b1576e1f8550f11300ad3e9`.
Repair workers have begun utility/native on fresh video sequences. At
21:06:50 they remained live alongside all four trainers. GPU0 had two track
receipts for the4221-frame `higherthwartbottle_quezhen` (complete PR20/SR
42.393365/27.807493; block32 7.274882/8.803882). GPU1 had34 receipts,
including utility native/sham and off beginning; these are partial per-arm
records, not a paired aggregate. The retry CPU witness passes parity/frozen
state checks, and its absolute self-source hash matches the current source.
