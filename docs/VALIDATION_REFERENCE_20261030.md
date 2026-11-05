# Fixed-input validation reference

## Completed Results

Worker4024185 completed both128-sample references in1114.01seconds.
At17:42:16Z its process was absent, the log endedCOMPLETE, and
completed.json/report.json were present. Both models used the same CPU,
cached inputs, batch size and task-loss implementation.

| Reference | Complete11 | MissingRGB01 | MissingTIR10 | Three-state mean |
|---|---:|---:|---:|---:|
| Availability-aware E0 | 4.412175 | 6.750802 | 5.062622 | 5.408533 |
| Historical A utility20 | 4.398898 | 5.788143 | 4.799574 | 4.995538 |

Historical A lowers this sampled mean loss by7.636%; most absolute
improvement comes from missingRGB01. Complete loss changes only0.013278.
This is local task-loss evidence, not a causal contribution test or
long-term tracking improvement. Historical A did not consistently improve
complete official scores over E0; the task-loss proxy cannot replace
trajectory validation.

Candidate epoch1 GPU means are5.041177 utility and4.980912 random.
These are numerically close to the historical CPU reference; CPU/GPU
equivalence has not been established, and one epoch cannot prove
convergence or selector superiority. Keep the existing stopping and
last-five full-trajectory validation protocol.

E0 checkpoint SHA256:
d20c8641e4263df3ceadf9e430d0ac96c181c13ad482bca28c2dddef592a8bb8.
Historical A checkpoint SHA256:
c4aabefb495344b6d5e6e17655ac20fd51c4e2dec575843133096b319e047db3.
Full batch receipts and hashes copied to
outputs/server28-validation-reference-20261030.

## Launch History

CPU worker4024185 launched on SSH2028 at17:22:36Z on2026-10-30.
Source: audit_validation_reference28.py, launch_validation_reference28.py.
Output: reports/server28-validation-reference-20261030 and sibling log.
Python3.7 compilation passed. Process verified alive at16seconds, CPU338%,
RSS approximately1.4GB, four CPU threads, CUDA disabled.
At17:24:24Z the same worker was live, elapsed2:02, CPU389%, RSS1.7GB.
E0 completed16samples at75.95seconds after input preparation. Budget about
20-25minutes for both references; next useful check around17:43-17:48Z,
with180-300seconds spacing near completion.

Recreated128 cached candidate validation tuples, seed1701, batch4.
Input SHA256 matches both candidate training jobs exactly:
13105d2512b095da153e0ec8cd38cf6264c0971db7d40b036ddfc10a71455391.
This assertion passed before model evaluation began.

Compare the availability-aware E0 initialization wrapper against historical
A utility epoch20. Full templates; search availability11/01/10; model.eval;
no patch dropout or updates. Report native task loss per state and mean.
Checkpoint hashes, per-batch losses and script hash remain in receipts.

Development diagnostic only: CPU/GPU results are not assumed bitwise equal;
historical E0 exposure to val97 remains uncertified. This does not measure
closed-loop tracking or PR/SR, select official test weights, or change the
running training and stopping protocol. Completion requires report.json,
completed.json, and a terminal worker state.

At17:21:40Z all four GPU training workers were live. Power limits250W;
temperatures84/81/79/76C. Native-grid CPU worker4006613 was live at18/46
observations, elapsed9:16. Expected completion17:35-17:38Z; inspect near
that milestone with180-300s spacing. Avoid duplicate launches.
