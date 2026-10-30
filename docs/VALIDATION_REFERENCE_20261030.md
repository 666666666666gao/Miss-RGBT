# Fixed-input validation reference

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
