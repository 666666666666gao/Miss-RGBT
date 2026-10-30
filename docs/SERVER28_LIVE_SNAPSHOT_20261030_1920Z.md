# Server 28 Live Snapshot

Observed 2026-10-30 19:20 UTC.

Fixed20 utility/random remain live on GPUs0/1, epoch5, latest log updates
10775/11125; recorded loss fields are finite. Batch10 utility/random
candidate processes and controllers remain live on GPUs2/3, epoch3,
updates4775/4900. Peak allocated memory is21118406144 bytes per process;
only epoch1/2 fixed128 task-loss validations have completed. Recent speed
projects the next random/utility epoch boundaries near20:17/20:26 UTC.

Historical A utility20 LasHeR-Miss evaluation PID4138772 remains live with
one worker on each of GPUs0/1. There are12/479 total receipts and79206
completed frames, all receipts from LasHeR-Miss. No complete metric result
exists.

The first12 receipts aggregate6160.9 worker-seconds over79206 frames,
12.86 frames per worker-second. At two workers this projects roughly2h47m
remaining for the337353-frame evaluation, around22:08 UTC. This is a rough
rate estimate from only the first12 longest sequences.

GPU0-3 temperatures were84/80/79/75 C, power limits250 W, and utilization
98-100%. Previously verified temperature targets remain85 C. No process was
restarted or modified. This is health/progress evidence only, not benchmark
results or validated contributions.
