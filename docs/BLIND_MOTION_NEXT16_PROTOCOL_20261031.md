# Fixed Motion Control On The Next16 Development Videos

The first four-video same-GPU control improved outage PR20/SR by
1.809524/1.336535pp, mostly during blind frames. It is too small to
establish a generally useful predictor. Keep the exact same constant
velocity rule and test the next16 complete train882 development videos.
No parameter fitting or validation/test threshold search.

The first control and its original evaluator/analyzer source are preserved
in Git commit `5d031312a759ba39c98d2e4c12e28bcdb260b46a`.
The evaluator now accepts an explicit cohort specification/output/GPU for
this larger run; the motion rule and native tracker remain unchanged.
Earlier receipts describe their original source hashes and Git version.

## Cohort And Execution

Use development sequence indices4..19 from the existing protected-pilot
input manifest, excluding the earlier four videos. These16 videos contain
7776 frames in full; do not truncate them or select by tracking outcomes.
They remain disjoint from705 training sequences and val97. They are new
to this motion comparison, though they belong to the already defined
train882 development pool, not a certified unseen benchmark.

GPU0/1 each run exactly one evaluator. Assign eight videos per GPU using
longest-video-first frame balancing; all arms/conditions for a video run
on that same GPU. No cross-GPU arm comparisons. Require3000MiB free on
each GPU before launching; previous evaluator925753 finished normally.
Original four GPU trainers continue. GPUs2/3 receive no evaluators.

E0 hold/velocity x complete/alternating-outage produces64 trajectories,
31104 tracking frames total. Use full videos with the same32-frame
11/01/11/10/11/00 pattern. No GT enters the predictor. Retain pre-gap
size, use true video-frame separation for observed velocity, and preserve
native model-call/template counters on blind frames.

Verify complete/no-blind/prefix equality per worker, frozen network state,
all trajectory/trace/GT hashes and per-video/aggregate metrics. Analyze
blind frames separately from recovery observations. Aggregate16 paired
video differences with10000 bootstrap resamples, seed2027, not frame
resampling or training-seed uncertainty.

## Timing And Receipts

The first3352-frame run took237.57seconds, about14.1fps under concurrent
training. Balanced15552frames per worker suggest roughly19minutes;
budget20-35minutes including scheduling/IO and analyzer overhead.
The controller checks subprocess completion every300seconds. Verify
the actual child PIDs/allocations once, then inspect near completion.
No automatic restarts on observation failure.

Controller `run_blind_motion_cohort28.py`;
launcher `launch_blind_motion_cohort28.py`.
Output `reports/server28-blind-motion-next16-20261031`, siblingcontroller
log and per-GPU logs inside the directory. Require final `report.json`
AND `completed.json`; each worker has its own completed and analysis
receipts. Ordinary CV remains a baseline, not a paper contribution.
