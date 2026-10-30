# Paired decoding control for native-scale grid

The completed51-candidate Hann audit restored center coverage at9of12
unsupported points, but only1had any predicted box reachingIoU>=0.5.
That result combines representation, box regression, crop centering and
Hann decoding. It cannot isolate the neural representation as the cause.

This control uses exactly the same46points and51proposals, historical A20,
initial templates and availability. Each forward pass is decoded twice:
native Hann-weighted score map, and raw score map without Hann. Both use
the same size/offset maps and native cal_bbox. No new proposals, GT-based
selection, fitted selector, threshold change or training update.

The standalone audit now accepts --output and retains raw_bbox,
raw_confidence and raw_diagnostic alongside the original Hann fields.
Completed prior outputs are preserved. Source hashes distinguish revisions.
This is an explicit controlled experiment, not a change to live tracking.

CPU worker4063222 launched17:44:20Z on SSH2028 using four threads and CUDA
disabled. Source audit_native_grid28.py; launcher
launch_native_grid_decode28.py. Python3.7 compilation passed.
At18:02:12Z it was live, elapsed17:58, the same46-point audit at36/46.
Elapsed is already longer than the prior21.91minute run's expected remaining;
re-estimate using recent point cadence and inspect near completion with
180-300seconds spacing. Do not start duplicate work.

Launch receipt started17:44:20Z. At17:45:57Z, elapsed1:43, CPU383%,
RSS1.05GB;3observations completed87.35s after initialization. Later
progress at18:02:12Z was36/46; the initial finish estimate was optimistic.
Use recent point cadence from the server log to estimate its remaining time.
All four training PIDs were live18:02:12Z.
Analyzer asserts matching observation identity/crop boxes and all valid
Hann bbox coordinates against the completed prior report before reporting
raw comparisons (rtol1e-4/atol0.05pixels).

After completed.json and terminal worker state are verified, run:

```bash
cd /home/gaob/Miss-RGBT
conda-envs/miss_rgbt/bin/python analyze_native_grid_selection28.py \
  --report-dir reports/server28-native-grid-decode-20261030 --raw
```

Compare all Hann predictions with the completed prior report before
interpreting raw/Hann differences. Report raw and Hann native, actual
peak-selected and GT-oracle outcomes, per state and native-unsupported
cohort. Raw oracle is only a candidate-box ceiling. If raw boxes still
cannot localize, an acceptance-only change cannot fix these candidate
sets. If raw has a better ceiling, it still needs a causal matched
selection/acceptance trial on train882 and full val97; no oracle-based
deployment or parameter tuning on these46points.

No recovery gain, novelty, final benchmark score or validated paper module
has been established by this pending control.

## Completed, 2026-10-30

Worker4063222 completed46 observations/2346 candidates in1355.57seconds;
process is absent and completed.json/report.json are present.
The analyzer confirmed every valid Hann bbox matches the earlier run with
maximum coordinate difference0pixels. Raw-score selected meanIoU is
0.431677, versus Hann-selected0.485004 and native0.564517. Raw peak causes
11 drops>0.1IoU versus native; Hann causes5. Raw oracle mean0.656347 versus
Hann oracle0.647675, just0.008672 higher, with the same33/45 predictions
atIoU>=0.5. In native-unsupported points, both decoders have just1/12
oracle successes. Removing Hann is not a useful recovery path; this
experiment does not identify which candidate representation/regression
component limits localization. See
docs/NATIVE_GRID_AUDIT_20261030.md and raw selection_analysis.json.
