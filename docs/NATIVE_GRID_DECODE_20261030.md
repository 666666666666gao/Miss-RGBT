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

CPU worker4063222 launched on SSH2028 using four threads and CUDA disabled.
Source audit_native_grid28.py; launcher launch_native_grid_decode28.py.
Output reports/server28-native-grid-decode-20261030 and sibling log.
Python3.7 compilation passed. Estimate about22minutes from the completed
identical-budget audit; replace with actual launch timestamp and throughput.
Check near completion with180-300seconds spacing.

Launch receipt started17:44:20Z. Verified live17:45:57Z, elapsed1:43,
CPU383%, RSS1.05GB;3observations completed87.35s after initialization.
Expected completion18:06-18:09Z. Inspect near that milestone, not repeatedly
before it. All four training PIDs were also live at17:45:57Z.
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
