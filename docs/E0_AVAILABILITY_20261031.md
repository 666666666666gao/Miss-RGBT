# Full Validation E0 Availability Groups

Offline analysis of the completed E0 paired val97 tracks. All194 trajectory
hashes and GT hashes are checked again. Both arms score exactly the same
valid video frames within each fixed alternating-outage state. Exclude
initialization/nonpositive-size GT:54289 scored frames total.

| Actual state RGB/TIR | Sequences with state | Scored frames | PR20 gap, pp | SR gap, pp |
|---|---:|---:|---:|---:|
| 00 hold | 62 | 7503 | -35.038245 | -26.475055 |
| 01 TIR only | 97 | 10086 | -16.922868 | -15.765364 |
| 10 RGB only | 84 | 8865 | -10.941712 | -8.069995 |
| 11 both observed | 97 | 27835 | -4.098580 | -2.937169 |

Gaps are outage-minus-complete sequence macros over sequences containing
that state. State rows have different sequence/frame membership and cannot
be added or directly interpreted as sensor-specific causal effects.
The JSON also reports frame-weighted values and every sequence-state row.

In particular, current complete-input frames in the outage trajectory
still show a gap. A current-frame-only missing-input check misses this
descriptive history effect. Position, scale, template history, difficulty
and outage timing are confounded; this does not prove harmful template
writes or justify changing template-update rules. The00 hold gap also
does not establish the benefit of any motion predictor.

The next recovery candidate must evaluate actual mixed histories and
restoration frames with matched full trajectories. Fit parameters on
train882 development, not these val97 groups or official test results.
This is a diagnostic, not a candidate benefit or a new paper module.

Runner: `analyze_e0_availability28.py`, CPU offline only.
Report: `reports/e0-availability-analysis-20261031.json`, SHA256
`7aee95037fae383560eb70ccd7656bb247c5925f525c40f2132fe15f062a6470`.
No tracking or training was rerun.
