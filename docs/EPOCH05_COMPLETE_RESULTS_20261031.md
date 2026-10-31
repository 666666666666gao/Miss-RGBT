# Epoch Five Complete-Input Tracking Results

Both batch10 epoch5 checkpoints completed native full-val97 tracking.
This is a fixed epoch5 health observation, not final-five checkpoint
selection or an official benchmark result.

## Verified Full Results

Every model covers97 sequences/54646 video frames/54289 scored frames.
All291 trajectory SHA256 values, GT hashes/shapes, per-sequence PR20/SR,
frame counts and aggregate metrics were recomputed and matched within1e-9.
Protocol, masks and inference source identities match across the three
models. Both candidates have10000 optimizer updates at effective batch10.

| Model | PR20 | SR | PR20 minus E0, pp | SR minus E0, pp |
|---|---:|---:|---:|---:|
| E0 | 75.879164 | 61.654017 | 0 | 0 |
| Utility epoch5 | 79.596578 | 64.472777 | +3.717415 | +2.818759 |
| Random epoch5 | 80.667388 | 65.321675 | +4.788224 | +3.667658 |

Paired-video percentile bootstrap95% intervals use10000 shared resampling
draws, seed2027:

| Contrast | PR20 gap interval, pp | SR gap interval, pp |
|---|---|---|
| Utility minus E0 | [0.109699,7.820159] | [0.047253,5.852760] |
| Random minus E0 | [1.029037,9.135709] | [0.748196,7.004693] |
| Utility minus random | [-3.770910,1.172835] | [-3.080249,0.994642] |

Utility-minus-random point estimates are-1.070809PR20/-0.848899SR.
SR sequence wins/ties/losses are52/1/44 for utility versus E0,51/1/45
for random versus E0, and46/1/50 for utility versus random.

The complete-input development improvement exceeds approximately2points,
but utility selection has not established an advantage over matched random
selection. Intervals across videos do not estimate training-seed uncertainty,
and historical E0 exposure to val97 remains uncertified. This result does
not prove LasHeR/RGBT234 official complete-input targets or any Miss target.

## Heterogeneity And Decision

Both models regress substantially on `redgirl1497`, `rightmirrorlikesky`
and `righttallholdball`. Utility SR gaps versus E0 there are
-45.343/-32.053/-21.860pp. Utility also trails random substantially on
`AQblkgirlbike` and `belowrightwhiteboy`, while outperforming it on
`rightboywithwhite` and `midboy`. Do not tune parameters on these val97
examples or interpret a few large gains as a generally improved selector.
Full per-contrast best/worst lists and bootstrap intervals are in the JSON.

Continue original training and the existing convergence rule; epoch5 alone
does not establish convergence. Retain final-five common-weight selection
after training. Do not delete candidate snapshots or promote random epoch5
as the final model based on this intermediate complete-only observation.
Wait for the same checkpoint's full alternating-outage comparison before
allocating the next method experiment. A paper contribution must demonstrate
benefit over random selection in matched final comparisons and mechanism
controls; overall E0 improvement alone is insufficient.

## Execution And Evidence

Utility complete took3529.75s and random3895.94s, compared with4391.13s
for E0 under concurrent training. These wall times are not controlled
speed benchmarks. Both original workers moved into alternating outage.
At10:32:58+08:00 all four trainers, controller570241, workers570254/570255
and postprocess613855 remained live. Utility outage had its first sequence
receipt; no full candidate outage result yet.

- `analyze_complete_health28.py` reuses the existing verifier's metric and
  paired-comparison functions without modifying the queued final analyzer.
- `reports/epoch05-complete-paired-analysis-20261031.json`, SHA256
  `b9756938ea9da97fbfab8cb0a2211f2b99cd8f6dfc1cd226c9f92a4c7a414fce`.
- `reports/epoch05-utility-complete-20261031.json`.
- `reports/epoch05-random-complete-20261031.json`.
- Same checkpoint hashes as the original epoch5 launch records; no training,
  LR, stopping, final-five selection or frozen-runtime modification.

Provisional full outage completion window11:15-11:40+08:00; refine from
actual sequence throughput. Final paired analysis remains the existing
six-cell postprocessing job and is not replaced by this stage report.
