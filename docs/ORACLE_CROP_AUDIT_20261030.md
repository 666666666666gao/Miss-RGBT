# Oracle Crop Diagnostic

This diagnostic uses current ground truth to construct crops. It must never
be reported as a deployable recovery method or benchmark improvement.

## Completed Results

Worker3991543 completed normally in73.88s,45 valid points/135 new crops.
Raw report, source/checkpoint/input hashes and analysis were copied locally
under outputs/server28-oracle-crop-20261030.

| Diagnostic | Mean IoU, all45 | IoU>=0.5, all45 | Mean IoU,12 unsupported | IoU>=0.5,12 unsupported |
|---|---:|---:|---:|---:|
| Native crop | 0.56452 | 32 | 0.00000 | 0 |
| Existing global candidates, Hann oracle | 0.60123 | 32 | 0.09291 | 0 |
| Existing global candidates, raw oracle | 0.61069 | 32 | 0.12255 | 0 |
| GT center, previous size | 0.70746 | 38 | 0.50256 | 6 |
| GT center, true size | 0.74790 | 42 | 0.65440 | 10 |
| GT center, coarse tile size | 0.54683 | 29 | 0.07684 | 0 |

Even the existing raw-response oracle cannot recover IoU>=0.5 at any of
the12 unsupported points. Removing Hann alone does not solve the earlier
finite proposal-set limitation.

The unsupported12 points come from three sequences. At all six
`leftdrillmasterstanding` points, GT-centered true-size crops exceed0.5;
previous-size crops do so at five. `ab_pingpongball` remains difficult:
only2/5 valid sampled points reach0.5 even with true-center/size crops.
These are correlated observations and not independent trials.

For the12 unsupported points, median target short-side size at the resized
search input is30.14pixels with previous-size crops,43.98pixels with
true-size crops and10.67pixels with coarse tiles. This is consistent with
scale/resolution being a bottleneck, but changing crop size also changes
context and Hann weighting; the experiment does not isolate resolution
as the sole cause, nor prove target identity is solved.

This supports a next diagnostic of native-resolution proposals across
larger spatial support with matched proposal budget and separate acceptance.
It does not justify deploying a GT-guided method, picking recovery
thresholds on these val observations, or claiming any PR/SR improvement.
All four GPU training workers continue; no formal runtime changes.

## Fixed Protocol

Reuse all45 valid-GT observations from the completed global proposal audit,
first8 val97 sequences. Preserve checkpoint, initial templates, missing-view
availability and native Hann/regression. Evaluate three centered crops:

1. GT center with previous predicted box width/height.
2. GT center with true current box width/height.
3. GT center with the same search side as the earlier half-image tiles.

No learned parameters, trigger thresholds or additional frame selection.
The first comparison removes center error; the second also removes size
error; the third examines centered coarse resolution. Centering changes
the native Hann prior and context, so differences are not a clean
single-factor proof of target identity.

Also compute GT-IoU oracle over existing raw-response candidate boxes.
That costs no new predictions and complements the earlier Hann oracle.
An oracle maximizes among a finite candidate set, not all possible crops.

## Execution

Server28 CPU worker3991543, four threads, CUDA disabled. Script
audit_oracle_crop28.py; log reports/server28-oracle-crop-20261030.log.
Verified live at17:04:10Z, six observations completed, no terminal error.
Estimated two minutes for45 points/135 new crops; do not restart a live
worker based on estimated duration. Four GPU trainings remain unchanged.

Expected evidence:

- reports/server28-oracle-crop-20261030/launch.json
- reports/server28-oracle-crop-20261030/report.json
- reports/server28-oracle-crop-20261030/analysis.json
- reports/server28-oracle-crop-20261030/completed.json

The execution paragraph records launch history; completed results above
supersede its pending state.
