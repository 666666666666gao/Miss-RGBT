# Fresh Response and Geometry Diagnostic

The frozen post-fusion pilot failed its registered promotion gate. Do not
retune or promote that adapter. Its complete-search-after-missing-template
cells worsened, while fully complete history remained equal by construction.

The actual CENTER head decodes size/offset at a discrete score-map argmax.
Regression gradients reach the chosen location but not the discrete
selection operation. This source fact does not establish harmful gradient
conflict or explain tracking failures. Measure those questions first.

Use unchanged E0, new seed39101,128 train882 fitting-partition tuples from
the existing705-name pool, first search only, all nine template/search
availability cells. Initial template always complete. Check no exact
sequence/template-frame/search-frame identity overlaps any previous pilot
cache. Sequence independence from previous fitting is not claimed.
No val97, optimizer, adapter, threshold search or official-test data.

Record native GIoU/L1 regression and focal losses; check their sum against
the real task_loss. Record native IoU, response-peak grid distance and
accuracy, plus per-example cosine between regression and focal gradients
with respect to frozen fusion features. Frozen model weights receive no
gradients and whole-state digests must match.

Decode the same size/offset maps at the ground-truth heatmap peak as an
oracle diagnostic. This uses labels and cannot be deployed at inference;
it is not an attainable tracking gain or an implementation proposal.
Use10000 sequence-bootstrap draws for cell means/effects. Multiple
exploratory cells are not confirmatory tests. There is no training promotion
gate in this descriptive diagnostic. Any ensuing mechanism needs a fresh
predeclared paired experiment with deployable inputs.

Run GPU2 only after occupancy check, existing Python3.7/Torch1.12 runtime.
Expected2-5minutes including nine-condition feature caching; inspect near
completion with180-300second polling if still running. Preserve feature
cache on server; collect small JSON. Original and final-five weights remain.

## Completed evidence

GPU2 PID4143074 completed45.296seconds after launch2026-11-01T23:35:58Z
(2026-11-02 07:35:58 UTC+08).128 tuples sampled111 actual training sequences;
1152 condition rows, all with exactly one target heatmap maximum. Exact
prior-pilot frame-tuple overlap check passed. The same sequences may have
appeared in earlier fitting; this is fresh sampling, not independent data.

Sequence-macro values below are crop diagnostics, not official metrics:

| Dynamic/search | Native IoU | GT-peak oracle IoU | Oracle gain | Peak distance(cells) | Gradient cosine | Conflict fraction |
|---|---:|---:|---:|---:|---:|---:|
| 11/11 | 0.768577 | 0.797548 | 0.028970 | 0.633268 | 0.124472 | 0.318318 |
| 11/01 | 0.561638 | 0.635005 | 0.073367 | 1.799542 | 0.149102 | 0.204204 |
| 11/10 | 0.682737 | 0.712224 | 0.029487 | 0.984070 | 0.164544 | 0.285285 |
| 01/11 | 0.767028 | 0.793167 | 0.026139 | 0.587583 | 0.126064 | 0.312312 |
| 10/11 | 0.766197 | 0.794749 | 0.028552 | 0.638182 | 0.131331 | 0.313814 |

With complete templates, TIR-only native IoU drops0.206940 relative to11/11
(paired95%CI[0.159192,0.257787]); RGB-only drops0.085840
([0.052281,0.123721]). TIR-only oracle gain exceeds complete by0.044397
([0.016508,0.074092]); RGB-only excess0.000517 has CI[-0.023829,0.025803].
Even at the GT peak, TIR-only IoU0.635005 remains below complete0.797548.
An exact peak intervention changes response location and gathered geometry
together; this is not a causal decomposition into disjoint errors.

All nine average gradient cosines are positive. TIR-only conflict fraction
decreases relative to complete by0.114114
(CI[-0.202703,-0.028529]); RGB-only difference CI crosses zero. This does
not support an increased gradient-conflict explanation on these crops.
No PCGrad-style training is justified from this observation.

Missing dynamic templates alone have small native complete-search IoU drops
0.001549/0.002380 with CIs spanning zero. That contrasts descriptively with
the previous adapter's recovery-cell loss increases, but different cohorts,
different loss/IoU endpoints and static crops prohibit a causal attribution
or closed-loop recovery claim.

## Source comparison and next question

Read primary source implementations at pinned GitHub commits:

- CenterNet `xingyizhou/CenterNet`,
  commit`4c50fd3a46bdf63dbf2082c5cbb3458d39579e6c`,
  `src/lib/models/losses.py`:RegL1Loss gathers regression outputs at supplied
  indices and uses a regression mask. It does not train regression solely at
  the current predicted response argmax.
- OSTrack `botaoye/OSTrack`,
  commit`33b5e12586216b7fd0e95d255bd01ba44cbec759`,
  `lib/models/ostrack/ostrack.py`:CENTER forward_head accepts gt_score_map
  and passes it to box_head. This API alone does not prove the training
  caller actually supplies the target map.

Local native CenterPredictor already supports gt_score_map; no dependency
or copied third-party code is necessary. These are ordinary supervision
controls, not novel modules. A next paired train-only experiment can test
whether regression at target-supported locations improves actual predicted
boxes versus matched native predicted-peak regression. It must keep native
inference, record response as well as box errors, protect complete and
recovery states, and use fresh audit tuples. Merely replacing inference
argmax with a GT peak is invalid. No training promotion is granted here.

## Verification and limits

Saved-evidence analyzer independently recomputed all90 cell statistics and
32 paired contrasts, checked1152 rows/128 nine-way identities, task-loss
decomposition, oracle arithmetic and matching-peak equality. Server-file
hash verification passed sources/base checkpoint/cache. Baseline whole-state
digest unchanged and parameter gradients absent. No independent neural
replay or external scientific review. Recorded sources do not include all
transitive imports. Crop augmentations and static template masks cannot
establish official benchmark performance or a temporal recovery mechanism.

Report SHA256:
`7521c438e18a80d4ec2a66881bf6d3907b06839439b148b8b7455314e27310e3`.
Small evidence:reports/server28_response_geometry_20261102/.
Initial protocol archived aspreregistered_protocol.md with SHA256
`4978f87db2f1a9025675e7c294e702cbee83aece067c665be8fb6633dcd86fb2`.
No optimizer, baseline update, val97, official-test run or checkpoint cleanup.
Official scores and full objective remain unchanged and incomplete.
