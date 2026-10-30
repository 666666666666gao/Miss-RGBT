# Independent initial-anchor evidence audit

Date: 2026-10-30. Historical A utility epoch20, CPU4threads, SSH2028.
The frozen tracker/runtime and all four live training jobs were preserved.

## Problem and Experiment

Existing layer6 template/search context is jointly contextualized and cannot
serve as independent first-frame identity evidence. This diagnostic instead
encodes initial and candidate template-sized crops separately through the
frozen patch embedding, template positional tokens, CLS and first6 blocks.
Foreground-minus-background vectors are normalized, compared by cosine per
modality and averaged over available modalities. Candidate streams do not
interact with each other or with the stored first-frame descriptor.
This is independent encoding with a tracking-trained backbone, not an
independently trained identity verifier.

Fixed first8 val97 sequences were used. The first64-frame-stride run compared
predicted and GT-centered candidates against own and cyclic next-sequence
anchors. It produced136 observations in15.94s. Own-anchor cosine exceeded
the wrong anchor on95.6% of predicted candidates, including92.3% of candidates
with IoU<0.2. Cross-sequence negatives therefore fail as sufficient evidence
of localization or original-target acceptance. Low own-anchor similarity AUC
for IoU<0.2 was0.6875 pooled, weaker than Hann0.7626 on these sampled frames.

## Same-Frame Background Control

The next control adds four same-size boxes offset by2*sqrt(GT area) along
each axis, clipped to image bounds and excluding any positive-area GT
overlap. GT-centered crops are oracle positives; these are diagnostic
candidates, not proposals available to an inference system.

The64-frame stride aligns with the32-frame outage schedule and samples only
11 restoration blocks. Its136 observations/504 background candidates show
oracle-positive cosine greater than every background on76.5% of observations;
the long `leftdrillmasterstanding` reaches only55.9%. These duplicated
complete/outage restoration measurements are not independent samples or
evidence about single-modality observations.

This sampling alias was explicitly corrected with a predetermined37-frame
stride. Both64 and37 results remain saved; no score threshold was tuned.
The37-stride audit completed in44.99s with222 observed condition/frame rows
and823 background candidates. Complete/outage rows share video frames, so
the pooled count must not be treated as222 independent examples.

| Condition/state | Observed rows | Oracle target above all backgrounds |
|---|---:|---:|
| Complete | 120 | 80.83% |
| Alternating outage, all observed states | 102 | 78.43% |
| Alternating outage,11 | 57 | 84.21% |
| Alternating outage,01 (TIR only) | 22 | 68.18% |
| Alternating outage,10 (RGB only) | 23 | 73.91% |

On `leftdrillmasterstanding`, the37-stride paired aggregate is62.39%.
Even oracle-centered target crops lose to at least one same-frame background
in a substantial subset. Therefore this cosine alone is not a justified
recovery acceptance rule. Zero GT overlap does not establish that a
background box is a semantically similar distractor; same-scene identity
selectivity and rejection still require stronger evidence.

## Next Executed Step

A separate CPU frozen-descriptor ranking pilot completed as PID3858597 in41.65s.
It uses the first32 train882 sequences:24 train and8 development, disjoint
from the heldout first8 val97 entries. Three fixed temporal quartiles per
sequence provide GT-positive/fixed-offset background groups. A small shared
per-modality1538->128->1 MLP learns ranking with candidate-group cross entropy;
11/01/10 scores average only available modalities. Backbone frozen.
The selected epoch minimizes development loss among final5, with a100epoch
ceiling and minimum30/patience10 proxy stopping. Heldout val97 never selects
epoch or stopping. This is a small ranking pilot, not an implemented tracker
module, novelty result, calibrated acceptance gate or benchmark gain.

The pilot used71 train,24 development and22 heldout candidate groups.
It stopped at30 and selected26 from final26..30. Its heldout top1 versus
plain cosine was18/22 versus18/22 (11),16/22 versus17/22 (01),17/22
versus16/22 (10). There is no consistent benefit in this small cohort.
The selected head remains; the other4 final-window checkpoints were deleted
with hash receipts. This small head's30-epoch proxy plateau does not establish
convergence of any full tracker training. The pilot is not promoted into the
formal tracker. Its checkpoint SHA256 is
11110ac9c04de0d234e1e23fef51aa087e57146a28d7d769dbdfe7453e117169.

## Evidence

- reports/server28-independent-anchor-20261030
- reports/server28-independent-anchor-hard-20261030
- reports/server28-independent-anchor-hard-20261030-stride37
- reports/server28-anchor-ranker-pilot-20261030 (completed pilot)
- Local analysis JSONs with matching names under outputs/

Launch/report files record source/checkpoint/protocol/GT/trajectory hashes,
sampling and exact candidates. No test-set scores or masks chose the sampling,
head, epoch or thresholds. E0's historical val97 exposure remains uncertified.
