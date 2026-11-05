# SAMURAI Motion And Memory Source Review

Primary repository: `yangchris11/samurai`.
Pinned HEAD: `76ba195984892b0d1e3db5d9c9f90bb62175680a`, resolved
with git ls-remote. GitHub API requests hit the unauthenticated rate limit;
no credentials were read. Source was downloaded from immutable raw URLs.
No external source, weights, dependency or threshold was integrated.

Inspected:

- sam2/sam2/modeling/sam2_base.py
- sam2/sam2/utils/kalman_filter.py
- root LICENSE, Apache2.0 text

Root licensing is recorded as provenance, not a blanket clearance of every
file/dependency. Copies are kept in local work/; the large external source
tree is not copied into the publication repository.

## Actual Mechanism

The filter state is x,y,aspect,height and their four velocities, with
constant-velocity transition dt1. Position/velocity process noise scales
with predicted height. This is ordinary Kalman filtering, not a new
missing-modality method.

In sam2_base.py the SAMURAI path operates on multimask outputs:

- Lines420-437 initialize motion from the mask with highest predicted IoU.
- Lines438-458 predict and conditionally update during warm-up.
- Lines459-498 compare candidate mask-box IoU with the predicted filter
  mean and combine that with predicted mask IoU for candidate selection.
  Low predicted mask quality resets the stable-frame counter; otherwise
  the chosen box updates the motion state.

Pinned defaults include15 stable frames, predicted IoU threshold0.3,
and motion-score weight0.15. They belong to SAM2 mask quality and this
implementation, not our Hann-window response or calibrated confidence.
They are not adopted.

Lines663-689 select historical memory using predicted mask IoU, object
score and, when available, motion consistency. The immediately previous
frame is appended even when absent from the filtered index set.
Thus this code does not enforce that every memory frame passes every
filter. It is incorrect to describe it as strict trusted-memory-only use.

The motion consistency score uses the predicted mean's box overlap with
already generated candidates. The inspected selection code does not
convert covariance into a search crop that restores missing target support.
It therefore does not demonstrate a solution to our observed recovery
frame384 lying outside the local search crop.

## Consequence For Our Evidence

Our fixed-CV audit found both inaccurate observed velocities and failure
of long-gap extrapolation. A filter that separates prediction from
measurement update is a useful ordinary control. If used later, it must
advance in video time through blind frames. The native tracker frame_id
counts neural calls, and blind frames currently do not increment it.
Using that counter as the filter's elapsed time would change the intended
motion model. Native template-update timing is a separate contract.

The existing tracker writes dynamic templates based on Hann-window
confidence and a model-call interval. These are not SAM2 object/mask
quality scores. The source review does not justify importing0.3/0.15/15
or treating motion agreement as independent target identity evidence.

Any future memory-selection experiment must keep motion prediction,
recovery crop placement and template writes separately measurable, with
ordinary Kalman/hold/CV and native/frozen-template controls where relevant.
The existing availability-history closed-loop comparison remains the next
decision point; this review does not trigger another GPU job or modify it.
Ordinary filtering, score thresholds and memory selection cannot simply
be named as three validated paper modules.

## Source Integrity

| Local pinned copy | SHA256 |
|---|---|
| work/samurai-sam2-base-76ba195.py | 5f86907ad50418249722b5ea9a6f1ce95e80b293de24c3f9f0ca99ffeae9963a |
| work/samurai-kalman-76ba195.py | 301121f50b943d0acd61570559a03dcfd0d1ce5b05924f28b3b33d3de5879f30 |
| work/samurai-LICENSE-76ba195.txt | c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4 |

At13:32:10 UTC+08 CPU869785 had completed its second arm's192 updates
and remained live doing final measurement/selection. It had no final
paired completion or independent analysis yet. Motion controller1044705,
workers1044714/1044715 and comparison1049221 remained live.
Full official targets and three validated contributions remain unfinished.
