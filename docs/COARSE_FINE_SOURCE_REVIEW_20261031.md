# Coarse To Fine Recovery Source Review

## Pinned Primary Source

Read the official `zju3dv/EfficientLoFTR` repository at commit
`07e9c1401e71e9c556b1fda9d86d060af36ba176`, dated
2026-09-15T07:34:43Z. The README identifies the work as CVPR2024
Efficient LoFTR. This is a source inspection, not a reproduction or tracker
experiment. Eight selected text/API files are saved outside the publication
checkout under `work/source-research-efficientloftr-20261031`.

The pinned license is Project Registration License v1.0, not MIT/Apache.
It explicitly describes research/education/personal use and organizational
project registration separately. No external implementation, weights or
dependency were integrated into Miss-RGBT; no registration was submitted.
The repository README specifies a separate environment including Torch2.0.
This review does not establish compatibility with the running environment.

## What The Code Actually Supplies

`src/loftr/utils/full_config.py` sets coarse/fine resolution to(8,1).
`src/loftr/loftr.py` retains half/quarter-resolution backbone features
alongside coarse features. `fine_preprocess.py:42-56` fuses those feature
levels and upsamples before extracting windows for predicted coarse
matches. Fine detail does not come from interpolating the coarse map alone.

`coarse_matching.py:119-127,152-169` optionally applies dual softmax, then
confidence thresholding and mutual-nearest matching. `fine_matching.py`
refines correspondences within local windows and outputs matched point
coordinates. It does not predict an object's width/height or establish
that a point belongs to the tracked identity. A correspondence score
therefore cannot replace our target-specific box acceptance evaluation.

Pinned code:

- https://github.com/zju3dv/EfficientLoFTR/blob/07e9c1401e71e9c556b1fda9d86d060af36ba176/src/loftr/loftr.py
- https://github.com/zju3dv/EfficientLoFTR/blob/07e9c1401e71e9c556b1fda9d86d060af36ba176/src/loftr/loftr_module/fine_preprocess.py
- https://github.com/zju3dv/EfficientLoFTR/blob/07e9c1401e71e9c556b1fda9d86d060af36ba176/src/loftr/utils/coarse_matching.py
- https://github.com/zju3dv/EfficientLoFTR/blob/07e9c1401e71e9c556b1fda9d86d060af36ba176/src/loftr/utils/fine_matching.py
- https://github.com/zju3dv/EfficientLoFTR/blob/07e9c1401e71e9c556b1fda9d86d060af36ba176/LICENSE

## Consequence For Miss-RGBT

Our completed oracle-crop diagnostic found successful boxes at6/12
unsupported observations with GT center/previous size, and10/12 with GT
center/true size. Coarse tile-size GT-centered crops succeeded at0/12.
The native51-crop audit restored center coverage at9/12, but only1/12 had
any successful candidate. This is evidence that candidate localization and
scale remain a bottleneck even before identity ranking; it is not proof
that a particular new matcher solves the problem.

The bounded next recovery candidate should separate:

1. Coarse target-conditioned region proposals using available observations
   and initial target information, without current GT at inference.
2. Fine image crops at an appropriate scale, followed by the existing
   tracker backbone/head. Cropping the original image retains information
   that a coarse feature-map resize cannot recover. The existing failed
   grid remains a mandatory matched control.
3. Target acceptance, compared against native confidence and initial-target
   matching with the same candidate budget. Measure both erroneous
   acceptance and recovery; proposal coverage alone is insufficient.

This is a prospective test, not three validated paper contributions.
Use train882 training/development partitions for fitting any proposal,
scale or acceptance parameters. Do not select budgets/thresholds from the
previous val97 audit or official test failures. Keep the existing backbone
and insertion boundaries; the external RepVGG matcher is not transplanted.
Full-val97 E0/epoch5 tracking now running will supply broader trajectory
evidence before allocating the next GPU training campaign.

## File Integrity

| Pinned text file | SHA256 |
|---|---|
| LICENSE | 8efca11dbcac37a13b1869cbe69ff986805ed87c4cb8881f8b732d698e3e1cc5 |
| README.md | 6a45d5d15e36244e7bb3058d41181e3dc170d15d2ac3f16a704e0073cb6f560e |
| src/loftr/loftr.py | 35d7e3b422bc21250525e104d609db5f206a80e961c52915a70be921c7024316 |
| src/loftr/loftr_module/fine_preprocess.py | 9893f2e39ef3f67386a779ff2a7eba4663582ed2d0c20bb6ec1127e7bc6da722 |
| src/loftr/utils/coarse_matching.py | e0bbe86d9883ed208c55a81611696c8f89f1464ae46a7872e236b424cbce8064 |
| src/loftr/utils/fine_matching.py | 6ce4bc3bce790201387b60dc61bbb37e2ca87809a764594fcb37853570c274c1 |
| src/loftr/utils/full_config.py | 2e4867e9d86ab77250bb7dfd9fc817c079c9add1e123b181ef0934c3c27aced2 |

Existing experiment references: `ORACLE_CROP_AUDIT_20261030.md`,
`NATIVE_GRID_AUDIT_20261030.md`, `INDEPENDENT_ANCHOR_AUDIT_20261030.md`.
