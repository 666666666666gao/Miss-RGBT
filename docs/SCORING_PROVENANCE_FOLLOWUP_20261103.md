# Scoring Provenance Follow-up - 2026-11-03

## New Public-Source Check

The paper's arXiv HTML code link resolves to the public
`Multi-Modality-Tracking/SCDT-CVPR2026` GitHub repository. That repository
currently renders the explicit message "This repository is empty." Its open
issue #1, created on 2026-07-21, asks when the code will be released and has
no visible maintainer response. A direct `git ls-remote` check returned no
refs for this repository. These observations establish that this audit could
not obtain versioned SCDT source from the paper-linked public repository on
this check; they do not establish that no private, separately hosted,
unindexed, or subsequently released evaluator exists.

The CVPR paper remains the source for the published SCDT scores and metric
names, but the public repository state does not provide an implementation
against which to prove exact scoring parity. The dataset/benchmark-code
repository is a separate source: its pinned resource page links evaluator
archives, but that is not evidence that SCDT used a particular archive or
that the archive matches SCDT's executed scorer.

Public references:

- Paper-linked SCDT repository: https://github.com/Multi-Modality-Tracking/SCDT-CVPR2026
- SCDT code-release issue: https://github.com/Multi-Modality-Tracking/SCDT-CVPR2026/issues/1
- CVPR 2026 paper: https://openaccess.thecvf.com/content/CVPR2026/html/Lu_Spatio-Temporal_Conditional_Denoising_Transformer_for_Modality-Missing_RGBT_Tracking_CVPR_2026_paper.html
- Author-deposited arXiv HTML and code link: https://arxiv.org/html/2607.24701
- Dataset and benchmark resource repository: https://github.com/mmic-lcl/Datasets-and-benchmark-code

## Consequence for This Project

The existing frozen evaluator and its independent verifier support
reproducibility under the archived `rgbt==1.0.1` implementation. They do not
establish equivalence to the SCDT scorer or to the dataset maintainers'
downloaded evaluator archives. Keep SCDT table values labeled as published
comparison targets, not as a controlled head-to-head result. Do not alter
historical scores or select a checkpoint based on this source gap.

If an SCDT evaluator becomes publicly available, the follow-up should pin its
revision and environment, then compare sequence lists, GT parsing, invalid
frame handling, box rounding/clipping, threshold grids, aggregation, and
Complete/Miss masking on identical frozen predictions before making any
cross-paper equivalence claim.
