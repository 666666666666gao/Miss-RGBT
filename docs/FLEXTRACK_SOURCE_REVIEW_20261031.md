# FlexTrackV2 Source Review

Pinned repository: `supertyd/FlexTrackV2`, commit
`30a5ff1b39b8f3004dde4018018b901e6cc1bf54`, API commit date
2026-07-20T06:33:18Z. This is source inspection, not reproduction or
verification of the repository's reported benchmark scores.

## Relevant Implementation

`lib/models/flextrackv2/flextrackv2.py:301-308` encodes both modalities
before building the complete teacher. The training student clones those
encoded tensors, then replaces search/template feature positions using
availability masks and prompts (`318-381`). Thus this implementation's
feature masking must not be assumed equivalent to our input masking
before every transformer block: retained contextual features can differ.
This distinction motivates our separate repair-scope intervention.

`lib/models/flextrackv2/moe_fusion.py:481-556` maps each modality to the
other, substitutes features based on a per-sample two-channel mask, adds
healthy-target reconstruction loss, and routes reconstructed features
through experts. The fusion mask is broadcast across token positions.
It does not directly supply our token-level dynamic-template availability
contract or the frozen complete-history output invariant.

The checked configuration uses five templates, two search frames,
112/224 crops, a Fast-iTPN encoder and BMR_HMoE. This is not a drop-in
replacement for our two-template CADTrack model.

## Decision

Keep the useful question: does missing-position-only correction leave
contextual damage unaddressed? Measure it using our own unchanged E0
and input-level masks before adding capacity or losses.
Ordinary cross-modal hallucination, MoE and feature reconstruction are
existing controls, not three new contributions.

No license file appeared in the pinned recursive tree search for
LICENSE/COPYING/NOTICE. No external code, weights or dependencies were
integrated. Copies read for inspection remain outside the publication
checkout under `work/flextrackv2-*-30a5ff1.*`.

## Pinned Sources

Base source URL:
`https://github.com/supertyd/FlexTrackV2/tree/30a5ff1b39b8f3004dde4018018b901e6cc1bf54`

| Relative path | Downloaded SHA256 |
|---|---|
| lib/models/flextrackv2/flextrackv2.py | a421d5887282a7c1424b5d2b22dbd2a86fd69502bc3391ed4886335d9f335338 |
| lib/models/flextrackv2/moe_fusion.py | 8c16213f24bb8524774210f04e5d243a561d23b5b05af504248e31efd2b1309f |
| experiments/flextrackv2/flextrackv2.yaml | 6101a6b94614dd30bfcbd3a4fe1e860987c73da260cf98d54f4b3b23a6756964 |
| lib/models/flextrackv2/encoder.py | a95d1f4ac25ed41720696477ff1c01bb0e84ce8b04a3bda6fb7439017d573091 |
