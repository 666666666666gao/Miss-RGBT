# ViPT Source Boundary

Read the official jiawen-zhu/ViPT repository, whose README identifies the
CVPR2023 Visual Prompt Multi-Modal Tracking implementation, at pinned
commit `f49f30186ebff5587600a61ff224bb341c0a3243`.
Source: `https://github.com/jiawen-zhu/ViPT/tree/f49f30186ebff5587600a61ff224bb341c0a3243`.
Root LICENSE is MIT, copyright2023 Jiawen Zhu.
Files were read from local git objects without executing or integrating them.

| File | Git blob |
|---|---|
| README.md | 5df1a24d6929e3bedbd333f66ca1552a7f510182 |
| LICENSE | 6dacd754829aa8a072b392b71c90fc86b44944e3 |
| lib/models/vipt/vit_prompt.py | e996c164ae71961d90b50fd67ab5496e2cbee09e |
| lib/models/vipt/vit_ce_prompt.py | 6ac6b5ffdcf857aa7ab4534f602dfd12aa2b7467 |
| lib/models/vipt/ostrack_prompt.py | b3a32c9e706b4266bbaa316ba4d389efe2bdbb18 |
| lib/train/base_functions.py | 810e100914fae8d712cca4804f76897c951bcdb8 |

## Actual Mechanism

The inspected non-CE prompt forward splits six-channel template/search
images into RGB and auxiliary three-channel images. RGB uses patch_embed;
the auxiliary uses patch_embed_prompt. Normalized RGB/auxiliary feature
maps are concatenated before a small Prompt_block.

Prompt_block separately projects each half with1x1convolutions to8hidden
channels, applies Fovea to the RGB projection, adds the auxiliary projection,
then projects back to the backbone width. Fovea is per-channel spatial
softmax weighting; smooth=True learns a scalar initialized to10. These
are learned prompt operations, not copying an observed stream unchanged.

The input prompt is added to RGB template and search tokens. The deep mode
updates prompt features and adds residual prompts before later transformer
blocks. This recurrence is across network depth within a forward, not a
video-time cache policy. The wrapper sends backbone search features to the
tracking box head.

For training types containing vipt, the optimizer selects parameter names
containing prompt and sets all other parameters requires_grad=False.
Parameter freezing alone does not establish unchanged outputs or buffers.
Prompt residuals can alter fully complete inputs, and the inspected prompt
forward provides no explicit RGB/TIR availability-state argument or
complete-history identity bypass. The spatial mask arguments are not proof
of a modality-availability contract.

## Relevance To Miss-RGBT

Frozen-main-model auxiliary adaptation and deep residual prompts are prior
art. They cannot supply a novel module merely by replacing LocalRepair.
The useful comparison is whether better auxiliary conditioning changes
closed-loop localization while protecting the desired complete behavior.
Our negative observed-copy result does not disprove trained prompt fusion;
our tiny selected template-coverage loss gain does not validate it either.

An adaptation would need the actual two-template/three-search layout,
explicit unavailable-input handling, mixed template histories, matched
initial weights and optimization, and full trajectory evaluation. ViPT
weights cannot be assumed compatible with the frozen CADTrack-derived
runtime. No external code, checkpoint or dependency has been imported or
executed; no reproduction, accuracy, runtime or new-module claim is made.
