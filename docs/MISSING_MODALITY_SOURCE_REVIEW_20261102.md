# Missing-Modality Source Review (2026-11-02)

## Scope

This is a source-level review only. It does not claim a new method, benchmark
gain, or causal explanation, and it does not modify the frozen runtime used by
the active official final-five evaluations.

## Current Runtime Findings

The official evaluator imports `MissingTracker` from
`/home/gaob/Miss-RGBT/runtime/task_utility_20261018/miss_rgbt/tracker.py`.
That tracker carries a binary availability pair for each stored template and
the current search frame. The corresponding `MissingTrackerModel` applies
availability when constructing RGB/TIR tokens: patch embeddings are
multiplied by the availability bit, then positional embeddings are added.

The shared CADTrack transformer blocks then process both modality streams
without receiving the availability mask. At the configured repair layer,
availability is passed to the learned residual repair. The task-utility
branch also operates at layer 6. The tracker maintains target boxes and a
bounded template list; it does not maintain a separate short/long feature
state or a modality-evolution token.

The RGB and TIR branches reuse the same block weights, but each block applies
self-attention and an MLP to the two streams in separate calls; these blocks
do not perform cross-modal attention. Availability gates input patch content
and the layer-6 repair, but it is not passed into the intervening per-stream
blocks. Consequently, a missing stream contains positional embeddings after
the input gate and still undergoes ordinary self-attention/MLP updates before
the availability-aware repair. This is a concrete diagnostic target, not
evidence that it causes the benchmark gap or that changing it improves
tracking.

## Source Identity

Read-only SHA256 values from server28:

| Runtime source | SHA256 |
|---|---|
| `miss_rgbt/model.py` | `5a22607266816f042f52fea152934a197fca19355b83f6be9f0593d2b0836b73` |
| `miss_rgbt/tracker.py` | `1d05274d2827de9a4e08044e5837fedaba971400162a72cb90e98c0d1679efa4` |
| `miss_rgbt/task_utility.py` | `6f9d5c8947b5915eb4d56010caacd72710b69353306e83d0a1221d87e25b67b5` |

These files are outside the published Git checkout under the separately
versioned runtime directory. Do not edit them while any official evaluation
is live.

## Prior-Art Boundary

SCDT, published at CVPR 2026, already formulates modality-missing RGBT
tracking as spatially and temporally conditioned feature denoising. It uses
available-modality spatial cues, short-term history, long-term modality
evolution, and noise-modulated adaptation for both missing and complete
inputs. A direct port or relabeling of temporal feature reconstruction or
noise-conditioned enhancement would therefore not establish a distinct
contribution here. Primary source: the [CVPR paper page](https://openaccess.thecvf.com/content/CVPR2026/html/Lu_Spatio-Temporal_Conditional_Denoising_Transformer_for_Modality-Missing_RGBT_Tracking_CVPR_2026_paper.html)
and [paper PDF](https://openaccess.thecvf.com/content/CVPR2026/papers/Lu_Spatio-Temporal_Conditional_Denoising_Transformer_for_Modality-Missing_RGBT_Tracking_CVPR_2026_paper.pdf).
The public [SCDT GitHub repository](https://github.com/mas-38/SCDenoiseTransformer_CVPR26)
was checked by a shallow clone on 2026-11-02 at commit
`44062ed`. It contains a paper README and a one-byte file named `src`
containing only a newline; no implementation source is present in that
revision, so code-level reproduction is not currently possible.

The public [FlexTrack-V2 repository](https://github.com/supertyd/FlexTrackV2)
was also inspected at commit `30a5ff1`. Its implementation includes bilateral
missing-feature reconstruction with a present-target consistency loss,
heterogeneous mixture-of-experts fusion, and curriculum missing augmentation
with complete-teacher/student distillation. These are direct prior-art
boundaries: porting reconstruction, generic expert fusion, or a missing-rate
curriculum would not by itself provide a distinct contribution. The source
reviewed includes `lib/models/flextrackv2/moe_fusion.py` and
`lib/models/flextrackv2/flextrackv2.py`.

## Next-Experiment Gate

Before allocating GPUs after the current official queue finishes:

- Use training data only to test whether availability-conditioned interaction
  changes the measured failure behavior; do not use val97 or RGBT234 for
  development or parameter selection.
- Pair the proposed mechanism with a parameter/compute-matched control and
  preserve complete-input parity as a first-class outcome.
- Require a mechanism-specific diagnostic to move in the predicted direction
  before training a full candidate. A better aggregate loss alone is not
  sufficient evidence for a paper module.
- Keep the existing task-utility, local-repair, state, crop, and motion
  findings as controls; do not rename previously rejected controls as
  validated modules.
- Evaluate trained terminal candidates only after the registered full
  training protocol and retain the final five epochs through official
  complete/Miss reporting. Disclose post-hoc test-set comparisons.

The three requested paper modules remain unvalidated. Current official
results also remain below the RGBT234-Miss targets.
