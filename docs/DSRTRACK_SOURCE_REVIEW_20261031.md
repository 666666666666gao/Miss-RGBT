# DSRTrack Source Review

Inspected repository `https://github.com/qixi-alt/DSRTrack`, pinned commit
`d9e23e08e5b98ede129e4512fc89220d68d307ca`. This is a source-only review:
no dependency, source implementation, weights or thresholds were integrated.
Selected exact Git blobs are retained in local `work/dsrtrack-source-review`.

## Identity And Scope

The pinned README names DSRTrack and describes disturbance-aware online-state
recovery. Its model implements anchor-guided cross-modal template repair.
The earlier assertion that the DCTrack paper links this repository was not
verified and is withdrawn. The earlier description of DCTrack as feature
disentanglement and missing-feature reconstruction was also inaccurate:
the inspected primary PDF describes availability routing, image substitution,
and motion prediction. See `DCTRACK_PRIMARY_REVIEW_20261031.md` for the exact
PDF receipt and bounded findings. Do not use this repository's scores or
mechanism as a verified DCTrack reproduction.

The complete pinned tree contains no filename matching LICENSE, LICENCE or
COPYING. Selected model/evaluation files have Apache-2.0 header notices, but
the new repair and queue files lack those notices. This review establishes
neither repository-wide license coverage nor permission to transplant them.
No code has been copied into our model.

## Actual Repair

`trackit/models/methods/DSRTrack/dsrtrack_full_finetune.py` lines54-94
construct and call `AnchorGuidedCrossModalRepair` before transformer fusion.
It repairs online-template patch embeddings using initial-template anchors.
The prediction head returns boxes/scores, not the repair statistics.

In `modules/fusion/anchor_guided_cross_model_repair.py`:

- Lines147-191 reconstruct each online modality by cross-attending to the
  other modality's online and initial-template tokens.
- Lines198-220 score concatenated current, reconstructed, cross-discrepancy
  and anchor-discrepancy features.
- Lines227-242 form sigmoid gates from opposite detached health minus own
  health minus margin3, then apply foreground-masked residual repair.
- The constructor stores auxiliary weight0.05 at the model call site, but
  this inspected forward returns statistics and does not compute that loss.
  Do not claim the configured value proves an auxiliary training objective.

This is related to our template corruption problem and rules out a broad
claim that anchor-guided cross-modal repair is by itself our novel module.
It does not implement our specific past complete-frame localization-offset
counterfactual in the inspected forward.

## Actual Temporal Management

`vatd_queue_updater.py` lines92-119 selects a template by
`confidence * dynamic_gamma ** queue_age`, where
`dynamic_gamma = 0.98 * exp(-15 * variance(recent_normalized_shifts))`.
Age is queue position, not elapsed video frames. Updates occur every5
calls with confidence>=0.78; configured queue length is2. The last5 shifts
drive volatility. The received anchor/frame-index extra_info is not used.

There is an important coordinate-contract concern before reusing any motion
formula: the evaluation pipeline clips four box coordinates independently
to image width/height via `bbox_clip_to_image_boundary_`, consistent with
corner-coordinate boxes. The updater calculates center as `x1 + x2/2`
and size as `sqrt(x2*y2)`, treating the third/fourth values as width/height.
Thus the inspected pipeline/updater combination does not establish the
intended center displacement normalized by target size. This is a static
source finding, not an executed upstream failure reproduction.

No availability-state argument or past complete-frame state-offset cache
appears in these inspected model, pipeline and updater files. This bounded
absence does not prove novelty across the repository or literature.
Nor does the queue demonstrate recovery when the target leaves the local
search crop. Do not transfer its0.78/15/5 settings to our confidence scale.

## Consequence

Continue the registered native/calibration-sham/corrected experiment without
changing its source or interpreting incomplete subsets. Even a positive
four-video result would first support a small development control, not a
validated paper module. Any larger test needs an independent cohort and
the same compute and complete-trajectory controls.

## Exact Source Receipts

Paths below are relative to the pinned repository.

| File | SHA256 |
|---|---|
| README.md | 29dedab983770e2f878d9c1efcbb763c1a3175059ba24e591a6868c6f0a72e9a |
| config/DSRTrack/run.yaml | 5c128934b04accb61d2849a9a0bf4ab3ce1d27cf6ae124c7fa16f43f0a49275e |
| config/DSRTrack/dinov2/config.yaml | f641f8a97b3c145d52fc23f0d88d39d33056d36635d699aca34481bbd134ff0d |
| trackit/models/methods/DSRTrack/dsrtrack_full_finetune.py | ed2f6b5d5d4a30079162e4a44d5a6d4d3675b156fcd34d25792cda0b335ee890 |
| trackit/models/methods/DSRTrack/modules/fusion/anchor_guided_cross_model_repair.py | 653f8946ded5672f33ee6becab91b762a11ea85187fae678a903dad0df5ee6b4 |
| trackit/runner/evaluation/distributed/tracker_evaluator/default/pipelines/one_stream/__init__.py | 720d2ca608a725ee81f2beca5a1ca68a7fe6a8e946b605b5aaee029d026ea972 |
| trackit/runner/evaluation/distributed/tracker_evaluator/components/template_updater/vatd_queue_updater.py | a368a9900464652908bc02209bfebd15334fac17c688d53e709bc4ab47b28e79 |
