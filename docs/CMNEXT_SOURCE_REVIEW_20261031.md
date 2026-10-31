# CMNeXt Source Boundary

Reviewed official InSAI-Lab/DELIVER, CVPR2023 arbitrary-modal segmentation,
at commit `b3a76c33a7b22098b7285d845f22c910abc5cb16`.
Source reference: `https://github.com/InSAI-Lab/DELIVER/tree/b3a76c33a7b22098b7285d845f22c910abc5cb16`.
The root license is Apache2.0. Local git objects were read directly.

| File | Git blob |
|---|---|
| README.md | e63c58f1f401e0ff5432b2f6b5d537c7ed2cc7fe |
| LICENSE | ac4f24556b2e04cf30cc4c6fa294b2b0d77c7bf2 |
| semseg/models/cmnext.py | 2653e7f40ce32b3d11b93ad8cf77fff9b5c91bb1 |
| semseg/models/backbones/cmnext.py | 3f6c7585aa7b1a47814e02a450164ef9b6b1056d |
| semseg/models/modules/ffm.py | ebe09316222c871a2335adb804576a472b5a1013 |

The backbone uses x[0] as camera and x[1:] as auxiliaries. With multiple
auxiliaries, sigmoid scores reweight each feature as score*x+x before an
elementwise torch.max reduction. This is not whole-modality argmax selection.
One auxiliary bypasses that predictor route. FRM adds learned channel and
spatial cross-stream residuals to both streams; FFM combines CrossPath
attention and ChannelEmbed with batch normalization.

The inspected forward exposes no explicit per-token availability-mask
contract or complete-history identity bypass. It is not a drop-in temporal
RGB/TIR template repair. This observation does not claim the project cannot
handle sensor degradation.

Selection and rectification are prior art, distinct from our zero-parameter
copying control. Any adaptation needs explicit temporal/missing-input
contracts and matched training controls. No external implementation,
checkpoint or dependency was integrated or executed; no reproduction,
runtime, tracking benefit or novel-module claim follows from this review.
