# Calibration Prior-Art Boundary

This bounded review accompanies the unchanged center/scale development
controls. It supplies neither a new module nor evidence of tracking gains.
No external code was copied into the project runtime or executed.

## Sensor Alignment Is Not Missing-State Prediction Correction

The author repository `NOP1224/Unaligned_RGBT_Tracking` was pinned at
`b0935e2e1ab588478268fb1a79915a7bf3c224d8`.
Five raw source files were read from its `PMATrack` subtree.
`alignnet.py` contains target/structure/detail experts with shift/scale
outputs and routing. Its `NeuroNet.infer` defaults to a forced expert;
whether that path is used in the intended tracker was not established.
The downloaded tracker contains a projective image-warp function and a
translation/scale homography constructor.

These mechanisms operate on RGB/TIR alignment representations. Our current
control caches differences between full and single-modality predictions on
past complete frames, then corrects later predicted boxes. Generic
translation/scale/routing mechanisms cannot be claimed as new merely by
changing this application. Conversely, a successful sensor-registration
method does not establish recovery from absent observations or stale caches.

There is a concrete static integration uncertainty: the downloaded
`PMATrack/lib/models/ostrack/ostrack.py` imports
`lib.models.sfcatrack.vit` and `lib.models.sfcatrack.vit_ce`, but the pinned
tree has no `PMATrack/lib/models/sfcatrack/` files. Other project subtrees
exist. The required package composition/entrypoint was not executed, so
this is not a reproduced runtime failure or a verified runnable transplant.
No LICENSE/COPYING path was found in the pinned recursive tree; no reuse
permission is inferred from repository visibility.

Local read-only reference copies are in workspace `work/pma-source-review`.
They are excluded from this publication; the following raw-source hashes
identify what was read.

| Source Within PMATrack | SHA256 |
|---|---|
| lib/models/layers/alignnet.py | b8607bef8e68fc72b445aaef75a5ba030a48e1cc8ffd31f68c726d5b4cc83490 |
| lib/models/ostrack/ostrack.py | 619759aba43b9ea73b36e07dba6253347e4fadf1127b9e7923fbe0439b28edb8 |
| lib/test/tracker/ostrack.py | cc4975207f9e27794bbbee7f02997d02aac3e07e474bb4eea0d1fd2be56b20cf |
| lib/models/ostrack/vit_ce.py | 484e1b6bf3dba5647ccbe7b318eedcbc3b4659f00cbf8d0b50869859b8cf9b3f |
| lib/models/ostrack/vit.py | 9c5ba5473c0e283ec24cb3d7ae430320ffa222db327d99e56dea3e90b345faf9 |

## Other Primary Records

The PAFCNet author abstract, arXiv2608.01807, describes adaptive fusion,
feature calibration and a missing-modality adapter. Only its abstract was
read here; no implementation, experimental reproduction or precise
architectural equivalence to our method was verified. It is a reason to
avoid a broad novelty claim around calibration/adaptation alone.

`supertyd/FlexTrack` at `a3f0e7811772a0082d035cc9c29251010706dbc3`
contains README and LICENSE only and redirects to `supertyd/FlexTrackV2`.
The latter already has a separate pinned review in
`docs/FLEXTRACK_SOURCE_REVIEW_20261031.md`. The wrapper repository is not
an additional independently reviewed implementation.

Primary records:

- `https://github.com/NOP1224/Unaligned_RGBT_Tracking/tree/b0935e2e1ab588478268fb1a79915a7bf3c224d8`
- `https://arxiv.org/abs/2608.01807`
- `https://github.com/supertyd/FlexTrack/tree/a3f0e7811772a0082d035cc9c29251010706dbc3`

## Decision

Keep current registered calibration controls unchanged until their complete
reports are independently checked. A useful paper module must resolve an
observed failure with deployable information, exceed ordinary controls and
replicate on prospectively fixed data. This source review does not satisfy
those requirements or justify adding a gate, decay coefficient or expert.
