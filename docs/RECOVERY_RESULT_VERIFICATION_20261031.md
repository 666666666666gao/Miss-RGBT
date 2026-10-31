# Independent Recovery Result Verification

The four recovery campaigns already select one common checkpoint by the
mean complete/synthetic-outage macro SR across their final five epochs,
then evaluate that weight on official complete/Miss datasets. Their
controller completion alone is insufficient evidence that the full requested
selection, metrics and cleanup succeeded.

`verify_recovery_campaign28.py` is an independent read-only verifier.
It leaves running training/evaluation sources, parameters and weights intact.

For each completed campaign it checks:

- All five epochs are contiguous and agree with the training completion.
- All970 development trajectories (5epochs x 2conditions x 97videos) match
  saved hashes, shapes, sequence/frame counts and GT hashes.
- Per-video PR20/SR and both macro scores reproduce from the saved boxes.
- The selected checkpoint maximizes the registered validation score, with
  earlier-epoch tie breaking, and agrees across selection/completion receipts.
- Exactly one recovery epoch weight remains, its SHA256 matches selection,
  and all four removed recovery paths/hashes agree with evaluated epochs.
- Both official conditions use that same selected checkpoint, each has
  479 videos/337353 frames, and all958 saved trajectories match receipts.
- Official source/GT/mask hashes match registration; the installed official
  scoring package independently reproduces all reported scores and curves.

The verifier reports complete-dataset deltas against the frozen recorded E0:

| Dataset | Baseline PR/MPR | NPR | SR/MSR | Baseline+2pp PR/MPR | Baseline+2pp SR/MSR |
|---|---:|---:|---:|---:|---:|
| LasHeR | 75.636503 | 71.772837 | 60.125714 | 77.636503 | 62.125714 |
| RGBT234 | 90.677134 | N/A | 66.845228 | 92.677134 | 68.845228 |

These+2pp references display the requested approximate gain without inventing
a tolerance. Separate strict missing-modality tests are LasHeR-Miss
PR>69.3/SR>54.4 and RGBT234-Miss MPR>88.1/MSR>64.3.
Passing these score checks does not establish three substantively validated
paper modules, convergence or baseline data-exposure certification.

## Execution Witness

The development helper was actually executed on the completed epoch5 Utility
complete97-video output: PR79.5965784457/SR64.4727766285 matched.
The official helper was actually executed on the historical Utility20
479-video Miss output: LasHeR68.5651743487/63.5765927022/54.6213761663,
RGBT23484.9643532160/60.5112855085 matched all official metrics and curves.
`witness_recovery_verifier28.py` repeats these complete checks and records
source hashes and results. This covers the metric helpers, not end-to-end
selection/cleanup for the still-running recovery campaigns.

`launch_recovery_verification28.py` registers one CPU-only waiting verifier
per recovery campaign. Each checks the actual controller command line while
waiting at300-second intervals, then verifies after official completion.
Original controller disappearance without a completion is an error; there
is no restart or inferred success. Queue source hashes are checked before
verification. Per-campaign output is `independent_verification.json`.

Registered17:34:41 UTC+08. Actual waiting processes128343/128344/128345/
128346 were live17:35:18 for fixed Utility/Random and batch10 Utility/Random.
Queue receipt: `reports/recovery-independent-verification-queue-20261031.json`.
Keep verifier/launcher and registered shared helpers unchanged until all four
independent verifications finish. No recovered campaign has final scores yet.

## Limits And Cleanup

This verifier checks saved trajectories/scoring/provenance; boxes alone
cannot prove inference used the masks at every frame. It checks registered
evaluator source and mask hashes instead of claiming a per-frame trace audit.
It confirms recovery-link cleanup only. Original interrupted-run weights
remain preserved and global cleanup is still pending, including completion
of the CPU epoch6 intervention audit that currently reads those originals.
No weight deletion is performed by this verifier or launcher.
