# Saved-Trajectory Offset Mechanism Audit

Input: completed four-video control,24 independently verified trajectories.
This audit uses no additional model calls or GPU allocation. It reads only
finished native/corrected outage trajectories and evaluation GT. It does
not alter the running independent16-video replication.

Report `reports/state-offset-mechanism4-verified-20261031.json`, SHA256
`44deca28fb1b641c06f930ff798e40f320e8dfd5528b81be6e8cfe461c95ea9c`.
Downloaded bytes match the server hash. Input report, worker-report,
trajectory and GT hashes are checked. Recomputed full per-video PR/SR
match the original receipts within1e-10, including the original scored-frame
counts. The report includes this audit's source hash.

## Immediate Same-History Effect

For each single-modality frame in the corrected trajectory, compare its
saved clipped no-offset prediction with its clipped corrected prediction.
Both arise from the same raw prediction and corrected tracker's history.
This holds crop/template history fixed for that frame, but it does not
evaluate an independently evolving no-offset tracker.

| State | Center error before/after, px | Error delta95% CI | PR delta, pp | SR delta, pp |
|---|---:|---|---:|---:|
| 01, TIR-only | 50.781617 / 48.854224 | -1.927394 [-3.689640,-.383854] | +1.332849 | +.493538 |
| 10, RGB-only | 47.453297 / 47.236450 | -.216847 [-.443589,.009896] | -1.086957 | -.182500 |

Each value is a mean across the four video's own frame means, not a
frame-pooled estimate. TIR-only center error improves3 videos and worsens1.
Its PR/SR intervals[-1.086957,3.947368]/[-.396903,1.102929] cross zero.
RGB-only PR interval[-3.260870,0] includes zero, SR interval
[-.543478,.096525] crosses zero. Center error is not interchangeable with
overlap or thresholded precision. No stable RGB-only benefit is established.

`leftredflag-lsz` has TIR-only error176.617052px before versus172.084947px
after the immediate correction. A smaller error can still leave severe
tracking failure; this audit does not demonstrate target recovery.
`bikefromnorth` improves total closed-loop SR4.396941pp despite immediate
TIR-only SR-.828157 and RGB-only SR-.724638pp. Thus the trajectory-level
gain cannot be described as a uniform immediate correction benefit.

## Closed-Loop State Slices

Compare native and corrected trajectories at each availability state in
the alternating-outage condition. These slices include all accumulated
crop, localization and template changes.

| State | SR delta, pp | Video95% interval |
|---|---:|---|
| 11 | +.567341 | [-1.513169,3.309379] |
| 01 | +1.669953 | [-.318341,4.510905] |
| 10 | -.932571 | [-3.340841,.677191] |
| 00 | +1.456278 | [-.297619,4.677177] |

Complete-observation frames inside an outage trajectory have different
histories, so equality is not expected there. This does not contradict
the bitwise equality of entirely complete-condition trajectories.
Blind-frame differences reflect inherited state: both arms still hold
the preceding box and perform no model inference on00 frames.

All four outage trajectories first differ at frame3. Template write
decision differences occur on leftredflag-lsz at300/414/684/1068 and
bikefromnorth at144/162/204. The latter has16 native writes versus15
corrected. Equal write decisions on the other two videos do not imply
equal template contents, because crop locations can differ.
These counts alone cannot separate template effects from crop recurrence;
that would require a distinct intervention.

## Decision

Keep the registered16-video correction policy, parameters and source hashes
unchanged. Do not choose a TIR-only gate from these post-hoc four-video
results, discard a failure, or equate bootstrap frame strata with new-seed
evidence. The fresh replication remains the next prospective decision.
This audit supports a bounded mechanism diagnosis, not official attainment
or one of the required three validated paper modules.
