# Matched Window-Aware Regression Supervision

## Evidence And Boundary

Current task_loss regresses the raw response argmax; actual tracker applies
Hann before selecting regression outputs. Existing fixed-candidate diagnostic
records allow an exploratory comparison without new neural runs:
on121 fitting sequences, raw/Hann peaks differ22/128 complete,146/384
RGB-missing and134/384 TIR-missing rows. Hann-minus-raw sequence-macro
IoU is-.003605CI[-.025610,.016731],-.035688CI[-.061237,-.012040],
and-.039545CI[-.072144,-.010324], respectively (10000draws,seed54101).
This post-hoc crop evidence motivates training alignment only. It is not a
closed-loop improvement, proof of a bug, or permission to remove Hann.
Prior crop/global-search evidence also found Hann useful in other settings.
Ordinary train/inference alignment is a control, not an established novel
paper module. The unsuccessful target-peak head pilot supplied GT training
indices; this proposal uses predicted raw versus predicted Hann indices.
Source: reports/server28_candidate_quality_20261102_v2/report.json,
SHA e774a5ff4b4cdacffae53c286c25bed0116d0feb6175a777f66a3061d4a35cda.

## Registered Setup

- Frozen original E0 SHA d20c8641e4263df3ceadf9e430d0ac96c181c13ad482bca28c2dddef592a8bb8.
  Two identical pretrained native CENTER head copies,6474117 parameters.
  Native backbone/fusion and all BN statistics frozen; no new architecture.
- Existing train882705/88/89 names,256/128/128 fresh tuples,seeds54101-3.
  All9 template/search availability cells, original template order, first
  search only. Same cached fused features/images/identities in both arms.
  Exclude all previous exact tuples including both chronology experiments.
  Reused names are not untouched sequences; E0 exposure uncertified.
- Raw arm trains original raw-peak task_loss. Hann arm replaces only the
  predicted boxes with native cal_bbox(score*Hann,size,offset) before the
  same2GIoU+5L1+focal task_loss. Focal still uses original raw score map.
  Both evaluation paths always use Hann; no GT enters inference.
- Batch64 actual backward/AdamW probe,75% linear allocated-memory estimate,
  largest divisor multiple16 of2304,cap256. This finite cached-head control
  retains at least9 updates/epoch; do not replicate data or force a
  near-full-cache batch just to fill24GB. Actual memory peaks recorded.
  LR1e-5 atbatch8,linear scaling,AdamW wd1e-4,clip.1,LRx.1 from7.
  Max20/min10epochs,patience3 without.2% common development loss improvement
  from7. Same permutations; choose minimum Hann task loss among final5.
- Audit sampled only after both stop/select. One optimizer seed. Preserve
  all weights. No val97,LasHeR test,RGBT234 or official missing masks.

## Advance Criterion

Hann-trained minus E0 macro crop-IoU >=.01 with positive95% lower bound in
each missing search direction,averaged across3 template states,bootstrap54301.
Hann-minus-raw training six-missing-cell contrast lower95%>0,seed54302.
Each complete-search cell11_11/01_11/10_11 must have nonnegative mean gain
and lower95%>-.005,seed54303.10000 sequence-bootstrap draws.
Verify actual partitions/tuple exclusions,source/base/cache/final5 hashes,
finite6474117parameter weights,unchanged head BN,baseline-state/gradients,
box-IoU arithmetic,stop/select decisions and separate cached-head replay.
Pass authorizes only fresh train882 prediction-driven closed-loop testing.
Fail closes this pilot; no audit retuning or official evaluation.
The complete+2 and benchmark targets remain separate unmet requirements.

GPU2 warm runtime,occupancy<500MiB,temp<85C,250W cap. Estimate3-8min,
refine from actual cache/epoch time;180-300s meaningful checks.

## Result

Corrected GPU2 PID679342 completed185.454seconds. Batch256/LR0.00032,
9updates/epoch. Both arms stopped10epochs/90updates; both selectedepoch6
from final-five6-10 using common Hann development task loss, before audit.
Audit128tuples sampled66 actual sequences,1152rows per head and baseline.
Peak allocated raw1559207936/Hann1559994368 bytes,reserved1931476992 each.
This finite cached-head control does not saturate24GB and is not full
image/backbone training or evidence of full-tracker convergence.

| Sequence-macro crop-IoU contrast | Mean | 95% CI |
|---|---:|---|
| Hann-trained minus E0,RGB missing | +.037455 | [.020163,.056675] |
| Hann-trained minus E0,TIR missing | +.043352 | [.018517,.072476] |
| Hann minus raw training,missing cells | +.000731 | [-.006815,.009476] |
| Hann-trained minus E0,complete11_11 | +.005781 | [-.009436,.025590] |
| Hann-trained minus E0,recovery01_11 | +.004673 | [-.011939,.024269] |
| Hann-trained minus E0,recovery10_11 | -.006195 | [-.022286,.011793] |

Registered advance=false. Both missing-direction gains over frozen E0
pass, but the matched raw-supervision control is not reliably exceeded.
All complete-search protection intervals miss the registered-.005 lower
bound,and10_11 also misses the nonnegative point criterion. Thus neither
the specific Hann-alignment contribution nor complete/recovery protection
is established. Do not describe positive E0 comparisons as proof of the
proposed mechanism. The raw arm is a control,not approved for promotion.
Do not expand/retune this exact pilot,run official evaluation,or count a
paper module. These results do not rule out all full-model alignment methods.

Server and local analyzers recomputed partitions,exact prior-tuple exclusions,
3456saved box IoUs,stopping/final-five selection,bootstrap/decision and source
exploratory window effects. Server checked original checkpoint/source/
threecache/all10finite6474117parameter final-five weights and unchanged
native head BN buffers. Separate-process native-head reconstruction replayed
baseline+selected heads3456rows exactly,with independently written Hann
box decode. Head layers and task_loss remain shared; no independent image/
backbone replay or external review. Baseline-state/gradient assertions were
checked by the producer,not independently observed during optimization.

Report SHA256:
`95200226fd20c340cf1e124fd3d704a9bed2ff9f52bfeee8f2cf287869c00612`.
Analyzer SHA256:
`11caa22cf7d8b0b3ac8153047d8602bab1c758f0ac134d3a66e7a19df2860d15`.
Small evidence `reports/server28_window_training_20261102_v2/`.
All initial/retry weights retained. No val97,official evaluation or cleanup.
After verification GPUs idle38/38/51/37C,all250W. Official maxima unchanged.
Complete+2,RGBT234-Miss thresholds and three validated modules remain unmet.

### Initial Reporting Failure

Initial PID654595 launched2026-11-02T04:29:02Z and terminated after
training/selection/audit caching,while building the source manifest. The
worker incorrectly assumed diagnose_joint_utility_sets.py was next to
it; the imported helper actually lives inside runtime/scripts. No report
or completion receipt was produced. Original log,launch/protocol receipts,
weights/caches and failed worker source are preserved. No original audit
scores were inspected for tuning. Corrected worker records the actual
imported module.__file__,and retries in exclusive output
server28_window_training_20261102_v2 with identical seeds/settings/gate.
This is a reporting-path correction,not a new independent experiment.
