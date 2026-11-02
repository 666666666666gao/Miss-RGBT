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
