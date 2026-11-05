# Last-observed modality template diagnostic

## Completed result

GPU2PID1011087 completed351.701seconds normally. All32 fresh128frame
clips/1088scored frames completed,377native writes including165incomplete
writes. Retained/native/complete-only predictions use identical native prior
boxes and crops; these are same-history shadow interventions, not closed-loop
policy evaluation or official benchmark metrics.

| Sequence-macro IoU contrast | Mean | 95% bootstrap interval |
|---|---:|---|
| Retained minus native, RGB missing | +0.008188 | [-0.003180, +0.021454] |
| Retained minus native, TIR missing | +0.005576 | [+0.001102, +0.012710] |
| Retained minus native, pooled missing | +0.006703 | [+0.000943, +0.013224] |
| Retained minus complete-only, pooled missing | -0.000440 | [-0.001530, +0.000563] |
| Retained minus native, complete current frames | +0.001598 | [+0.000318, +0.003017] |

All32sequences contribute to each readout. Registered advance=false.
The two directional gains do not reach.01; RGBmissing uncertainty spans
zero, and pooledmissing superiority over complete-only is not established.
Complete-current protection passes, but it cannot substitute for the failed
missing/control gates. These small positive native-history effects do not
establish distinct value of per-sensor retention or closed-loop gains.
Close this exact policy/control experiment; no closed-loop promotion,
training expansion, audit retuning or module claim. Complete-only is a
matched control, not independently approved for promotion.

Server and local independent arithmetic/provenance analyzers passed:
all1088boxIoUs,377event/source-frame transitions, exclusion/report hashes,
bootstrap/gate,96native repeat shams,731complete-history parity rows.
Server additionally verified all listed source/checkpoint hashes, actualGT/
visibility, seeded clip/schedule reconstruction, and repeated the first
clip from raw images:34scored rows and18native writes exactly matched,
including template hashes, boxes/confidences and source-frame identities.
Replay uses the same policy worker/native tracker, not independently written
neural inference or full-dataset image replay. State/gradient assertions were
also rechecked in that replay. This is not external scientific acceptance.

ReportSHA256:
`b0e0cbb393f4e449dde5df845aa104898b7a5cf55cf8cd03a6a086d7bffb7963`.
Evidence reports/server28_last_observed_template_20261102/. Worker source
unchanged at f98e64eefea7d54451c90516ccfa493358f3e6d259a2885d55bcaccca74ca569.
No optimizer/weights created or deleted, no val97/official tests, no new
official metrics. FinalGPU0-3 temperatures38/37/45/37C,all250W,allidle.
Full goal remains unmet: complete-input approximately+2, RGBT234-Miss
targets and three experimentally defensible modules are unproven.

## Question and prior-art boundary

Native tracker replaces its whole6-channel dynamic template at accepted
writes and records its availability. A single-modality write therefore
replaces an earlier observed template in the absent channel with a masked
unobserved channel. The original complete template is retained. Backbone
streams independently process their own templates until late fusion.
Does retaining the last actually observed dynamic template for each sensor
help under the SAME native crop/write history?

This is distinct from freezing all writes, requiring complete-frame writes,
reordering templates, or erasing absent search features. Static earlier
diagnostics found small missing-template effects; useful trajectory-level
benefit is unproven. A positive result would still need fresh closed-loop
tracking and matched training before any module claim.

MPTrack official source at70ae6b768a5fad920fd4bba7e170aaa3d75cc77d already
stores initial/synchronous/asynchronous RGB/asynchronous TIR template slots
and conditionally updates them using reliability. Root LICENSE is MIT.
Source inspected read-only:
lib/test/tracker/ostrack_st1_quadra_zfusetrans_3ioubranch_rgbtpos_regwcls_3head_specific2anotherall_wsq.py.
This establishes an asynchronous-template prior-art boundary, not evidence
for our mechanism or performance. No code/checkpoints/dependencies copied.
Checkout hit Windows long-path errors for unrelated experiment YAMLs; Git
objects and the inspected tracker/LICENSE were available at the pinned commit.

https://github.com/liulei970507/MPTrack/tree/70ae6b768a5fad920fd4bba7e170aaa3d75cc77d

## Fixed protocol

- Original E0/base and availability-wrapper checkpoints, unchanged warm
  server28 Python3.7/Torch1.12 environment. Frozen weights/no gradients,
  no optimization, no trainable parameters or threshold changes.
-32 fresh128frame clips from existing705 fitting names only, seed59101.
  Exclude all prior tuple/burst/crop diagnostic frames and all image-training
  fitting/development/amended-audit clips. Metadata-only preflight found140
  eligible128frame fitting sequences (175 at96), before any new predictions.
  All frames finite positive visible GT; initialization complete.
- Fixed IID availability .4/.3/.3 complete/RGBmissing/TIRmissing afterinit.
  No00frames in this diagnostic; no inference claim for blind outages.
- Native tracker alone commits all crops, boxes, writes and template history.
  At every4th call plus the first3calls, shadow-predict from identical native
  priorbox and currentimage/state with alternative dynamic templates.
  Shadow box/confidence/writes are discarded; restore state/list references
  before native call. No GT crop, reset, selection, or inference afterinit.
- Last-observed bank: starts at initial template. At each NATIVE accepted
  write, copy only currently observed3channel slice from its actual
  prediction-centered candidate crop. Hold the other sensor's last actual
  observed tensor. Both bank availability bits are1: both contain actual
  earlier observations. Keep per-sensor source frame ids and tensor hashes.
  No fabricated reconstruction, age threshold, interpolation or pixel leak.
- Control complete-only bank: starts at initial template; at same native
  write events, accept candidate only if both sensors observed. Both bits1.
  Same2template slots, same network/crop/Hann computation per shadow call.
  It holds both sensors at incomplete writes, unlike the target bank.
- Persist native/shadow boxes/confidences, per-row template provenance,
  current GT only for scoring, schedules, image hashes and write events.
  First3calls independently repeat native as sham. Whenever native dynamic
  availability is11, banks equal native dynamic tensor bitwise and all
  shadow outputs must match native exactly. First3calls have no priorwrite.
  Assert no native state/list mutation after restoring each shadow.
- Bootstrap10000 seed59301 clustered bysequence. Readouts: retained-native
  and retained-complete-only IoU on each current missing state; pooledmissing;
  complete current frames inside mixed trajectories. They are internal
  native-history one-step IoUs, not independent evolving policy rollouts.

## Gate and execution

Advance only to a NEW fresh train882 closed-loop diagnostic if retained-E0
meanIoU>=.01 and lower95%>0 for BOTH missing states, all32seq contribute,
pooledmissing retained-complete-only lower95%>0, and complete-current-frame
retained-native mean>=0/lower95%>-.01, all32seq. Parity/source/provenance/
finite/state checks required. Failure closes this exact policy/control;
no retuning its cohort or presenting it as a paper module.

GPU2 idle<500MiB,temp<85C,existing250W; batch1 sequential native history
is intrinsic, do not duplicate examples to fill VRAM. Expected5-10min,
observe180-300s milestones, refine ETA from clips. No official masks/test,
val97, RGBT234 tuning, checkpoint deletion or modifications to frozen runtime.
Save immutable protocol/source hashes before execution. Server/local
independent box/provenance/exclusion/bootstrap verification and firstclip
native-image replay follow completion. Complete benchmark gains and all
three modules remain unmet regardless of this diagnostic outcome.
