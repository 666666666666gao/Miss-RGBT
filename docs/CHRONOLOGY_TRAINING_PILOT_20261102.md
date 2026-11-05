# Matched Temporal Mask Training

The preceding frozen diagnostic supports an age/missingness binding effect,
not an improved policy. Test training efficacy with matched native versus
chronological assignment, using new tuples and unchanged native inference.
This is a data-contract control, not a novel three-module contribution.

## Registered Setup

- Original E0 SHA d20c8641e4263df3ceadf9e430d0ac96c181c13ad482bca28c2dddef592a8bb8.
  Native final-two-block attention-qkv rank5 adapters,30720 parameters each.
  Identical initialization, optimizer and permutations using existing
  probe_tail_lowrank helpers. Baseline parameters/BN frozen,eval mode.
- Existing train882705/88/89 names, fresh256/128/128 tuples,seeds53101-3.
  Exclude exact prior tuples including chronology diagnostic. Reused names
  are not untouched sequences; E0 historical exposure uncertified.
- Both fitting caches use the same sampled images/augmentations/identities.
  Native: newercomplete/olderdynamic. Chronological: oldercomplete/newerdynamic.
  Reverse only template_images for chronological cache; original ids remain
  newer,older in identity manifests, with explicit transform metadata.
  Per-sample original-order tensor hashes must match across fitting arms,
  checking actual images/augmentations/search boxes before reversal.
  No change to source data loader. All9 states,2304rows per fitting arm.
- Both development/audit use chronological input,1152rows each. First
  search only. Audit sampled only after both arms stop/select. No val97,
  official test data/masks, RGBT234 or patch masking.
- Cache detached preblock10 RGB/TIR float32 tokens, target/native boxes/
  loss/IoU and input digests. Native tail replay exact, batch16.
  This adapts final two blocks only; not full backbone training.
- Existing measured memory rule: backward probe16 in attention/MLP,
  conservative max of both to choose largest multiple16 dividing2304,
  max256,under75%linear allocated-memory estimate. Same batch both arms.
  AdamW1e-4 atbatch32 scaled linearly,wd1e-4,clip.1,max20,min10,
  LR x.1 from7,patience3 without.2% dev loss improvement after7.
  Both choose min9-cell chronological development loss among final5.
  Retain all epoch weights until verification. One optimizer seed.
- OneGPU2 runs both sequentially with same cache-image RNG and order.
  Check occupancy<500MiB,temp<85C,cap250W;warm runtime unchanged.
  Estimate10-20min; refine from actual cache/epoch speed,checks180-300s.

## Gate

Chronological-trained minus E0 sequence-macro crop-IoU >=.01 and positive
95%lower bound in complete11_11 and both missing search directions pooled
over3template states (10000sequence bootstrap,seed53301).
Chronological-minus-native-trained overall9cells must have positive lower
bound(seed53302). Original state/gradients,parity,finite losses,tuple
exclusions/source/cache/final5 hashes and selected-tail replay must pass.
Pass only authorizes fresh train882 closed-loop test. Fail closes this
exact intervention; do not tune the audit or run official evaluation.
No claim that chronology is the sole benchmark failure cause.

## Result

Completed in704.600s on GPU2, PID589499. Both arms stopped at10epochs/
120updates; both final-five windows are6-10. Common development loss chose
nativeepoch6 and chronologicalepoch9 before sampling the delayed audit.
Actual audit:128tuples/69sequences/1152rows per arm. Exact original-order
input digests matched across all256 fitting tuples.

| Sequence-macro crop-IoU contrast | Mean | 95% CI |
|---|---:|---|
| Chronological-trained minus E0, complete11_11 | +.003912 | [-.013950,+.029749] |
| Chronological-trained minus E0, RGB missing | +.010976 | [-.008068,+.029101] |
| Chronological-trained minus E0, TIR missing | +.009991 | [-.002026,+.023430] |
| Chronological minus native training, all9cells | -.003260 | [-.007479,+.000038] |

Registered advance=false. These observations do not establish a benefit
from changing the temporal missingness assignment with this adapter budget.
The preceding frozen diagnostic's detectable difference is not evidence
that this training intervention fixes it. Do not expand this exact pilot,
retune its audit, launch official evaluation, or count it as a paper module.
This does not establish that all chronological/full-backbone training fails.

Server and local analyzers verified actual name partitions, strict template
chronology, prior exact-tuple exclusions, input pairing, box-IoU arithmetic,
stopping/final-five selection, bootstrap/decision and report/protocol hashes.
Server also checked base/source/fourcache hashes and all10 final-five
weights, finite30720parameters each. Separate-process cached-tail replay
matched1152 originalE0 and2304 selected-adapter records exactly. Replay
uses shared tail/measurement implementation, not independent image/backbone
replay or external review. Producer reports original state unchanged and
gradients absent; these checks are producer assertions.

Report SHA256:
`645a7665abf1479a9620dd04fd9ffef6f70b498e6274802d360a48512999a0ce`.
Evidence: `reports/server28_chronology_training_20261102/`.
All weights retained. No val97, official evaluation or cleanup.
Official maxima unchanged; complete+2,RGBT234-Miss and three-module goals
remain incomplete. A next method needs distinct prospective evidence,
not further fitting to this audit.

### Launch Observation

GPU2 PID589499 launched2026-11-02T04:02:16Z. At elapsed5:09,
native fitting reached epoch3/update36. Both fitting caches and common
development cache completed, and exact original-order input pairing passed.
Measured batch192/LR0.0006,12updates per epoch; native epoch time about20s.
Observed GPU22156MiB/73C/250W. Remaining estimate at that snapshot was6-9min.
After verification GPUs0-3 were idle,38/38/44/37C,all250W.
