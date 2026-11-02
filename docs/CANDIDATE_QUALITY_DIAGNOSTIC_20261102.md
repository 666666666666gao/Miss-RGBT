# Fixed-Candidate Quality Diagnostic

## Question And Limits

Can changing the ranking of a small set of native high-response locations
improve box quality without changing any regression map? Source motivation
and the distinct failed target-index regression control are documented in
`QUALITY_ALIGNMENT_SOURCE_REVIEW_20261102.md`. Ordinary task-aligned scoring
is prior art. No proposed module is considered validated.

This is a GT-IoU oracle diagnostic, not an inference algorithm or a tracking
benchmark. A positive result shows candidate headroom, not learnability.
Prior raw-versus-Hann experiments found Hann removal harmful; keep the
native Hann prior and separately report raw decoding for context.

## Registered Data And Model

- Original frozen E0 epoch20, SHA256
  `d20c8641e4263df3ceadf9e430d0ac96c181c13ad482bca28c2dddef592a8bb8`.
  No optimizer, parameter update, learned ranking or LoRA adapter.
- Train882 fitting705 names only; names with index%5 !=0.128 fresh tuples,
  seed50101, first search only. Exclude exact identities from the previous
  fusion/head/geometry/position/KV/late-fusion/context/tail pilots.
  These names were used before; new tuples are not independent new
  sequences, and E0 historical exposure remains uncertified.
- All9 states: complete initial template, three dynamic-template states
  crossed with complete/RGB-missing/TIR-missing searches.1152 rows.
  No val97, RGBT234, LasHeR test frames or official missing masks.
- Batch16, float32 eval mode, deterministic CUDA settings. Record original
  model digest before/after, absent gradients, identities, input digests,
  all native score/size/offset/box maps and target boxes in a server cache.

## Fixed Candidate Protocol

Decode all256 native boxes from unchanged size/offset maps. Gathered boxes
at the raw and Hann peaks must be exactly equal to the native CENTER
decoder outputs. Dense-box IoU must match native peak IoU within
atol2e-6/rtol1e-6.

Primary candidates are exactly the top8 Hann-weighted responses, with
descending stable order (lowest flat index wins ties). GT is never used to
construct this set. Store all8 indices, raw/Hann scores, boxes and IoUs.
The primary endpoint is best candidate GT-IoU minus native Hann-peak IoU.
Uniform top8 mean is the expected random-candidate control, not a learned
ranking. Raw top8 oracle and all256-box oracle are descriptive upper bounds
and cannot authorize promotion by themselves.

Report sequence-macro means and10000-resample95% sequence-bootstrap
intervals, seed50102. Primary categories:

1. Full complete input: only cell11_11,128rows.
2. RGB-missing search: all3 dynamic-template states,384rows.
3. TIR-missing search: all3 dynamic-template states,384rows.

Native Hann/raw results and all9 cells are descriptive. Do not choose k,
alter windows or select sequences from these results.

## Decision

Advance to a separately registered, matched quality-ranking training pilot
only if top8 oracle-minus-native-Hann mean >=.02 and95% lower bound>0 in
all3 primary categories, and parity/state/source/evidence checks pass.
If the gate fails, close this exact candidate set and do not tune it on
these tuples. A pass does not authorize an official test or a paper claim;
a learned selector still needs fresh stopping/audit data and closed-loop
validation. Preserve every existing model weight.

## Execution

GPU2 must be free(<500MiB),temperature<85C,cap250W. Existing2028 environment
is reused unchanged. Estimate1-3minutes; first check after180seconds, then
180-300seconds only if the actual process remains live.
Independent saved-evidence analysis must check candidate membership/order,
box IoU, report/protocol/source/cache hashes, tuple exclusions and gate;
server-map replay must reconstruct candidates from saved dense native maps.
That replay is not independent neural image/backbone inference.

## Result

Corrected GPU2 PID460256 completed44.809seconds.128tuples from121 actual
fitting sequences,1152 state rows. Report SHA256:
`e774a5ff4b4cdacffae53c286c25bed0116d0feb6175a777f66a3061d4a35cda`.
Evidence: `reports/server28_candidate_quality_20261102_v2/`.

Internal sequence-macro IoUs, not tracking benchmark scores:

| Condition | Native Hann | Uniform top8 | GT-best top8 | Oracle gain | Gain 95% CI |
|---|---:|---:|---:|---:|---|
| Full complete11_11 | .789298 | .665058 | .842335 | +.053037 | [.032385,.078354] |
| RGB-missing | .581449 | .477743 | .659580 | +.078130 | [.056767,.103489] |
| TIR-missing | .663633 | .528791 | .742888 | +.079255 | [.052755,.109509] |

All3 categories pass the registered oracle-headroom gate. This authorizes
only a new matched train882 quality-ranking pilot. It does not prove a
learned selector can recover these gains or meet any official target.
The random-candidate expectation is worse than native, so headroom alone
is not a usable selection rule.

Raw peak IoUs are .792903/.617138/.703178; dense256 oracle
.857707/.710661/.778515 for complete/RGBmissing/TIRmissing.
Hann-minus-raw on these augmented static crops is-.003605/-.035688/
-.039545. This does not justify removing Hann: the earlier closed-loop
candidate ranking investigation found that harmful. These endpoints and
cohorts differ; no full-tracking policy selection follows from this audit.

Server saved-evidence verifier passed prior exact-tuple exclusions,
candidate order/membership, boxes/IoU, oracle arithmetic, bootstrap/gate,
report/protocol/base/source/cache hashes and actual fitting names.
Independent NumPy reconstruction from saved native maps checked1152rows/
294912dense boxes, candidate indices/scores/boxes/IoUs and raw/dense
oracles. This is mathematical map replay, not neural image/backbone replay.
Local saved-evidence analysis also passed. Producer reports original state
unchanged, gradients absent, raw/Hann decode exact and native IoU matched.

No optimizer, val97, official evaluation or checkpoint cleanup.
Official best metrics unchanged; all outstanding full-goal requirements
remain. Follow-up registered separately in QUALITY_RANKER_PILOT_20261102.md,
using fresh512/128/128 tuples and delayed audit sampling.

### Execution Correction

First PID425514 stopped on the first batch before saving any result:
installed torch.argsort rejects stable=True. Its output directory, launch
receipt and traceback log remain. Replace that call with NumPy mergesort
on descending scores and move the resulting indices back to the device.
This preserves stable lowest-index tie resolution; no score perturbation,
candidate rule or gate change. Run corrected code in a new v2 directory.
