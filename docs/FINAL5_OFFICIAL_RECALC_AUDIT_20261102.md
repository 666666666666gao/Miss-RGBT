# Final-Five Official Recalculation Audit

Completed on server28 at 2026-11-02 05:46 CST.

## Coverage

Independently rechecked all 40 registered final-five official evaluations:

- Fixed A Random and Utility, epochs 16-20, each in `complete` and `miss`.
- Batch10 Random and Utility, epochs 13-17, each in `complete` and `miss`.

Each run passed the isolated `verify_official` check in
`verify_recovery_campaign28.py`. The audit checked the checkpoint file SHA256,
`limit=null`, matching run/completion scope of 479 sequences and 337,353
frames, all 479 receipt identities, bbox shapes and hashes, ground-truth and
mask hashes, frozen runtime/evaluator source hashes, and recomputed toolkit
metrics and curves. This audit did not invoke `verify_development`, evaluate
val97, or use val97 scores.

All 40 runs passed. The 20 epoch checkpoints remain present. No checkpoint
selection or cleanup was performed using official test scores. GPU0-3 were
idle at the end of the audit, with 250 W power limits unchanged.

## Retained Checkpoints

Each file exists and was rehashed. Its digest and exact path match both
condition run manifests for that epoch.

| Candidate | Epoch | SHA256 |
|---|---:|---|
| Fixed A Random | 16 | `28432e2183dff61304d0aa0b4015f757bbdec8a7ea43415baa845a424e6c957c` |
| Fixed A Random | 17 | `d85413a045a259d61f1747182b07a1da0608b727272b8c83b44c7c23e45f0682` |
| Fixed A Random | 18 | `929d6538ae350a5dcfb7198620c46c49fa77f752e4c177bd2fd5a2a300da7dc1` |
| Fixed A Random | 19 | `0444578e5d157e8801e537a166c83f4973ed0f56846374b3fd139f7da4ffe8e9` |
| Fixed A Random | 20 | `6f014866e45393edd0c32594a916d1b0e677f1c6bea8f2d898daa7938f4e5e39` |
| Fixed A Utility | 16 | `42ed78bb821a34f2695423a3a5d8db3ea0af0ab7820022b78588378fea803c8d` |
| Fixed A Utility | 17 | `c83adcf220273e451631bd58f6ac17e07450b6d2b9e8e6e9f2add8bc0d31ddca` |
| Fixed A Utility | 18 | `1f907ae2238d94a509b4924dcb051699ff97c90d50f75a2998c4d10cd234f492` |
| Fixed A Utility | 19 | `9a5683045ec8ba7b3560856c20a52ba50e8def85fe12988d25b8dc629825dfbe` |
| Fixed A Utility | 20 | `19c008d03f261ebda37c89eff63a0b959bcb6795c0da548e761ef12c91878170` |
| Batch10 Random | 13 | `23d78b1b750e1353a43ff0a7b4449263b09a9f322f2a0cdf850ad672cfb8959e` |
| Batch10 Random | 14 | `6948096562b0bec8fc9ff85ea74bd28b187299343ddc58ac58760fa05cfe0149` |
| Batch10 Random | 15 | `88399efa2a5db68ad941d83cc84b4ca7d5b812bb1ef16b1f8b0199b71208228e` |
| Batch10 Random | 16 | `16dc20c62d8464c884147aecaa732f5acbdbe5529daf1d55867297567a0da932` |
| Batch10 Random | 17 | `995679a1fbf061da0915984a295596072f2ee277894aa571ba97d63f4011695a` |
| Batch10 Utility | 13 | `6648160c353ecaee2c1205f9eeda95696afec516f1598d0e0199d7fb54befc07` |
| Batch10 Utility | 14 | `d5754b3532352ea0c50a1166520b3c6f17e7cf4e66485344cf6cfae606bf45a5` |
| Batch10 Utility | 15 | `d987852253e3b49a1037290d7723c2e4d124c167b8e99a18823c8feba7f32ee0` |
| Batch10 Utility | 16 | `0d0eb55270a2aabed8a89a14c40b5cbe2d508f163ce36946cd264c4d36c174dc` |
| Batch10 Utility | 17 | `fbeaf71a9b4bbfb46e926251b34c7fb1960c4ba41d3da75745342638fd4dc062` |

## Audited Per-Metric Maxima

| Condition / dataset | Metric | Candidate | Score |
|---|---|---|---:|
| Complete / LasHeR | PR | Batch10 Utility ep17 | 75.751492 |
| Complete / LasHeR | NPR | Batch10 Random ep17 | 72.171251 |
| Complete / LasHeR | SR | Batch10 Random ep17 | 60.419370 |
| Complete / RGBT234 | MPR | Fixed A Random ep16 | 91.043051 |
| Complete / RGBT234 | MSR | CADTrack E0 | 66.845228 |
| Miss / LasHeR | PR | Batch10 Utility ep17 | 70.397396 |
| Miss / LasHeR | NPR | Batch10 Utility ep17 | 65.127562 |
| Miss / LasHeR | SR | Batch10 Utility ep17 | 55.789882 |
| Miss / RGBT234 | MPR | Fixed A Random ep20 | 86.184864 |
| Miss / RGBT234 | MSR | Fixed A Random ep19 | 61.660594 |

The strict LasHeR-Miss PR/SR goals (69.3/54.4) are exceeded by
1.097396/1.389882 points. RGBT234-Miss MPR/MSR remain short of 88.1/64.3 by
1.915136/2.639406 points.

Against the recorded complete-input E0 baseline, the best observed gains are
LasHeR PR/NPR/SR `+0.114989/+0.398414/+0.293656` and RGBT234 MPR/MSR
`+0.365917/+0.000000` points. The approximate +2-point complete-input goal
is not met. These are per-metric post-hoc maxima across official test
evaluations, not an unbiased checkpoint-selection rule or a single model
achieving every maximum.

## Next Gate

Do not clean final-five checkpoints based on official-test ranking. The
registered recovery verifier also evaluates val97 and must not be launched
under the user's no-val97 instruction. Continue with a no-update,
train882-only mechanism diagnostic; advance to matched training only if the
diagnostic clears its predeclared effect, sham, and complete-parity criteria.
The three paper modules remain unvalidated.
