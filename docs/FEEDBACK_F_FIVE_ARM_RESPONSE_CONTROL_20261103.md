# F Five-Arm Response-Control Amendment

## Motivation

The previous four-arm comparison changed both the depth of the search
representation and whether the first-pass response map was available.
Consequently, `shallow` versus `deep` could not attribute a gain to response
conditioning independently of deeper features.

The isolated F protocol now adds `deep-no-response`:

| Arm | Shallow features | Deep features | First-pass response map |
|---|---:|---:|---:|
| `single` | Native single pass | No feedback pass | No |
| `none` | Native two-pass control | No feedback correction | No |
| `shallow` | Used | Zeroed | Zeroed |
| `deep-no-response` | Used | Used | Zeroed |
| `deep` | Used | Used | Used |

The new arm keeps the same model shape and feedback initialization as `deep`;
only the response-map input is zeroed. The intended contrasts are
`shallow` versus `deep-no-response` for deep-feature contribution and
`deep-no-response` versus `deep` for response-map conditioning. These are
planned controlled contrasts, not causal or efficacy findings.

All five arms remain paired on their chosen curriculum, initialization,
training samples, masks, optimization settings and checkpoint policy.
The formal campaign still requires the selected P parent, an accepted or
explicitly closed M1 decision, frozen-source checks and five matched
real-image smokes. This amendment does not authorize preparation, training,
or evaluation.

## Verification

On Server 28, the modified F sources were staged under
`/tmp/codex_feedback_f_five_arm_20261103`, separate from active runtimes and
campaigns. Checks used the existing Python 3.7 / PyTorch 1.12 environment
with `CUDA_VISIBLE_DEVICES=''`.

- `compileall` passed.
- Training contracts passed 90 curriculum/mode/epoch cases, six mask/state
  roundtrips, five actual OSTrack CPU initializations, and synthetic
  response/deep-feature sensitivity checks.
- Runner contracts passed 30 curriculum/mode cases, 720 microsteps and 360
  optimizer updates.
- Validation checks passed for five modes across three availability states,
  including actual OSTrack CPU forwards on synthetic tensors.
- Saved-evidence checks passed 30 cases and 720 synthetic steps.
- Five-arm smoke-gate tests passed 2 tests; official-plan tests passed 2
  tests and enumerated 300 destinations across six alternative curricula.
- Official controller/report tests passed 8 tests, using mocked evaluation
  and verification results. Parent-gate fixtures passed with mocked official
  verifiers.

Raw CPU reports are in
`reports/server28_feedback_five_arm_checks_20261103/`. The 300 planned
destinations cover six possible curriculum choices; after a curriculum is
selected, one five-arm formal campaign requires 50 full evaluations
(five arms x five retained epochs x Complete/Miss). Each evaluation must
cover 479 sequences / 337,353 frames, including LasHeR 245 / 220,703 and
RGBT234 234 / 116,650. Checkpoints are retained; official test metrics do
not select weights.

These checks establish code contracts only. No real-image smoke, F training,
learned feedback, full tracker GPU check, official metric, target attainment
or module promotion is established. The separate `check_tracker.py` path
uses CUDA and was not run while all four GPUs were occupied. No runtime,
configuration, active queue, checkpoint or dataset manifest was changed.

## Remaining Work

The A training receipts and saved-evidence `verification_v2.json` are
complete, but the A official queue remains `OFFICIAL_RUNNING`; its formal
campaign verification has not passed. R1/R2 training and the RPPM queue are
also still live or gated. No next-stage experiment is authorized by this
amendment. The benchmark targets and the three-module efficacy claim remain
unproven.
