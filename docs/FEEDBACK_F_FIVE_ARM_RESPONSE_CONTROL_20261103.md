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

Raw CPU reports are retained in the local Git repository at
`reports/server28_feedback_five_arm_checks_20261103/`. This is not a
deployed report directory under the server project root. The 300 planned
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

## Deployment Boundary Checked On 2026-11-04

A read-only check at approximately09:31UTC confirmed that the server project
root `/home/gaob/Miss-RGBT/research/feedback_f/` still contains the older
four-arm implementation: `single`, `none`, `shallow`, `deep`. The five-arm
amendment exists in the isolated staging directory
`/tmp/codex_feedback_f_five_arm_20261103/research/feedback_f/`, including
`deep-no-response`. Do not treat the server-root entrypoints as the amended
implementation or infer deployment from the local Git history.

Three key staged files have the same SHA256 as the local tracked sources:

| File | SHA256 |
| --- | --- |
| protocol_training.py | 272b0af34ec8bafd9adc7d8223939685d6e7f0c3773c3b2eee1456f1e021aa7b |
| protocol_smokes.py | 58352b0d5294f113afc344e3d0466449ff1024a59abe4157ad3277e2c1875e3d |
| protocol_parent.py | 41b25b22009b530cbcd3ee0a56dd59994098b458ec28e85d0b281c1a9a6644b8 |

This three-file comparison is not a whole-package deployment audit. When
the actual P and M1 gates allow F preparation, freeze and verify the entire
five-arm package and its dependencies in the designated isolated runtime,
then require all five matched real-image smokes before formal training.
The staging files alone do not authorize a launch; their continued existence
in `/tmp` must be checked again at preparation time.

The local report directory above exists. The corresponding path under
`/home/gaob/Miss-RGBT/reports/` was absent at this check; the staging root
contains `training.json`, `runner.json`, `validation.json`, `parent.json`
and `verify_retry.json`. Report identities have not been compared here.
No server source, active runtime, configuration, process, or checkpoint was
changed by this check.
